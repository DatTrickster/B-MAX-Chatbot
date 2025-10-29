import os
import uvicorn
import json
import boto3
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "af-south-1")
DYNAMODB_TABLE_TENDERS = os.getenv("DYNAMODB_TABLE_DEST", "ProcessedTender")
DYNAMODB_TABLE_USERS = os.getenv("DYNAMODB_TABLE_USERS", "UserProfiles")
DYNAMODB_TABLE_BOOKMARKS = os.getenv("DYNAMODB_TABLE_BOOKMARKS", "UserBookmarks")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")

# Initialize AWS clients
try:
    dynamodb = boto3.client(
        'dynamodb',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=AWS_REGION
    )
    
    # Initialize Cognito client
    if COGNITO_USER_POOL_ID:
        cognito = boto3.client(
            'cognito-idp',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=AWS_REGION
        )
        print("✅ AWS Clients (DynamoDB + Cognito) initialized successfully")
    else:
        cognito = None
        print("✅ AWS DynamoDB client initialized (Cognito disabled)")
        
except Exception as e:
    print(f"❌ AWS Client initialization error: {e}")
    dynamodb = None
    cognito = None

# Try to import Ollama, but make it optional for health checks
try:
    from ollama import Client
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
    if OLLAMA_API_KEY:
        client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        )
        ollama_available = True
        print("✅ Ollama client initialized successfully")
    else:
        ollama_available = False
        print("❌ Ollama API key not found")
except ImportError:
    ollama_available = False
    print("❌ Ollama package not installed")
except Exception as e:
    ollama_available = False
    print(f"❌ Ollama client initialization error: {e}")

app = FastAPI(title="B-Max AI Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
user_sessions = {}

class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "guest"

# --- Cognito Helper Functions ---
def get_cognito_user_by_username(username: str):
    """Get user details from Cognito by username"""
    try:
        if not cognito or not COGNITO_USER_POOL_ID:
            print("❌ Cognito not configured")
            return None
            
        response = cognito.admin_get_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=username
        )
        
        user_attributes = {}
        for attr in response.get('UserAttributes', []):
            user_attributes[attr['Name']] = attr['Value']
        
        cognito_user = {
            'username': response.get('Username'),
            'user_id': response.get('UserSub'),  # This is the UUID
            'email': user_attributes.get('email'),
            'email_verified': user_attributes.get('email_verified', 'false') == 'true',
            'status': response.get('UserStatus'),
            'enabled': response.get('Enabled', False),
            'created': response.get('UserCreateDate'),
            'modified': response.get('UserLastModifiedDate'),
            'attributes': user_attributes
        }
        
        print(f"✅ Found Cognito user: {username} -> UUID: {cognito_user['user_id']}")
        return cognito_user
        
    except cognito.exceptions.UserNotFoundException:
        print(f"❌ Cognito user not found: {username}")
        return None
    except Exception as e:
        print(f"❌ Error fetching Cognito user {username}: {e}")
        return None

def get_cognito_user_by_email(email: str):
    """Get user details from Cognito by email"""
    try:
        if not cognito or not COGNITO_USER_POOL_ID:
            return None
            
        response = cognito.list_users(
            UserPoolId=COGNITO_USER_POOL_ID,
            Filter=f'email = "{email}"'
        )
        
        if response.get('Users') and len(response['Users']) > 0:
            user = response['Users'][0]
            user_attributes = {}
            for attr in user.get('Attributes', []):
                user_attributes[attr['Name']] = attr['Value']
            
            cognito_user = {
                'username': user.get('Username'),
                'user_id': user.get('UserSub'),  # This is the UUID
                'email': user_attributes.get('email'),
                'email_verified': user_attributes.get('email_verified', 'false') == 'true',
                'status': user.get('UserStatus'),
                'enabled': user.get('Enabled', False),
                'created': user.get('UserCreateDate'),
                'modified': user.get('UserLastModifiedDate'),
                'attributes': user_attributes
            }
            
            print(f"✅ Found Cognito user by email: {email} -> Username: {cognito_user['username']}")
            return cognito_user
        
        return None
        
    except Exception as e:
        print(f"❌ Error fetching Cognito user by email {email}: {e}")
        return None

# --- DynamoDB Helper Functions ---
def dd_to_py(item):
    """Convert DynamoDB item to Python dict"""
    if not item:
        return {}
    result = {}
    for k, v in item.items():
        if 'S' in v:
            result[k] = v['S']
        elif 'N' in v:
            n = v['N']
            result[k] = int(n) if n.isdigit() else float(n)
        elif 'BOOL' in v:
            result[k] = v['BOOL']
        elif 'SS' in v:
            result[k] = v['SS']
        elif 'M' in v:
            result[k] = dd_to_py(v['M'])
        elif 'L' in v:
            result[k] = [dd_to_py(el) for el in v['L']]
    return result

def get_user_profile_by_user_id(user_id: str):
    """Find user profile by userId (UUID)"""
    try:
        resp = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="userId = :uid",
            ExpressionAttributeValues={":uid": {"S": user_id}}
        )
        items = resp.get("Items", [])
        return dd_to_py(items[0]) if items else None
    except Exception as e:
        print(f"❌ Error scanning for user profile: {e}")
        return None

def get_user_profile_by_email(email: str):
    """Find user profile by email"""
    try:
        resp = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="email = :email",
            ExpressionAttributeValues={":email": {"S": email}}
        )
        items = resp.get("Items", [])
        return dd_to_py(items[0]) if items else None
    except Exception as e:
        print(f"❌ Error scanning for user by email: {e}")
        return None

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = None
        self.cognito_user = None
        self.chat_context = []
        self.last_active = datetime.now()
        self.greeted = False
        
        print(f"🎯 Creating session for user_id: {user_id}")
        
        # Load user profile using Cognito + DynamoDB
        self.load_user_profile()
        
        # Set up system prompt AFTER profile is loaded
        username = self.get_display_name()
        first_name = self.get_first_name()
        self.chat_context = [
            {"role": "system", "content": self.create_system_prompt(username, first_name)}
        ]
        
        print(f"✅ Session created - Name: {first_name}, Profile loaded: {self.user_profile is not None}")

    def create_system_prompt(self, username: str, first_name: str):
        company = self.user_profile.get('companyName', 'Unknown') if self.user_profile else 'Unknown'
        position = self.user_profile.get('position', 'Unknown') if self.user_profile else 'Unknown'
        location = self.user_profile.get('location', 'Unknown') if self.user_profile else 'Unknown'
        
        # Get preferred categories if available
        preferred_categories = []
        if self.user_profile and 'preferredCategories' in self.user_profile:
            if isinstance(self.user_profile['preferredCategories'], list):
                preferred_categories = self.user_profile['preferredCategories']
            elif isinstance(self.user_profile['preferredCategories'], set):
                preferred_categories = list(self.user_profile['preferredCategories'])
        
        categories_str = ", ".join(preferred_categories) if preferred_categories else "Not specified"
        
        return f"""You are B-Max, an AI assistant for TenderConnect. 

CRITICAL RULES - FOLLOW THESE EXACTLY:
1. ALWAYS address the user by their first name "{first_name}" in EVERY response
2. Never use "User" - always use "{first_name}"
3. Be warm, friendly, and professional
4. Remember context from previous messages
5. If you don't know something, be honest and say so

User Profile:
- Full Name: {username}
- First Name: {first_name} (USE THIS IN ALL RESPONSES)
- Company: {company}
- Position: {position}
- Location: {location}
- Preferred Categories: {categories_str}

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMPORTANT: Start your first response with "Hi {first_name}! 👋" and always use their name in responses."""

    def load_user_profile(self):
        """Load user profile using Cognito to get UUID, then DynamoDB"""
        try:
            if not dynamodb:
                self.user_profile = self.create_default_profile()
                return

            print(f"🔍 Loading profile for: {self.user_id}")
            
            # Strategy 1: Check if user_id is already a UUID (starts with Cognito pattern)
            if self.user_id.startswith(('us-east-', 'us-west-', 'af-south-')) or len(self.user_id) > 20:
                # This might already be a UUID, try direct lookup
                profile = get_user_profile_by_user_id(self.user_id)
                if profile:
                    self.user_profile = profile
                    print(f"✅ Profile found via direct UUID: {self.user_id}")
                    print(f"   firstName: {profile.get('firstName', 'NOT FOUND')}")
                    return
            
            # Strategy 2: Query Cognito to get UUID from username/email
            print(f"🔍 Querying Cognito for: {self.user_id}")
            self.cognito_user = get_cognito_user_by_username(self.user_id)
            
            if not self.cognito_user and '@' in self.user_id:
                # If user_id is an email, try Cognito email lookup
                self.cognito_user = get_cognito_user_by_email(self.user_id)
            
            if self.cognito_user:
                cognito_uuid = self.cognito_user['user_id']
                print(f"✅ Found Cognito UUID: {cognito_uuid} for username: {self.user_id}")
                
                # Now lookup in DynamoDB using the Cognito UUID
                profile = get_user_profile_by_user_id(cognito_uuid)
                if profile:
                    self.user_profile = profile
                    print(f"✅ Profile found via Cognito UUID: {cognito_uuid}")
                    print(f"   firstName: {profile.get('firstName', 'NOT FOUND')}")
                    return
                
                # If not found by UUID, try by email from Cognito
                cognito_email = self.cognito_user.get('email')
                if cognito_email:
                    profile = get_user_profile_by_email(cognito_email)
                    if profile:
                        self.user_profile = profile
                        print(f"✅ Profile found via Cognito email: {cognito_email}")
                        print(f"   firstName: {profile.get('firstName', 'NOT FOUND')}")
                        return
            
            # Strategy 3: Fallback - direct email lookup in DynamoDB
            if '@' in self.user_id:
                profile = get_user_profile_by_email(self.user_id)
                if profile:
                    self.user_profile = profile
                    print(f"✅ Profile found via direct email: {self.user_id}")
                    return
            
            # If no profile found, use default
            print(f"❌ No profile found for: {self.user_id}")
            self.user_profile = self.create_default_profile()
                
        except Exception as e:
            print(f"❌ Error loading user profile: {e}")
            self.user_profile = self.create_default_profile()

    def create_default_profile(self):
        """Create a default profile when user not found"""
        default_profile = {
            'firstName': 'User',
            'lastName': '',
            'companyName': 'Unknown',
            'position': 'User',
            'location': 'Unknown',
            'preferredCategories': []
        }
        print("⚠️ Using default profile")
        return default_profile

    def get_display_name(self):
        if self.user_profile:
            first_name = self.user_profile.get('firstName', 'User')
            last_name = self.user_profile.get('lastName', '')
            return f"{first_name} {last_name}".strip()
        return "User"

    def get_first_name(self):
        if self.user_profile:
            first_name = self.user_profile.get('firstName', 'User')
            return first_name
        return "User"

    def update_activity(self):
        self.last_active = datetime.now()

    def add_message(self, role, content):
        self.chat_context.append({"role": role, "content": content})
        if len(self.chat_context) > 15:
            self.chat_context = [self.chat_context[0]] + self.chat_context[-14:]

def get_user_session(user_id: str) -> UserSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    user_sessions[user_id].update_activity()
    return user_sessions[user_id]

def cleanup_old_sessions():
    current_time = datetime.now()
    expired_users = []
    for user_id, session in user_sessions.items():
        if (current_time - session.last_active).total_seconds() > 7200:
            expired_users.append(user_id)
    for user_id in expired_users:
        del user_sessions[user_id]

# Debug endpoint to test Cognito lookup
@app.get("/api/cognito-user/{username}")
async def get_cognito_user_info(username: str):
    """Get Cognito user information for debugging"""
    try:
        cognito_user = get_cognito_user_by_username(username)
        if cognito_user:
            return {
                "username": username,
                "cognito_user": {
                    "user_id": cognito_user.get('user_id'),
                    "email": cognito_user.get('email'),
                    "status": cognito_user.get('status'),
                    "attributes": cognito_user.get('attributes', {})
                }
            }
        else:
            return {
                "username": username,
                "cognito_user": None,
                "error": "User not found in Cognito"
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {
        "message": "B-Max AI Assistant API is running!",
        "status": "healthy" if ollama_available else "degraded",
        "cognito_enabled": cognito is not None,
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health",
            "session_info": "/session/{user_id}",
            "cognito_user": "/api/cognito-user/{username}",
            "find_user": "/api/find-user/{identifier}",
            "all_users": "/api/all-users"
        }
    }

# ... (rest of the endpoints remain the same as previous version)

if __name__ == "__main__":
    print("🚀 Starting B-Max AI Assistant...")
    print("💬 Endpoint: POST /chat")
    print("🔧 Health: GET /health")
    print("📊 Database:", "Connected" if dynamodb else "Disconnected")
    print("🔐 Cognito:", "Connected" if cognito else "Disabled")
    print("🤖 Ollama:", "Connected" if ollama_available else "Disconnected")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        access_log=True
    )
