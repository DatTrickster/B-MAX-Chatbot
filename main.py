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

# Predefined Categories
CATEGORIES = [
    "Engineering Services", "IT Services", "Construction", "Consulting",
    "Supplies", "Maintenance", "Logistics", "Healthcare"
]

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

def get_all_users():
    """Get all users from DynamoDB"""
    try:
        resp = dynamodb.scan(TableName=DYNAMODB_TABLE_USERS)
        users = [dd_to_py(item) for item in resp.get('Items', [])]
        
        # Return only safe fields
        safe_users = []
        for user in users:
            safe_user = {
                'userId': user.get('userId'),
                'email': user.get('email'),
                'firstName': user.get('firstName'),
                'lastName': user.get('lastName'),
                'companyName': user.get('companyName'),
                'position': user.get('position')
            }
            safe_users.append(safe_user)
        
        return safe_users
    except Exception as e:
        print(f"❌ Error getting all users: {e}")
        return []

def get_tender_information(query: str = None, user_id: str = None):
    """Get tender information from database for context"""
    try:
        if not dynamodb:
            return "Database currently unavailable"
        
        # Get all relevant tender fields
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS,
            Limit=10,
            Select='SPECIFIC_ATTRIBUTES',
            ProjectionExpression='title, Category, closingDate, sourceAgency, status, link, referenceNumber, contactEmail, contactName, contactNumber, sourceUrl'
        )
        
        items = response.get('Items', [])
        if not items:
            return "No tenders found in the database."
        
        tenders_info = []
        for item in items:
            tender = {}
            for key, value in item.items():
                if 'S' in value:
                    tender[key] = value['S']
            tenders_info.append(tender)
        
        return format_tenders_for_ai(tenders_info)
        
    except Exception as e:
        return f"Error accessing tender database: {str(e)}"

def format_tenders_for_ai(tenders):
    """Format tender information for AI context with all relevant fields"""
    if not tenders:
        return "No tender data available."
    
    formatted = "RECENT TENDERS AVAILABLE:\n\n"
    for i, tender in enumerate(tenders, 1):
        title = tender.get('title', 'Unknown Title')
        category = tender.get('Category', 'Uncategorized')
        closing_date = tender.get('closingDate', 'Unknown')
        agency = tender.get('sourceAgency', 'Unknown Agency')
        status = tender.get('status', 'Unknown')
        reference_number = tender.get('referenceNumber', 'N/A')
        link = tender.get('link', 'No link available')
        contact_name = tender.get('contactName', 'N/A')
        contact_email = tender.get('contactEmail', 'N/A')
        contact_number = tender.get('contactNumber', 'N/A')
        source_url = tender.get('sourceUrl', 'N/A')
        
        formatted += f"🚀 TENDER #{i}\n"
        formatted += f"📋 Title: {title}\n"
        formatted += f"🏷️ Reference: {reference_number}\n"
        formatted += f"📊 Category: {category}\n"
        formatted += f"🏢 Agency: {agency}\n"
        formatted += f"📅 Closing Date: {closing_date}\n"
        formatted += f"📈 Status: {status}\n"
        
        if contact_name != 'N/A' or contact_email != 'N/A' or contact_number != 'N/A':
            formatted += f"👤 Contact Info:\n"
            if contact_name != 'N/A':
                formatted += f"   - Name: {contact_name}\n"
            if contact_email != 'N/A':
                formatted += f"   - Email: {contact_email}\n"
            if contact_number != 'N/A':
                formatted += f"   - Phone: {contact_number}\n"
        
        formatted += f"🔗 Documents: {link}\n"
        formatted += f"🌐 Source: {source_url}\n"
        formatted += "─" * 50 + "\n\n"
    
    formatted += "💡 I can help you analyze these tenders, find specific opportunities, or provide recommendations based on your preferences!"
    return formatted

def search_tenders_by_category(category: str):
    """Search tenders by predefined category"""
    try:
        if not dynamodb:
            return None
            
        # Map category keywords to predefined categories
        category_mapping = {
            'engineering': 'Engineering Services',
            'it': 'IT Services',
            'technology': 'IT Services',
            'construction': 'Construction',
            'consulting': 'Consulting',
            'supplies': 'Supplies',
            'goods': 'Supplies',
            'maintenance': 'Maintenance',
            'logistics': 'Logistics',
            'healthcare': 'Healthcare',
            'medical': 'Healthcare'
        }
        
        # Normalize category input
        search_category = category.lower()
        mapped_category = category_mapping.get(search_category, category)
        
        print(f"🔍 Searching tenders for category: {category} -> mapped to: {mapped_category}")
            
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS,
            FilterExpression="contains(Category, :cat)",
            ExpressionAttributeValues={":cat": {"S": mapped_category}},
            Limit=8
        )
        
        items = response.get('Items', [])
        tenders = []
        for item in items:
            tender = {}
            for key, value in item.items():
                if 'S' in value:
                    tender[key] = value['S']
            tenders.append(tender)
        
        print(f"✅ Found {len(tenders)} tenders for category: {mapped_category}")
        return tenders
    except Exception as e:
        print(f"❌ Error searching tenders by category: {e}")
        return None

def search_tenders_by_keyword(keyword: str):
    """Search tenders by keyword in title"""
    try:
        if not dynamodb:
            return None
            
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS,
            FilterExpression="contains(title, :kw)",
            ExpressionAttributeValues={":kw": {"S": keyword}},
            Limit=5
        )
        
        items = response.get('Items', [])
        tenders = []
        for item in items:
            tender = {}
            for key, value in item.items():
                if 'S' in value:
                    tender[key] = value['S']
            tenders.append(tender)
        
        return tenders
    except Exception as e:
        print(f"❌ Error searching tenders by keyword: {e}")
        return None

def get_tenders_by_multiple_categories(categories: list):
    """Get tenders that match any of the specified categories"""
    try:
        if not dynamodb:
            return None
            
        all_tenders = []
        for category in categories:
            tenders = search_tenders_by_category(category)
            if tenders:
                all_tenders.extend(tenders)
        
        # Remove duplicates based on tenderId
        seen_ids = set()
        unique_tenders = []
        for tender in all_tenders:
            tender_id = tender.get('tenderId') or tender.get('tender_id')
            if tender_id and tender_id not in seen_ids:
                seen_ids.add(tender_id)
                unique_tenders.append(tender)
        
        return unique_tenders[:10]  # Return max 10 tenders
    except Exception as e:
        print(f"❌ Error getting tenders by multiple categories: {e}")
        return None

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = None
        self.cognito_user = None
        self.chat_context = []
        self.last_active = datetime.now()
        self.greeted = False
        self.total_messages = 0
        
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
        
        # Filter preferred categories to only include valid ones from our predefined list
        valid_preferred_categories = [cat for cat in preferred_categories if cat in CATEGORIES]
        categories_str = ", ".join(valid_preferred_categories) if valid_preferred_categories else "Not specified"
        
        available_categories_str = ", ".join(CATEGORIES)
        
        return f"""You are B-Max, an AI assistant for TenderConnect. 

CRITICAL RULES - FOLLOW THESE EXACTLY:
1. ALWAYS address the user by their first name "{first_name}" in EVERY response
2. Never use "User" or the username - always use "{first_name}"
3. Be warm, friendly, and professional
4. Remember context from previous messages in this conversation
5. If you don't know something, be honest and say so
6. Use emojis occasionally to make conversations friendly
7. Keep responses concise but informative
8. Focus on tender-related topics and procurement
9. NEVER mention or reference other users or their information
10. NEVER share any user profile details beyond what's necessary for the conversation
11. When users ask about tenders, provide relevant tender information and recommendations from the database
12. Use the user's preferences to personalize tender recommendations when appropriate
13. ALWAYS include specific tender details like title, reference number, closing date, agency, and links when discussing tenders
14. Make tender information easy to read with clear formatting
15. Maintain conversation context and refer back to previous discussions when relevant
16. Use the predefined categories when discussing tender types: {available_categories_str}

AVAILABLE TENDER CATEGORIES:
{available_categories_str}

User Profile (FOR CONTEXT ONLY - DO NOT SHARE):
- First Name: {first_name} (USE THIS IN ALL RESPONSES)
- Company: {company}
- Position: {position}
- Location: {location}
- Preferred Categories: {categories_str}

Your capabilities:
- Answer questions about tenders and procurement processes
- Provide tender recommendations based on user preferences
- Explain tender categories and requirements
- Help with tender search strategies
- Provide information about tender deadlines and procedures
- Search and present specific tender opportunities with all relevant details
- Remember previous conversations and maintain context
- Search tenders by specific categories: {available_categories_str}

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMPORTANT: 
- Start conversations naturally based on the user's first message
- Always use "{first_name}" but don't force it into every sentence
- Focus on providing helpful tender information and recommendations
- When discussing tenders, include key details: title, reference number, category, agency, closing date, status, and links
- Remember the full conversation history and refer back to it when appropriate
- Use the predefined categories when categorizing tenders
- Never discuss other users or their data
- Keep responses professional and tender-focused"""

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
        """Add message to chat context with smarter context management"""
        self.chat_context.append({"role": role, "content": content})
        self.total_messages += 1
        
        # Smarter context management - keep more context for better conversations
        # Keep system message + last 20 messages (increased from 15)
        if len(self.chat_context) > 21:  # 1 system message + 20 conversation turns
            # Always keep the system message
            system_message = self.chat_context[0]
            # Keep the most recent messages
            recent_messages = self.chat_context[-20:]
            self.chat_context = [system_message] + recent_messages
        
        print(f"💬 Message added - Role: {role}, Total messages: {self.total_messages}, Context length: {len(self.chat_context)}")

    def get_context_summary(self):
        """Get a summary of the conversation context for debugging"""
        return {
            "user_id": self.user_id,
            "total_messages": self.total_messages,
            "context_length": len(self.chat_context),
            "last_active": self.last_active.isoformat(),
            "user_profile_loaded": self.user_profile is not None
        }

def get_user_session(user_id: str) -> UserSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    else:
        print(f"🔄 Reusing existing session for: {user_id}")
        # Print session info for debugging
        session_info = user_sessions[user_id].get_context_summary()
        print(f"📊 Session info: {session_info}")
    
    user_sessions[user_id].update_activity()
    return user_sessions[user_id]

def cleanup_old_sessions():
    current_time = datetime.now()
    expired_users = []
    for user_id, session in user_sessions.items():
        if (current_time - session.last_active).total_seconds() > 7200:  # 2 hours
            expired_users.append(user_id)
    for user_id in expired_users:
        print(f"🧹 Cleaning up expired session for: {user_id}")
        del user_sessions[user_id]

def enhance_prompt_with_context(user_prompt: str, session: UserSession) -> str:
    """Enhance user prompt with tender context and personalization"""
    database_context = ""
    
    # Enhanced category mapping with predefined categories
    category_keywords = {
        'Engineering Services': ['engineering', 'engineer', 'technical', 'design', 'infrastructure'],
        'IT Services': ['IT', 'technology', 'software', 'hardware', 'computer', 'digital', 'tech', 'programming'],
        'Construction': ['construction', 'building', 'civil', 'contractor', 'build', 'renovation'],
        'Consulting': ['consulting', 'consultant', 'advisory', 'strategy', 'management'],
        'Supplies': ['supplies', 'supply', 'goods', 'materials', 'equipment', 'products'],
        'Maintenance': ['maintenance', 'repair', 'service', 'upkeep', 'support'],
        'Logistics': ['logistics', 'transport', 'shipping', 'delivery', 'supply chain'],
        'Healthcare': ['healthcare', 'medical', 'health', 'hospital', 'clinic', 'pharmaceutical']
    }
    
    user_prompt_lower = user_prompt.lower()
    
    # Check for specific category searches
    matched_categories = []
    for category, keywords in category_keywords.items():
        if any(keyword in user_prompt_lower for keyword in keywords):
            matched_categories.append(category)
    
    # Add tender context if the message is tender-related
    tender_keywords = ['tender', 'tenders', 'procurement', 'bid', 'category', 'recommend', 'suggest', 'opportunity', 'RFP', 'RFQ']
    
    if any(keyword in user_prompt_lower for keyword in tender_keywords) or matched_categories:
        if matched_categories:
            # Search for tenders in matched categories
            specific_tenders = get_tenders_by_multiple_categories(matched_categories)
            if specific_tenders:
                database_context = format_tenders_for_ai(specific_tenders)
                database_context = f"🔍 Tenders in categories: {', '.join(matched_categories)}\n\n{database_context}"
            else:
                database_context = f"📊 No specific tenders found for categories: {', '.join(matched_categories)}. Here are recent tenders instead:\n\n{get_tender_information()}"
        else:
            # Get general tenders
            database_context = get_tender_information(user_prompt, session.user_id)
    
    user_first_name = session.get_first_name()
    
    enhanced_prompt = f"""
User: {user_first_name}
Message: {user_prompt}

{database_context if database_context else ""}

Available Categories: {", ".join(CATEGORIES)}

Remember to address the user by their first name "{user_first_name}" naturally in your response.
Focus on providing helpful tender information and recommendations.
When discussing tenders, include specific details like title, reference number, closing date, and links.
Maintain conversation context and refer back to previous discussions when relevant.
Use the predefined categories when discussing tender types.
"""
    return enhanced_prompt

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    return {
        "message": "B-Max AI Assistant API is running!",
        "status": "healthy" if ollama_available else "degraded",
        "cognito_enabled": cognito is not None,
        "timestamp": datetime.now().isoformat(),
        "categories": CATEGORIES
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "B-Max AI Assistant",
        "dynamodb": "connected" if dynamodb else "disconnected",
        "cognito": "connected" if cognito else "disabled",
        "ollama": "connected" if ollama_available else "disconnected",
        "active_sessions": len(user_sessions),
        "categories": CATEGORIES,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/categories")
async def get_categories():
    """Get all available tender categories"""
    return {
        "categories": CATEGORIES,
        "count": len(CATEGORIES)
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not ollama_available:
            raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
        
        cleanup_old_sessions()
        
        print(f"💬 Chat request received - user_id: {request.user_id}, prompt: {request.prompt}")
        
        session = get_user_session(request.user_id)
        user_first_name = session.get_first_name()
        
        print(f"🎯 Using session - First name: {user_first_name}")
        print(f"📊 Current context length: {len(session.chat_context)} messages")
        
        # Enhance prompt with tender context if relevant
        enhanced_prompt = enhance_prompt_with_context(request.prompt, session)
        
        # Add enhanced user message to context
        session.add_message("user", enhanced_prompt)
        
        # Get AI response
        try:
            print(f"🤖 Sending to Ollama - Context length: {len(session.chat_context)}")
            response = client.chat(
                'deepseek-v3.1:671b-cloud', 
                messages=session.chat_context
            )
            response_text = response['message']['content']
        except Exception as e:
            print(f"❌ Ollama API error: {e}")
            response_text = f"I apologize {user_first_name}, but I'm having trouble processing your request right now. Please try again in a moment."
        
        # Add assistant response to context
        session.add_message("assistant", response_text)
        
        print(f"✅ Response sent to {user_first_name}")
        print(f"📈 Session stats - Total messages: {session.total_messages}, Context length: {len(session.chat_context)}")
        
        return {
            "response": response_text,
            "user_id": request.user_id,
            "username": user_first_name,
            "full_name": session.get_display_name(),
            "timestamp": datetime.now().isoformat(),
            "session_active": True,
            "session_stats": {
                "total_messages": session.total_messages,
                "context_length": len(session.chat_context)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/session/{user_id}")
async def get_session_info(user_id: str):
    if user_id not in user_sessions:
        return {"error": "Session not found"}
    
    session = user_sessions[user_id]
    return {
        "user_id": session.user_id,
        "username": session.get_display_name(),
        "first_name": session.get_first_name(),
        "company": session.user_profile.get('companyName', 'Unknown') if session.user_profile else 'Unknown',
        "position": session.user_profile.get('position', 'Unknown') if session.user_profile else 'Unknown',
        "message_count": session.total_messages,
        "context_length": len(session.chat_context),
        "last_active": session.last_active.isoformat(),
        "session_info": session.get_context_summary()
    }

@app.delete("/session/{user_id}")
async def clear_session(user_id: str):
    if user_id in user_sessions:
        session_info = user_sessions[user_id].get_context_summary()
        del user_sessions[user_id]
        return {
            "message": "Session cleared successfully",
            "cleared_session": session_info
        }
    return {"message": "Session not found"}

@app.post("/session/{user_id}/clear-context")
async def clear_chat_context(user_id: str):
    """Clear the chat context but keep the session"""
    if user_id in user_sessions:
        session = user_sessions[user_id]
        old_context_length = len(session.chat_context)
        
        # Reset to just system message
        first_name = session.get_first_name()
        system_prompt = session.create_system_prompt(session.get_display_name(), first_name)
        session.chat_context = [{"role": "system", "content": system_prompt}]
        session.total_messages = 0
        
        return {
            "message": "Chat context cleared successfully",
            "user_id": user_id,
            "old_context_length": old_context_length,
            "new_context_length": len(session.chat_context)
        }
    return {"error": "Session not found"}

# Debug endpoints
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

@app.get("/api/all-users")
async def get_all_users_endpoint():
    """Get all users for debugging"""
    try:
        users = get_all_users()
        return {
            "total_users": len(users),
            "users": users
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/find-user/{identifier}")
async def find_user_by_identifier(identifier: str):
    """Find user by any identifier (userId, email, cognitoUsername)"""
    try:
        # Try all lookup methods
        profile_by_id = get_user_profile_by_user_id(identifier)
        if profile_by_id:
            return {
                "identifier": identifier,
                "found_by": "userId",
                "profile": {
                    "userId": profile_by_id.get('userId'),
                    "firstName": profile_by_id.get('firstName'),
                    "lastName": profile_by_id.get('lastName'),
                    "email": profile_by_id.get('email'),
                    "companyName": profile_by_id.get('companyName')
                }
            }
        
        # Try by email
        if '@' in identifier:
            profile_by_email = get_user_profile_by_email(identifier)
            if profile_by_email:
                return {
                    "identifier": identifier,
                    "found_by": "email",
                    "profile": {
                        "userId": profile_by_email.get('userId'),
                        "firstName": profile_by_email.get('firstName'),
                        "lastName": profile_by_email.get('lastName'),
                        "email": profile_by_email.get('email'),
                        "companyName": profile_by_email.get('companyName')
                    }
                }
        
        # Try Cognito lookup
        cognito_user = get_cognito_user_by_username(identifier)
        if cognito_user:
            return {
                "identifier": identifier,
                "found_by": "cognito",
                "cognito_user": {
                    "user_id": cognito_user.get('user_id'),
                    "email": cognito_user.get('email')
                },
                "profile": None
            }
        
        return {
            "identifier": identifier,
            "found_by": "not_found",
            "profile": None
        }
            
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("🚀 Starting B-Max AI Assistant...")
    print("💬 Endpoint: POST /chat")
    print("🔧 Health: GET /health")
    print("📊 Database:", "Connected" if dynamodb else "Disconnected")
    print("🔐 Cognito:", "Connected" if cognito else "Disabled")
    print("🤖 Ollama:", "Connected" if ollama_available else "Disconnected")
    print("📋 Available Categories:", ", ".join(CATEGORIES))
    print(f"🌐 Server running on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
