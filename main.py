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

# Initialize AWS clients
try:
    dynamodb = boto3.client(
        'dynamodb',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=AWS_REGION
    )
    print("✅ AWS Clients initialized successfully")
except Exception as e:
    print(f"❌ AWS Client initialization error: {e}")
    dynamodb = None

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
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
user_sessions = {}

class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "guest"

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = None
        self.chat_context = []
        self.last_active = datetime.now()
        self.greeted = False
        self.bookmark_patterns = None
        
        # Load user profile FIRST before creating system prompt
        self.load_user_profile()
        
        # Set up system prompt AFTER profile is loaded
        username = self.get_display_name()
        first_name = self.get_first_name()
        self.chat_context = [
            {"role": "system", "content": self.create_system_prompt(username, first_name)}
        ]
        
        print(f"🎯 Created session for: {first_name} ({username})")

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

CRITICAL RULES:
1. ALWAYS address the user by their first name "{first_name}" in your responses
2. Be warm, friendly, and professional
3. Remember context from previous messages
4. If you don't know something, be honest and say so
5. Keep responses concise but informative
6. Use emojis occasionally to make conversations friendly

Your capabilities:
- Answer questions about tenders and procurement
- Provide information about tender categories and deadlines
- Help users understand the TenderConnect platform
- Provide personalized recommendations based on user preferences

User Profile:
- Full Name: {username}
- First Name: {first_name} (USE THIS NAME IN ALL RESPONSES)
- Company: {company}
- Position: {position}
- Location: {location}
- Preferred Categories: {categories_str}

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMPORTANT: Start your first response with a warm greeting that includes the user's first name "{first_name}". For example: "Hi {first_name}! 👋" or "Hello {first_name}! 😊""

    def load_user_profile(self):
        """Load user profile from UserProfiles table using the correct schema"""
        try:
            if dynamodb:
                print(f"🔍 Loading profile for user: {self.user_id}")
                response = dynamodb.get_item(
                    TableName=DYNAMODB_TABLE_USERS,
                    Key={'userId': {'S': self.user_id}}
                )
                if 'Item' in response:
                    self.user_profile = self.dynamodb_to_python_enhanced(response['Item'])
                    print(f"✅ Loaded profile for user: {self.user_id}")
                    print(f"   Name: {self.get_display_name()}")
                    print(f"   First Name: {self.get_first_name()}")
                    print(f"   Company: {self.user_profile.get('companyName', 'Unknown')}")
                else:
                    print(f"❌ No profile found for user: {self.user_id}")
                    self.user_profile = self.create_default_profile()
        except Exception as e:
            print(f"❌ Error loading user profile: {e}")
            self.user_profile = self.create_default_profile()

    def dynamodb_to_python_enhanced(self, item):
        """Enhanced DynamoDB to Python conversion that handles all attribute types"""
        result = {}
        for key, value in item.items():
            if 'S' in value:
                result[key] = value['S']
            elif 'N' in value:
                result[key] = float(value['N']) if '.' in value['N'] else int(value['N'])
            elif 'BOOL' in value:
                result[key] = value['BOOL']
            elif 'M' in value:
                result[key] = self.dynamodb_to_python_enhanced(value['M'])
            elif 'L' in value:
                result[key] = [self.dynamodb_to_python_enhanced({'item': item})['item'] for item in value['L']]
            elif 'SS' in value:
                result[key] = list(value['SS'])  # String Set
            elif 'NS' in value:
                result[key] = [float(n) if '.' in n else int(n) for n in value['NS']]  # Number Set
            elif 'BS' in value:
                result[key] = list(value['BS'])  # Binary Set
            elif 'NULL' in value and value['NULL']:
                result[key] = None
        return result

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
            # Get first name from profile, default to 'User'
            first_name = self.user_profile.get('firstName', 'User')
            return first_name
        return "User"

    def update_activity(self):
        self.last_active = datetime.now()

    def add_message(self, role, content):
        self.chat_context.append({"role": role, "content": content})
        # Keep only last 15 messages to manage context
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
        if (current_time - session.last_active).total_seconds() > 7200:  # 2 hours
            expired_users.append(user_id)
    for user_id in expired_users:
        del user_sessions[user_id]

def get_tender_information(query: str = None, user_id: str = None):
    """Get tender information from database"""
    try:
        if not dynamodb:
            return "Database currently unavailable"
        
        # Simple implementation for demo
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS,
            Limit=3,
            Select='SPECIFIC_ATTRIBUTES',
            ProjectionExpression='title, Category, closingDate'
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
        return f"Error accessing database: {str(e)}"

def format_tenders_for_ai(tenders):
    if not tenders:
        return "No tender data available."
    
    formatted = "Recent Tenders:\n"
    for i, tender in enumerate(tenders, 1):
        title = tender.get('title', 'Unknown Title')[:50] + '...' if len(tender.get('title', '')) > 50 else tender.get('title', 'Unknown Title')
        category = tender.get('Category', 'Uncategorized')
        closing_date = tender.get('closingDate', 'Unknown')
        
        formatted += f"{i}. {title}\n"
        formatted += f"   Category: {category}\n"
        formatted += f"   Closes: {closing_date}\n\n"
    
    return formatted

def enhance_prompt_with_context(user_prompt: str, session: UserSession) -> str:
    database_context = ""
    tender_keywords = ['tender', 'tenders', 'procurement', 'bid', 'category']
    
    if any(keyword in user_prompt.lower() for keyword in tender_keywords):
        database_context = get_tender_information(user_prompt, session.user_id)
    
    user_first_name = session.get_first_name()
    
    enhanced_prompt = f"""
User: {user_first_name}
Message: {user_prompt}

{database_context if database_context else ""}

Remember to address the user by their first name "{user_first_name}" in your response.
"""
    return enhanced_prompt

@app.get("/")
async def root():
    return {
        "message": "B-Max AI Assistant API is running!",
        "status": "healthy" if ollama_available else "degraded",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health",
            "session_info": "/session/{user_id}"
        }
    }

@app.get("/health")
async def health_check():
    db_status = "healthy" if dynamodb else "unavailable"
    
    ollama_status = "unavailable"
    if ollama_available:
        try:
            # Simple test without streaming
            test_response = client.chat('deepseek-v3.1:671b-cloud', messages=[{"role": "user", "content": "Say OK"}])
            if test_response and test_response.get('message', {}).get('content'):
                ollama_status = "healthy"
            else:
                ollama_status = "unhealthy"
        except Exception as e:
            ollama_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok",
        "assistant": "B-Max AI",
        "database": db_status,
        "ollama": ollama_status,
        "active_sessions": len(user_sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not ollama_available:
            raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
        
        cleanup_old_sessions()
        
        session = get_user_session(request.user_id)
        user_first_name = session.get_first_name()
        
        print(f"🎯 Processing chat for: {user_first_name} (ID: {request.user_id})")
        
        # For the first message, ensure we use a fresh context with the user's name
        if len(session.chat_context) <= 1:  # Only system message
            enhanced_prompt = f"Hello, my name is {user_first_name}. {request.prompt}"
        else:
            enhanced_prompt = enhance_prompt_with_context(request.prompt, session)
        
        session.add_message("user", enhanced_prompt)
        
        print(f"💬 {user_first_name}: {request.prompt}")
        print(f"📋 Context messages: {len(session.chat_context)}")
        
        # Use non-streaming approach for better reliability
        try:
            response = client.chat(
                'deepseek-v3.1:671b-cloud', 
                messages=session.chat_context
            )
            response_text = response['message']['content']
        except Exception as e:
            print(f"❌ Ollama API error: {e}")
            response_text = f"I apologize {user_first_name}, but I'm having trouble processing your request right now. Please try again in a moment."
        
        session.add_message("assistant", response_text)
        
        if not session.greeted:
            session.greeted = True
        
        print(f"✅ B-Max responded to {user_first_name}")
        print(f"💡 Response: {response_text[:100]}...")
        
        return {
            "response": response_text,
            "user_id": request.user_id,
            "username": user_first_name,
            "full_name": session.get_display_name(),
            "timestamp": datetime.now().isoformat(),
            "session_active": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500,
