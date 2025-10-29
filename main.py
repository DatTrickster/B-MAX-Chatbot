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

def get_tender_information(query: str = None, user_id: str = None):
    """Get REAL tender information from ProcessedTender database"""
    try:
        if not dynamodb:
            print("❌ DynamoDB client not available")
            return "Database currently unavailable"
        
        print(f"🔍 Scanning REAL data from table: {DYNAMODB_TABLE_TENDERS}")
        
        # Get ALL fields from the table to ensure we capture everything
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS,
            Limit=8  # Get recent tenders
        )
        
        items = response.get('Items', [])
        print(f"📊 Found {len(items)} REAL tender items in database")
        
        if not items:
            print("❌ No REAL tenders found in database")
            return "No tenders found in the database. The system is currently being populated with tender data."
        
        # Convert all items to Python dicts
        tenders_info = []
        for item in items:
            tender = dd_to_py(item)
            tenders_info.append(tender)
        
        print(f"✅ Processed {len(tenders_info)} REAL tenders from database")
        
        # Log first tender for debugging
        if tenders_info:
            first_tender = tenders_info[0]
            print(f"📝 Sample tender data - Title: {first_tender.get('title', 'N/A')}, Category: {first_tender.get('Category', 'N/A')}")
        
        return format_real_tenders_for_ai(tenders_info)
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return f"Error accessing tender database: {str(e)}"

def format_real_tenders_for_ai(tenders):
    """Format REAL tender information from database for AI context"""
    if not tenders:
        return "No tender data available in the system."
    
    formatted = "REAL TENDERS FROM DATABASE:\n\n"
    for i, tender in enumerate(tenders, 1):
        # Use actual field names from your DynamoDB table
        title = tender.get('title', 'No title available')
        category = tender.get('Category', 'Uncategorized')
        closing_date = tender.get('closingDate', 'No closing date')
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
        
        if link != 'No link available':
            formatted += f"🔗 Documents: {link}\n"
        if source_url != 'N/A':
            formatted += f"🌐 Source: {source_url}\n"
        formatted += "─" * 50 + "\n\n"
    
    formatted += "💡 These are REAL tenders from the database. I can help you analyze them or find specific opportunities!"
    return formatted

def search_tenders_by_category(category: str):
    """Search REAL tenders by category in the database"""
    try:
        if not dynamodb:
            return None
            
        print(f"🔍 Searching REAL tenders for category: {category}")
            
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS,
            FilterExpression="contains(Category, :cat)",
            ExpressionAttributeValues={":cat": {"S": category}},
            Limit=6
        )
        
        items = response.get('Items', [])
        tenders = []
        for item in items:
            tender = dd_to_py(item)
            tenders.append(tender)
        
        print(f"✅ Found {len(tenders)} REAL tenders for category: {category}")
        return tenders
    except Exception as e:
        print(f"❌ Error searching REAL tenders by category: {e}")
        return None

def get_recent_tenders(limit: int = 5):
    """Get recent tenders from the database"""
    try:
        if not dynamodb:
            return None
            
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS,
            Limit=limit
        )
        
        items = response.get('Items', [])
        tenders = []
        for item in items:
            tender = dd_to_py(item)
            tenders.append(tender)
        
        return tenders
    except Exception as e:
        print(f"❌ Error getting recent REAL tenders: {e}")
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
        
        # Load user profile
        self.load_user_profile()
        
        # Set up system prompt AFTER profile is loaded
        username = self.get_display_name()
        first_name = self.get_first_name()
        self.chat_context = [
            {"role": "system", "content": self.create_system_prompt(username, first_name)}
        ]
        
        print(f"✅ Session created - Name: {first_name}")

    def create_system_prompt(self, username: str, first_name: str):
        available_categories_str = ", ".join(CATEGORIES)
        
        return f"""You are B-Max, an AI assistant for TenderConnect. 

CRITICAL RULES - FOLLOW THESE EXACTLY:
1. ALWAYS address the user by their first name "{first_name}" in EVERY response
2. NEVER invent or create fake tender data. Only use REAL data from the database.
3. If no tenders are found, say "No tenders found in the database" instead of making examples.
4. Be warm, friendly, and professional
5. Remember context from previous messages
6. Use emojis occasionally to make conversations friendly
7. Keep responses concise but informative
8. Focus on tender-related topics and procurement
9. NEVER mention or reference other users or their information
10. ALWAYS provide REAL tender information from the database when available

AVAILABLE TENDER CATEGORIES:
{available_categories_str}

Your capabilities:
- Answer questions about REAL tenders from the database
- Provide tender recommendations based on REAL data
- Explain tender categories and requirements
- Help with tender search strategies
- Provide information about REAL tender deadlines and procedures

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMPORTANT: 
- Only discuss REAL tenders from the database
- Never invent fake tender examples
- If database has no tenders, be honest about it
- Focus on providing helpful information based on REAL data"""

    def load_user_profile(self):
        """Simple profile loading"""
        self.user_profile = {
            'firstName': 'User',
            'lastName': '',
            'companyName': 'Unknown',
            'position': 'User',
            'location': 'Unknown',
            'preferredCategories': []
        }

    def get_display_name(self):
        return self.user_profile.get('firstName', 'User')

    def get_first_name(self):
        return self.user_profile.get('firstName', 'User')

    def update_activity(self):
        self.last_active = datetime.now()

    def add_message(self, role, content):
        self.chat_context.append({"role": role, "content": content})
        self.total_messages += 1
        
        # Keep reasonable context length
        if len(self.chat_context) > 21:
            system_message = self.chat_context[0]
            recent_messages = self.chat_context[-20:]
            self.chat_context = [system_message] + recent_messages
        
        print(f"💬 Message added - Role: {role}, Total messages: {self.total_messages}")

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

def enhance_prompt_with_context(user_prompt: str, session: UserSession) -> str:
    """Enhance user prompt with REAL tender context"""
    database_context = ""
    
    # Enhanced category mapping
    category_keywords = {
        'Engineering Services': ['engineering', 'engineer', 'technical', 'design', 'infrastructure'],
        'IT Services': ['IT', 'technology', 'software', 'hardware', 'computer', 'digital', 'tech'],
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
    
    # Add REAL tender context if the message is tender-related
    tender_keywords = ['tender', 'tenders', 'procurement', 'bid', 'category', 'recommend', 'suggest', 'opportunity', 'RFP', 'RFQ']
    
    if any(keyword in user_prompt_lower for keyword in tender_keywords) or matched_categories:
        print(f"🔍 Looking for REAL tenders related to: {user_prompt}")
        
        if matched_categories:
            # Try to find tenders in specific categories
            for category in matched_categories:
                specific_tenders = search_tenders_by_category(category)
                if specific_tenders:
                    database_context = format_real_tenders_for_ai(specific_tenders)
                    database_context = f"🔍 REAL Tenders in {category}:\n\n{database_context}"
                    break
            
            # If no specific category tenders found, get general tenders
            if not database_context:
                database_context = get_tender_information()
                if "No tenders found" not in database_context:
                    database_context = f"📊 No specific tenders found for those categories. Here are recent REAL tenders:\n\n{database_context}"
        else:
            # Get general REAL tenders
            database_context = get_tender_information()
    
    user_first_name = session.get_first_name()
    
    enhanced_prompt = f"""
User: {user_first_name}
Message: {user_prompt}

{database_context if database_context else "No tender data available for this query."}

CRITICAL: Only provide REAL tender information from the database. Never invent fake examples.
If no tenders are found, be honest and say so.
Address the user as {user_first_name}.
"""
    return enhanced_prompt

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    return {
        "message": "B-Max AI Assistant API is running!",
        "status": "healthy" if ollama_available else "degraded",
        "cognito_enabled": cognito is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    # Test database connection
    db_status = "connected" if dynamodb else "disconnected"
    tender_count = 0
    
    if dynamodb:
        try:
            response = dynamodb.scan(TableName=DYNAMODB_TABLE_TENDERS, Limit=1)
            tender_count = len(response.get('Items', []))
        except Exception as e:
            db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "service": "B-Max AI Assistant",
        "dynamodb": db_status,
        "tender_count": tender_count,
        "cognito": "connected" if cognito else "disabled",
        "ollama": "connected" if ollama_available else "disconnected",
        "active_sessions": len(user_sessions),
        "timestamp": datetime.now().isoformat()
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
        
        # Enhance prompt with REAL tender context
        enhanced_prompt = enhance_prompt_with_context(request.prompt, session)
        
        # Add enhanced user message to context
        session.add_message("user", enhanced_prompt)
        
        # Get AI response
        try:
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
        "message_count": session.total_messages,
        "last_active": session.last_active.isoformat()
    }

@app.delete("/session/{user_id}")
async def clear_session(user_id: str):
    if user_id in user_sessions:
        del user_sessions[user_id]
        return {"message": "Session cleared successfully"}
    return {"message": "Session not found"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("🚀 Starting B-Max AI Assistant...")
    print("💬 Endpoint: POST /chat")
    print("🔧 Health: GET /health")
    print("📊 Database:", "Connected" if dynamodb else "Disconnected")
    print("🔐 Cognito:", "Connected" if cognito else "Disabled")
    print("🤖 Ollama:", "Connected" if ollama_available else "Disconnected")
    print(f"🌐 Server running on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
