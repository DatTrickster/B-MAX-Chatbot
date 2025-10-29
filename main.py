import os
import uvicorn
import json
import boto3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "af-south-1")
DYNAMODB_TABLE_TENDERS = os.getenv("DYNAMODB_TABLE_DEST", "ProcessedTender")
DYNAMODB_TABLE_USERS = os.getenv("DYNAMODB_TABLE_USERS", "TenderConnectUsers")

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

# Ollama Client
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

app = FastAPI(title="B-Max AI Assistant", version="1.0.0")

# In-memory session storage (in production, use Redis or database)
user_sessions = {}

class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "guest"  # Default user ID
    username: str = "User"  # User's display name

class UserSession:
    def __init__(self, user_id, username):
        self.user_id = user_id
        self.username = username
        self.chat_context = [
            {"role": "system", "content": f"""You are B-Max, an AI assistant for TenderConnect. 

IMPORTANT RULES:
1. Always be respectful, professional, and helpful
2. Remember context from previous messages in this conversation
3. You can access tender database information when relevant
4. Personalize responses using the user's name: {username}
5. If you don't know something, be honest and say so
6. Keep responses concise but informative
7. Use emojis occasionally to make conversations friendly

Your capabilities:
- Answer questions about tenders, procurement, and the TenderConnect platform
- Provide information about tender categories, deadlines, and processes
- Help users understand how to use the platform
- Access real tender data from the database when needed

Current user: {username}
Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Start by greeting the user warmly and introducing yourself."""}
        ]
        self.last_active = datetime.now()
        self.greeted = False

    def update_activity(self):
        self.last_active = datetime.now()

    def add_message(self, role, content):
        self.chat_context.append({"role": role, "content": content})
        # Keep only last 20 messages to manage context length
        if len(self.chat_context) > 20:
            self.chat_context = [self.chat_context[0]] + self.chat_context[-19:]

def get_user_session(user_id: str, username: str) -> UserSession:
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id, username)
    user_sessions[user_id].update_activity()
    return user_sessions[user_id]

def cleanup_old_sessions():
    """Remove sessions older than 2 hours"""
    current_time = datetime.now()
    expired_users = []
    for user_id, session in user_sessions.items():
        if (current_time - session.last_active).total_seconds() > 7200:  # 2 hours
            expired_users.append(user_id)
    for user_id in expired_users:
        del user_sessions[user_id]

def get_tender_information(query: str = None):
    """Get relevant tender information from database"""
    try:
        if not dynamodb:
            return "Database currently unavailable"
        
        # Build query based on user's question
        if query and any(keyword in query.lower() for keyword in ['recent', 'latest', 'new']):
            # Get latest tenders
            response = dynamodb.scan(
                TableName=DYNAMODB_TABLE_TENDERS,
                Limit=5,
                Select='SPECIFIC_ATTRIBUTES',
                ProjectionExpression='title, Category, closingDate, sourceAgency'
            )
        elif query and any(keyword in query.lower() for keyword in ['category', 'categories', 'type']):
            # Get tenders by category
            response = dynamodb.scan(
                TableName=DYNAMODB_TABLE_TENDERS,
                Select='SPECIFIC_ATTRIBUTES',
                ProjectionExpression='title, Category, closingDate'
            )
        else:
            # General tender count and stats
            response = dynamodb.scan(
                TableName=DYNAMODB_TABLE_TENDERS,
                Select='COUNT'
            )
            return f"Total tenders in database: {response.get('Count', 0)}"
        
        items = response.get('Items', [])
        if not items:
            return "No tenders found in the database."
        
        # Format the response
        tenders_info = []
        for item in items[:5]:  # Limit to 5 tenders
            tender = {}
            for key, value in item.items():
                if 'S' in value:
                    tender[key] = value['S']
                elif 'N' in value:
                    tender[key] = value['N']
            tenders_info.append(tender)
        
        return format_tenders_for_ai(tenders_info)
        
    except Exception as e:
        return f"Error accessing database: {str(e)}"

def format_tenders_for_ai(tenders):
    """Format tender data for AI consumption"""
    if not tenders:
        return "No tender data available."
    
    formatted = "Recent Tenders:\n"
    for i, tender in enumerate(tenders, 1):
        title = tender.get('title', 'Unknown Title')
        category = tender.get('Category', 'Uncategorized')
        closing_date = tender.get('closingDate', 'Unknown')
        agency = tender.get('sourceAgency', 'Unknown Agency')
        
        formatted += f"{i}. {title}\n"
        formatted += f"   Category: {category}\n"
        formatted += f"   Agency: {agency}\n"
        formatted += f"   Closes: {closing_date}\n\n"
    
    return formatted

def enhance_prompt_with_context(user_prompt: str, username: str, session: UserSession) -> str:
    """Enhance user prompt with context and database information"""
    
    # Check if we need to query database
    database_context = ""
    tender_keywords = ['tender', 'tenders', 'procurement', 'bid', 'bids', 'contract', 'rfp', 'rfi', 'category', 'categories']
    
    if any(keyword in user_prompt.lower() for keyword in tender_keywords):
        database_context = get_tender_information(user_prompt)
    
    enhanced_prompt = f"""
User: {username}
Message: {user_prompt}

Current Context:
- Conversation history available in chat context
- User is asking about: {user_prompt}

{database_context if database_context else "No additional database context needed for this query."}

Please respond naturally while:
1. Using the user's name: {username}
2. Maintaining conversation context
3. Being helpful and professional
4. Using database information if provided above
"""
    return enhanced_prompt

@app.get("/")
async def root():
    return {
        "message": "B-Max AI Assistant API is running!",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health",
            "session_info": "/session/{user_id}"
        }
    }

@app.get("/health")
async def health_check():
    db_status = "healthy" if dynamodb else "unavailable"
    try:
        # Test Ollama connection
        test_response = ""
        for part in client.chat('deepseek-v3.1:671b-cloud', messages=[{"role": "user", "content": "Say OK"}], stream=True):
            test_response += part['message']['content']
        ollama_status = "healthy"
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
    """Main chat endpoint with context memory and database access"""
    try:
        # Clean up old sessions periodically
        cleanup_old_sessions()
        
        # Get or create user session
        session = get_user_session(request.user_id, request.username)
        
        # Enhance prompt with context and database info
        enhanced_prompt = enhance_prompt_with_context(request.prompt, request.username, session)
        
        # Add user message to context
        session.add_message("user", enhanced_prompt)
        
        # Generate response
        response_text = ""
        print(f"💬 {request.username}: {request.prompt}")
        print("🤖 B-Max thinking...", end="", flush=True)
        
        # Stream response from AI
        for part in client.chat('deepseek-v3.1:671b-cloud', messages=session.chat_context, stream=True):
            response_text += part['message']['content']
            print(".", end="", flush=True)
        
        print()  # New line after thinking dots
        
        # Add AI response to context
        session.add_message("assistant", response_text)
        
        # Mark as greeted after first interaction
        if not session.greeted:
            session.greeted = True
        
        print(f"✅ B-Max: {response_text[:100]}...")
        
        return {
            "response": response_text,
            "user_id": request.user_id,
            "username": request.username,
            "timestamp": datetime.now().isoformat(),
            "session_active": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/session/{user_id}")
async def get_session_info(user_id: str):
    """Get information about a user's session"""
    if user_id not in user_sessions:
        return {"error": "Session not found"}
    
    session = user_sessions[user_id]
    return {
        "user_id": session.user_id,
        "username": session.username,
        "message_count": len(session.chat_context) - 1,  # Exclude system message
        "last_active": session.last_active.isoformat(),
        "greeted": session.greeted
    }

@app.delete("/session/{user_id}")
async def clear_session(user_id: str):
    """Clear a user's session"""
    if user_id in user_sessions:
        del user_sessions[user_id]
        return {"message": "Session cleared successfully"}
    return {"message": "Session not found"}

@app.get("/tenders")
async def get_tenders_preview():
    """Get a preview of tenders for context"""
    try:
        tender_info = get_tender_information("recent")
        return {
            "tenders_preview": tender_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to get tender preview: {str(e)}"}

if __name__ == "__main__":
    print("🚀 Starting B-Max AI Assistant...")
    print("💬 Endpoint: POST /chat")
    print("🔧 Health: GET /health")
    print("📊 Tenders Preview: GET /tenders")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
