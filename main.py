import os
import uvicorn
import json
import boto3
import time
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from difflib import SequenceMatcher

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

# Try to import Ollama
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def scan_all_tenders():
    """Scan ALL tenders from database with pagination"""
    try:
        if not dynamodb:
            return []
        
        all_tenders = []
        last_evaluated_key = None
        
        while True:
            scan_params = {
                'TableName': DYNAMODB_TABLE_TENDERS,
                'Limit': 100  # Process in batches of 100
            }
            
            if last_evaluated_key:
                scan_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.scan(**scan_params)
            items = response.get('Items', [])
            
            for item in items:
                tender = dd_to_py(item)
                all_tenders.append(tender)
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        print(f"📊 Scanned {len(all_tenders)} total tenders from database")
        return all_tenders
        
    except Exception as e:
        print(f"❌ Error scanning all tenders: {e}")
        return []

def calculate_relevance_score(tender, query, search_type="general"):
    """Calculate relevance score for a tender based on query"""
    score = 0
    query_lower = query.lower()
    
    # Extract potential reference numbers from query
    ref_patterns = re.findall(r'[A-Z0-9-/]+', query.upper())
    
    # Check for exact reference number match (highest priority)
    tender_ref = str(tender.get('referenceNumber', '')).upper()
    if tender_ref and any(ref in tender_ref for ref in ref_patterns if len(ref) > 3):
        score += 1000  # Very high score for reference match
        print(f"🎯 Exact reference match found: {tender_ref}")
    
    # Check title relevance
    title = str(tender.get('title', '')).lower()
    if title:
        # Exact phrase match in title
        if query_lower in title:
            score += 100
        # Word matches in title
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in title:
                score += 20
        # Fuzzy match for title
        similarity = SequenceMatcher(None, query_lower, title).ratio()
        score += similarity * 50
    
    # Check category relevance
    category = str(tender.get('Category', '')).lower()
    if category and category in query_lower:
        score += 80
    
    # Check agency relevance
    agency = str(tender.get('sourceAgency', '')).lower()
    if agency:
        if query_lower in agency or agency in query_lower:
            score += 60
        # Check for partial agency name matches
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in agency:
                score += 15
    
    # Bonus for active/open status
    status = str(tender.get('status', '')).lower()
    if 'open' in status or 'active' in status:
        score += 10
    
    # Check closing date (prefer sooner deadlines if available)
    closing_date = tender.get('closingDate', '')
    if closing_date and closing_date != 'No closing date':
        try:
            # Parse date and give bonus to upcoming deadlines
            # This is a simple implementation - adjust based on your date format
            score += 5
        except:
            pass
    
    return score

def search_specific_tender(query):
    """Search for a specific tender by reference or exact match"""
    try:
        all_tenders = scan_all_tenders()
        
        if not all_tenders:
            return None
        
        # Look for reference number patterns
        ref_patterns = re.findall(r'[A-Z0-9-/]+', query.upper())
        
        # First pass: exact reference match
        for tender in all_tenders:
            tender_ref = str(tender.get('referenceNumber', '')).upper()
            if tender_ref:
                for ref in ref_patterns:
                    if len(ref) > 3 and ref in tender_ref:
                        print(f"✅ Found exact reference match: {tender_ref}")
                        return [tender]
        
        # Second pass: exact title match
        query_lower = query.lower()
        for tender in all_tenders:
            title = str(tender.get('title', '')).lower()
            if query_lower in title or title in query_lower:
                if len(query_lower) > 10 or len(title) > 10:  # Avoid short false matches
                    print(f"✅ Found exact title match: {title}")
                    return [tender]
        
        # Third pass: score-based fuzzy matching
        scored_tenders = []
        for tender in all_tenders:
            score = calculate_relevance_score(tender, query, "specific")
            if score > 50:  # Only include tenders with decent relevance
                scored_tenders.append((score, tender))
        
        scored_tenders.sort(reverse=True, key=lambda x: x[0])
        
        if scored_tenders:
            print(f"✅ Found {len(scored_tenders)} relevant matches (showing top result)")
            return [scored_tenders[0][1]]
        
        return None
        
    except Exception as e:
        print(f"❌ Error in specific tender search: {e}")
        return None

def smart_tender_search(query, limit=5):
    """Intelligently search tenders based on query"""
    try:
        print(f"🔍 Smart search for: {query}")
        
        # Check if this is a specific tender request
        specific_indicators = ['reference', 'tender number', 'ref:', 'number:', 'specific']
        query_lower = query.lower()
        
        is_specific_search = any(indicator in query_lower for indicator in specific_indicators)
        has_reference_pattern = bool(re.search(r'[A-Z]{2,}[-/]?\d+', query.upper()))
        
        if is_specific_search or has_reference_pattern:
            print("🎯 Detected specific tender search")
            result = search_specific_tender(query)
            if result:
                return result
        
        # General search: scan all and rank by relevance
        all_tenders = scan_all_tenders()
        
        if not all_tenders:
            return []
        
        # Score all tenders
        scored_tenders = []
        for tender in all_tenders:
            score = calculate_relevance_score(tender, query)
            if score > 0:  # Only include tenders with some relevance
                scored_tenders.append((score, tender))
        
        # Sort by score (highest first)
        scored_tenders.sort(reverse=True, key=lambda x: x[0])
        
        # Return top N results
        top_tenders = [tender for score, tender in scored_tenders[:limit]]
        
        print(f"✅ Found {len(scored_tenders)} relevant tenders, returning top {len(top_tenders)}")
        if top_tenders and scored_tenders:
            print(f"📊 Top score: {scored_tenders[0][0]}, Lowest score: {scored_tenders[-1][0] if len(scored_tenders) > 0 else 'N/A'}")
        
        return top_tenders
        
    except Exception as e:
        print(f"❌ Error in smart tender search: {e}")
        return []

def get_tender_information(query: str = None, user_id: str = None):
    """Get REAL tender information with smart search"""
    try:
        if not dynamodb:
            print("❌ DynamoDB client not available")
            return "Database currently unavailable"
        
        if query:
            # Use smart search with query
            tenders = smart_tender_search(query, limit=5)
        else:
            # Get recent tenders if no query
            tenders = scan_all_tenders()[:5]
        
        if not tenders:
            print("❌ No relevant tenders found")
            return "No relevant tenders found in the database matching your query."
        
        print(f"✅ Returning {len(tenders)} tenders")
        return format_real_tenders_for_ai(tenders)
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return f"Error accessing tender database: {str(e)}"

def format_real_tenders_for_ai(tenders):
    """Format REAL tender information from database for AI context"""
    if not tenders:
        return "No tender data available in the system."
    
    formatted = "REAL TENDERS FROM DATABASE:\n\n"
    for i, tender in enumerate(tenders, 1):
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
    
    formatted += f"💡 Showing {len(tenders)} most relevant tender(s) from the database."
    return formatted

def search_tenders_by_category(category: str, limit=5):
    """Search REAL tenders by category"""
    try:
        if not dynamodb:
            return None
        
        print(f"🔍 Searching tenders for category: {category}")
        
        all_tenders = scan_all_tenders()
        
        # Filter by category
        category_tenders = []
        for tender in all_tenders:
            tender_category = str(tender.get('Category', '')).lower()
            if category.lower() in tender_category:
                category_tenders.append(tender)
        
        # Sort by relevance (status, closing date, etc.)
        category_tenders.sort(key=lambda t: (
            'open' in str(t.get('status', '')).lower(),
            t.get('closingDate', '') != 'No closing date'
        ), reverse=True)
        
        result = category_tenders[:limit]
        print(f"✅ Found {len(category_tenders)} tenders in category, returning top {len(result)}")
        return result
        
    except Exception as e:
        print(f"❌ Error searching tenders by category: {e}")
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
        
        self.load_user_profile()
        
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
11. When showing tender results, present them clearly with all relevant details

AVAILABLE TENDER CATEGORIES:
{available_categories_str}

Your capabilities:
- Search for specific tenders by reference number or exact criteria
- Answer questions about REAL tenders from the database
- Provide tender recommendations based on REAL data
- Explain tender categories and requirements
- Help with tender search strategies
- Provide information about REAL tender deadlines and procedures

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMPORTANT: 
- The system scans the ENTIRE database and returns the most relevant results
- When users ask for specific tenders (by reference, name, etc.), find the exact match
- Only show the top 5 most relevant tenders unless otherwise specified
- Only discuss REAL tenders from the database
- Never invent fake tender examples
- If database has no tenders, be honest about it"""

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
    """Enhance user prompt with REAL tender context using smart search"""
    database_context = ""
    
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
    
    # Check if this is a tender-related query
    tender_keywords = ['tender', 'tenders', 'procurement', 'bid', 'category', 'recommend', 
                       'suggest', 'opportunity', 'RFP', 'RFQ', 'reference', 'find', 'show', 'search']
    
    if any(keyword in user_prompt_lower for keyword in tender_keywords) or matched_categories:
        print(f"🔍 Processing tender-related query: {user_prompt}")
        
        if matched_categories:
            # Search by category
            for category in matched_categories:
                specific_tenders = search_tenders_by_category(category, limit=5)
                if specific_tenders:
                    database_context = format_real_tenders_for_ai(specific_tenders)
                    database_context = f"🔍 REAL Tenders in {category}:\n\n{database_context}"
                    break
        else:
            # Use smart search with the full query
            database_context = get_tender_information(query=user_prompt)
    
    user_first_name = session.get_first_name()
    
    enhanced_prompt = f"""
User: {user_first_name}
Message: {user_prompt}

{database_context if database_context else "No specific tender data needed for this query."}

CRITICAL: Only provide REAL tender information from the database. Never invent fake examples.
The system has searched the entire database and provided the most relevant results.
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
    db_status = "connected" if dynamodb else "disconnected"
    tender_count = 0
    
    if dynamodb:
        try:
            all_tenders = scan_all_tenders()
            tender_count = len(all_tenders)
        except Exception as e:
            db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "service": "B-Max AI Assistant",
        "dynamodb": db_status,
        "total_tenders": tender_count,
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
        
        # Enhance prompt with REAL tender context using smart search
        enhanced_prompt = enhance_prompt_with_context(request.prompt, session)
        
        session.add_message("user", enhanced_prompt)
        
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
