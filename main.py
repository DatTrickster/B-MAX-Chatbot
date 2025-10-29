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
    try:
        if not dynamodb:
            return []
        
        all_tenders = []
        last_evaluated_key = None
        
        while True:
            scan_params = {
                'TableName': DYNAMODB_TABLE_TENDERS,
                'Limit': 100
            }
            if last_evaluated_key:
                scan_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.scan(**scan_params)
            items = response.get('Items', [])
            for item in items:
                all_tenders.append(dd_to_py(item))
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        return all_tenders
        
    except Exception as e:
        print(f"❌ Error scanning tenders: {e}")
        return []

# --- Dynamic Categories ---
def get_dynamic_categories():
    """Fetch all unique tender categories dynamically from DB"""
    tenders = scan_all_tenders()
    categories = set()
    for t in tenders:
        category = t.get('Category')
        if category:
            categories.add(category)
    return sorted(list(categories)) if categories else ["Uncategorized"]

# --- Relevance and Search Functions ---
def calculate_relevance_score(tender, query, search_type="general"):
    score = 0
    query_lower = query.lower()
    ref_patterns = re.findall(r'[A-Z0-9-/]+', query.upper())
    tender_ref = str(tender.get('referenceNumber', '')).upper()
    if tender_ref and any(ref in tender_ref for ref in ref_patterns if len(ref) > 3):
        score += 1000
    title = str(tender.get('title', '')).lower()
    if title:
        if query_lower in title:
            score += 100
        for word in query_lower.split():
            if len(word) > 3 and word in title:
                score += 20
        score += SequenceMatcher(None, query_lower, title).ratio() * 50
    category = str(tender.get('Category', '')).lower()
    if category and category in query_lower:
        score += 80
    agency = str(tender.get('sourceAgency', '')).lower()
    if agency:
        if query_lower in agency or agency in query_lower:
            score += 60
        for word in query_lower.split():
            if len(word) > 3 and word in agency:
                score += 15
    status = str(tender.get('status', '')).lower()
    if 'open' in status or 'active' in status:
        score += 10
    return score

def search_specific_tender(query):
    all_tenders = scan_all_tenders()
    if not all_tenders:
        return None
    ref_patterns = re.findall(r'[A-Z0-9-/]+', query.upper())
    for tender in all_tenders:
        tender_ref = str(tender.get('referenceNumber', '')).upper()
        if tender_ref:
            for ref in ref_patterns:
                if len(ref) > 3 and ref in tender_ref:
                    return [tender]
    query_lower = query.lower()
    for tender in all_tenders:
        title = str(tender.get('title', '')).lower()
        if query_lower in title or title in query_lower:
            if len(query_lower) > 10 or len(title) > 10:
                return [tender]
    scored_tenders = []
    for tender in all_tenders:
        score = calculate_relevance_score(tender, query, "specific")
        if score > 50:
            scored_tenders.append((score, tender))
    scored_tenders.sort(reverse=True, key=lambda x: x[0])
    if scored_tenders:
        return [scored_tenders[0][1]]
    return None

def smart_tender_search(query, limit=5):
    all_tenders = scan_all_tenders()
    scored_tenders = []
    for tender in all_tenders:
        score = calculate_relevance_score(tender, query)
        if score > 0:
            scored_tenders.append((score, tender))
    scored_tenders.sort(reverse=True, key=lambda x: x[0])
    return [tender for score, tender in scored_tenders[:limit]]

def format_real_tenders_for_ai(tenders):
    if not tenders:
        return "No tender data available in the system."
    formatted = "REAL TENDERS FROM DATABASE:\n\n"
    for i, tender in enumerate(tenders, 1):
        formatted += f"🚀 TENDER #{i}\n"
        formatted += f"📋 Title: {tender.get('title','N/A')}\n"
        formatted += f"🏷️ Reference: {tender.get('referenceNumber','N/A')}\n"
        formatted += f"📊 Category: {tender.get('Category','Uncategorized')}\n"
        formatted += f"🏢 Agency: {tender.get('sourceAgency','N/A')}\n"
        formatted += f"📅 Closing Date: {tender.get('closingDate','N/A')}\n"
        formatted += f"📈 Status: {tender.get('status','N/A')}\n"
        formatted += "─" * 50 + "\n\n"
    formatted += f"💡 Showing {len(tenders)} most relevant tender(s) from the database."
    return formatted

def search_tenders_by_category(category: str, limit=5):
    all_tenders = scan_all_tenders()
    category_tenders = [t for t in all_tenders if category.lower() in str(t.get('Category','')).lower()]
    return category_tenders[:limit]

# --- User Session Management ---
class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = {'firstName':'User'}
        self.chat_context = []
        self.last_active = datetime.now()
        self.total_messages = 0
        self.load_session_prompt()
    def load_session_prompt(self):
        first_name = self.user_profile.get('firstName','User')
        dynamic_categories = get_dynamic_categories()
        categories_str = ", ".join(dynamic_categories)
        system_prompt = f"""
You are B-Max, an AI assistant for TenderConnect.

CRITICAL RULES:
- Always address user as {first_name}
- Only use REAL tender data
- Available categories: {categories_str}
"""
        self.chat_context.append({"role":"system","content":system_prompt})
    def get_first_name(self):
        return self.user_profile.get('firstName','User')
    def update_activity(self):
        self.last_active = datetime.now()
    def add_message(self, role, content):
        self.chat_context.append({"role":role,"content":content})
        self.total_messages += 1
        if len(self.chat_context) > 21:
            self.chat_context = [self.chat_context[0]] + self.chat_context[-20:]

def get_user_session(user_id: str) -> UserSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    user_sessions[user_id].update_activity()
    return user_sessions[user_id]

def cleanup_old_sessions():
    now = datetime.now()
    expired = [uid for uid, s in user_sessions.items() if (now - s.last_active).total_seconds() > 7200]
    for uid in expired:
        del user_sessions[uid]

def enhance_prompt_with_context(user_prompt: str, session: UserSession) -> str:
    context = get_tender_information(user_prompt) if any(k in user_prompt.lower() for k in ['tender','bid','rfq','rfp']) else ""
    first_name = session.get_first_name()
    return f"""
User: {first_name}
Message: {user_prompt}

{context if context else "No specific tender data needed for this query."}

CRITICAL: Only provide REAL tender information. Address the user as {first_name}.
"""

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message":"B-Max AI Assistant API is running!","status":"healthy" if ollama_available else "degraded"}

@app.get("/health")
async def health_check():
    tender_count = len(scan_all_tenders()) if dynamodb else 0
    return {"status":"ok","total_tenders":tender_count,"cognito": "connected" if cognito else "disabled","ollama":"connected" if ollama_available else "disconnected","active_sessions":len(user_sessions)}

@app.post("/chat")
async def chat(request: ChatRequest):
    cleanup_old_sessions()
    session = get_user_session(request.user_id)
    enhanced_prompt = enhance_prompt_with_context(request.prompt, session)
    session.add_message("user", enhanced_prompt)
    try:
        response = client.chat('deepseek-v3.1:671b-cloud', messages=session.chat_context)
        response_text = response['message']['content']
    except:
        response_text = f"I apologize {session.get_first_name()}, AI service temporarily unavailable."
    session.add_message("assistant", response_text)
    return {"response":response_text,"user_id":request.user_id,"username":session.get_first_name(),"timestamp":datetime.now().isoformat()}

@app.get("/session/{user_id}")
async def get_session_info(user_id: str):
    if user_id not in user_sessions:
        return {"error":"Session not found"}
    session = user_sessions[user_id]
    return {"user_id":session.user_id,"username":session.get_first_name(),"message_count":session.total_messages,"last_active":session.last_active.isoformat()}

@app.delete("/session/{user_id}")
async def clear_session(user_id: str):
    if user_id in user_sessions:
        del user_sessions[user_id]
        return {"message":"Session cleared successfully"}
    return {"message":"Session not found"}

if __name__ == "__main__":
    port = int(os.getenv("PORT",8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
