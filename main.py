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
        print("AWS Clients (DynamoDB + Cognito) initialized successfully")
    else:
        cognito = None
        print("AWS DynamoDB client initialized (Cognito disabled)")
       
except Exception as e:
    print(f"AWS Client initialization error: {e}")
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
        print("Ollama client initialized successfully")
    else:
        ollama_available = False
        print("Ollama API key not found")
except ImportError:
    ollama_available = False
    print("Ollama package not installed")
except Exception as e:
    ollama_available = False
    print(f"Ollama client initialization error: {e}")

app = FastAPI(title="B-Max AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session & data
user_sessions = {}
embedded_tender_table = None
last_table_update = None
available_agencies = set()

class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "guest"

# --- Content Filtering ---
class ContentFilter:
    def __init__(self):
        self.inappropriate_keywords = [
            'nigger', 'nigga', 'chink', 'spic', 'kike', 'raghead', 'towelhead', 'cracker', 'honky',
            'wetback', 'gook', 'dyke', 'fag', 'faggot', 'tranny', 'retard', 'midget',
            'fuck', 'shit', 'asshole', 'bitch', 'cunt', 'pussy', 'dick', 'cock', 'whore', 'slut',
            'motherfucker', 'bastard', 'douchebag', 'shithead', 'dipshit',
            'kill you', 'hurt you', 'attack you', 'destroy you', 'harm you', 'beat you',
            'rape', 'murder', 'suicide', 'bomb', 'terrorist',
            'naked', 'nude', 'porn', 'sex', 'sexual', 'fuck you', 'suck my',
            'kill myself', 'end my life', 'self harm'
        ]
        self.tender_keywords = [
            'tender', 'bid', 'proposal', 'procurement', 'contract', 'rfp', 'rfq',
            'government', 'municipal', 'supply', 'service', 'construction', 'it',
            'engineering', 'consulting', 'maintenance', 'logistics', 'healthcare',
            'document', 'deadline', 'closing', 'submission', 'requirements',
            'specification', 'evaluation', 'award', 'vendor', 'supplier',
            'category', 'agency', 'department', 'opportunity', 'business',
            'company', 'industry', 'sector', 'project', 'work', 'job',
            'price', 'quotation', 'estimate', 'budget', 'cost',
            'compliance', 'regulation', 'policy', 'guideline',
            'download', 'link', 'pdf', 'document', 'attachment',
            'contact', 'email', 'phone', 'address', 'location'
        ]

    def contains_inappropriate_content(self, text):
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.inappropriate_keywords)

    def is_tender_related(self, text):
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.tender_keywords):
            return True
        return any(phrase in text_lower for phrase in [
            'who are you', 'what are you', 'your name', 'your purpose',
            'hello', 'hi ', 'hey ', 'good morning', 'good afternoon', 'good evening',
            'help', 'assist', 'support', 'thank', 'thanks', 'bye', 'goodbye'
        ])

    def should_respond(self, prompt):
        if self.contains_inappropriate_content(prompt):
            return False, "I apologize, but I cannot respond to that type of content. I'm here to help with tender-related questions and business opportunities."
        if not self.is_tender_related(prompt):
            return False, "I'm sorry, but I'm specifically designed to assist with tender-related questions and business opportunities through TenderConnect."
        return True, None

content_filter = ContentFilter()

# --- DynamoDB Helpers ---
def dd_to_py(item):
    if not item:
        return {}
    result = {}
    for k, v in item.items():
        if 'S' in v: result[k] = v['S']
        elif 'N' in v: result[k] = int(v['N']) if v['N'].isdigit() else float(v['N'])
        elif 'BOOL' in v: result[k] = v['BOOL']
        elif 'SS' in v: result[k] = v['SS']
        elif 'M' in v: result[k] = dd_to_py(v['M'])
        elif 'L' in v: result[k] = [dd_to_py(el) for el in v['L']]
    return result

def get_user_profile_by_user_id(user_id: str):
    try:
        resp = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="userId = :uid",
            ExpressionAttributeValues={":uid": {"S": user_id}}
        )
        items = resp.get("Items", [])
        return dd_to_py(items[0]) if items else None
    except Exception as e:
        print(f"Error scanning for user profile: {e}")
        return None

def get_user_profile_by_email(email: str):
    try:
        resp = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="email = :email",
            ExpressionAttributeValues={":email": {"S": email}}
        )
        items = resp.get("Items", [])
        return dd_to_py(items[0]) if items else None
    except Exception as e:
        print(f"Error scanning for user by email: {e}")
        return None

def get_cognito_user_by_username(username: str):
    try:
        if not cognito or not COGNITO_USER_POOL_ID:
            return None
        response = cognito.admin_get_user(UserPoolId=COGNITO_USER_POOL_ID, Username=username)
        user_attributes = {attr['Name']: attr['Value'] for attr in response.get('UserAttributes', [])}
        user_sub = response.get('UserSub')
        return {
            'username': response.get('Username'),
            'user_id': user_sub,
            'email': user_attributes.get('email'),
            'email_verified': user_attributes.get('email_verified', 'false') == 'true',
            'status': response.get('UserStatus'),
            'enabled': response.get('Enabled', False),
            'created': response.get('UserCreateDate'),
            'modified': response.get('UserLastModifiedDate'),
            'attributes': user_attributes
        }
    except Exception as e:
        print(f"Error fetching Cognito user {username}: {e}")
        return None

# --- Document Links: ONLY USE `link` FIELD ---
def extract_document_links(tender):
    """Extract **ONLY** the official document link from the `link` field."""
    links = []
    if 'link' in tender and tender['link']:
        link_value = tender['link'].strip()
        if link_value:
            links.append({
                'type': 'Primary Document',
                'url': link_value,
                'is_primary': True
            })
    return links

def format_tender_with_links(tender):
    """Format tender with **only** the `link` field as download."""
    title = tender.get('title', 'No title')
    reference = tender.get('referenceNumber', 'N/A')
    category = tender.get('Category', 'Unknown')
    agency = tender.get('sourceAgency', 'Unknown')
    closing_date = tender.get('closingDate', 'Unknown')
    status = tender.get('status', 'Unknown')

    document_links = extract_document_links(tender)

    formatted = f"**{title}**\n"
    formatted += f"• **Reference**: {reference}\n"
    formatted += f"• **Category**: {category}\n"
    formatted += f"• **Agency**: {agency}\n"
    formatted += f"• **Closing Date**: {closing_date}\n"
    formatted += f"• **Status**: {status}\n"

    if document_links:
        primary = document_links[0]
        formatted += f"• **Document**: [Download Tender Documents]({primary['url']})\n"
    else:
        formatted += f"• **Document Links**: No direct links available\n"

    return formatted

# --- Agencies & Embedding ---
def extract_available_agencies(tenders):
    global available_agencies
    agencies = {tender.get('sourceAgency', '').strip() for tender in tenders if tender.get('sourceAgency')}
    available_agencies = agencies
    print(f"Updated available agencies: {len(agencies)} found")
    return agencies

def embed_tender_table():
    global embedded_tender_table, last_table_update
    try:
        if not dynamodb:
            print("DynamoDB client not available")
            return None

        print("Embedding ProcessedTender table...")
        all_tenders = []
        last_evaluated_key = None

        while True:
            params = {"TableName": DYNAMODB_TABLE_TENDERS}
            if last_evaluated_key:
                params["ExclusiveStartKey"] = last_evaluated_key
            response = dynamodb.scan(**params)
            items = response.get('Items', [])
            for item in items:
                all_tenders.append(dd_to_py(item))
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break

        embedded_tender_table = all_tenders
        last_table_update = datetime.now()
        extract_available_agencies(all_tenders)

        print(f"Embedded {len(all_tenders)} tenders")
        return all_tenders
    except Exception as e:
        print(f"Error embedding table: {e}")
        return None

def get_embedded_table():
    global embedded_tender_table, last_table_update
    if (embedded_tender_table is None or
        last_table_update is None or
        (datetime.now() - last_table_update).total_seconds() > 1800):
        return embed_tender_table()
    return embedded_tender_table

def format_embedded_table_for_ai(tenders, user_preferences=None):
    if not tenders:
        return "EMBEDDED PROCESSEDTENDER TABLE: No data available"

    total_tenders = len(tenders)
    categories = {}
    agencies = {}
    statuses = {}
    tenders_with_links = 0

    for tender in tenders:
        categories[tender.get('Category', 'Unknown')] = categories.get(tender.get('Category', 'Unknown'), 0) + 1
        agencies[tender.get('sourceAgency', 'Unknown')] = agencies.get(tender.get('sourceAgency', 'Unknown'), 0) + 1
        statuses[tender.get('status', 'Unknown')] = statuses.get(tender.get('status', 'Unknown'), 0) + 1
        if extract_document_links(tender):
            tenders_with_links += 1

    summary = "COMPLETE TENDER DATABASE CONTEXT:\n\n"
    summary += f"📊 DATABASE OVERVIEW:\n"
    summary += f"• Total Tenders: {total_tenders}\n"
    summary += f"• Categories: {len(categories)}\n"
    summary += f"• Agencies: {len(agencies)}\n"
    summary += f"• Statuses: {len(statuses)}\n"
    summary += f"• Tenders with Document Links: {tenders_with_links} ({tenders_with_links/total_tenders*100:.1f}%)\n\n"

    if available_agencies:
        summary += "AVAILABLE TENDER SOURCE AGENCIES (ALL):\n"
        for agency in sorted(available_agencies):
            summary += f"• {agency}\n"
        summary += "\n"

    if user_preferences:
        preferred = user_preferences.get('preferredCategories', [])
        summary += f"USER PREFERENCES:\n"
        summary += f"• Preferred Categories: {', '.join(preferred) if preferred else 'None'}\n\n"

    summary += "TOP CATEGORIES:\n"
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:6]:
        summary += f"• {cat}: {count} tenders\n"
    summary += "\n"

    summary += "MANDATORY DOCUMENT-LINK RULE:\n"
    summary += "- ONLY the value of the `link` field may be given to the user.\n"
    summary += "- NEVER return `sourceUrl`, any URL in description, or other fields.\n"
    summary += "- If `link` is missing → say \"No direct document link available\".\n"
    summary += "- Format: [Download Tender Documents](<exact-link-from-DB>)\n"

    summary += "\nAGENCY AWARENESS: You can find tenders from ANY agency in the list above. Use them all.\n"

    return summary

def get_relevant_tenders(user_prompt: str, tenders: list, user_preferences: dict):
    if not tenders:
        return []

    user_prompt_lower = user_prompt.lower()
    preferred_categories = [c.lower() for c in user_preferences.get('preferredCategories', [])]
    preferred_sites = [s.lower() for s in user_preferences.get('preferredSites', [])]

    scored = []

    for tender in tenders:
        score = 0
        reasons = []

        cat = tender.get('Category', '').lower()
        if any(pc in cat for pc in preferred_categories):
            score += 10
            reasons.append("Preferred category")

        src = tender.get('sourceUrl', '').lower()
        if any(ps in src for ps in preferred_sites):
            score += 5
            reasons.append("Preferred source")

        fields = ['title', 'description', 'Category', 'sourceAgency']
        for field in fields:
            val = tender.get(field, '').lower()
            if any(word in val for word in user_prompt_lower.split() if len(word) > 3):
                score += 3
                reasons.append(f"Matches in {field}")

        if extract_document_links(tender):
            score += 5
            reasons.append("Has document link")

        closing = tender.get('closingDate', '')
        if closing:
            try:
                dt = datetime.fromisoformat(closing.replace('Z', '+00:00'))
                if 0 <= (dt - datetime.now()).days <= 7:
                    score += 4
                    reasons.append("Closing soon")
            except:
                pass

        if score > 0:
            scored.append({'tender': tender, 'score': score, 'match_reasons': reasons})

    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:20]

# --- User Session ---
class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = None
        self.cognito_user = None
        self.chat_context = []
        self.last_active = datetime.now()
        self.total_messages = 0
        self.session_id = f"{user_id}_{int(time.time())}"
        self.load_user_profile()
        self.initialize_chat_context(self.get_first_name())

    def initialize_chat_context(self, first_name: str):
        tenders = get_embedded_table()
        prefs = self.get_user_preferences()
        table_context = format_embedded_table_for_ai(tenders, prefs) if tenders else "No data."

        system_prompt = f"""You are B-Max, an AI assistant for TenderConnect.

CRITICAL RULES:
1. Address user as "{first_name}" in every response.
2. NEVER mention database or capabilities.
3. Use ONLY the embedded database.
4. Be natural, warm, and helpful.
5. DOCUMENT LINK POLICY (MUST OBEY):
   • ONLY the `link` field contains the downloadable document.
   • NEVER output `sourceUrl` or any other URL.
   • If `link` missing → say "No direct document link available".
   • Format: [Download Tender Documents](<exact-link>)
6. Use ALL agencies listed in context.
7. If no match: "No matching tenders found in the database."

USER:
- Name: {first_name}
- Preferences: {', '.join(prefs.get('preferredCategories', [])) or 'None'}

DATABASE CONTEXT:
{table_context}

RESPONSE RULES:
- Natural & conversational
- Use emojis sparingly
- Only use data from context
- Never invent links or search externally
"""

        if not self.chat_context:
            self.chat_context = [{"role": "system", "content": system_prompt}]
        else:
            self.chat_context[0] = {"role": "system", "content": system_prompt}

    def load_user_profile(self):
        if not dynamodb:
            self.user_profile = self.create_default_profile()
            return

        profile = get_user_profile_by_user_id(self.user_id)
        if profile:
            self.user_profile = profile
            return

        if '@' in self.user_id:
            profile = get_user_profile_by_email(self.user_id)
            if profile:
                self.user_profile = profile
                return

        self.user_profile = self.create_default_profile()

    def create_default_profile(self):
        return {'firstName': 'User', 'preferredCategories': []}

    def get_user_preferences(self):
        return {
            'preferredCategories': self.user_profile.get('preferredCategories', []),
            'preferredSites': self.user_profile.get('preferredSites', [])
        }

    def get_first_name(self):
        return self.user_profile.get('firstName', 'User')

    def update_activity(self):
        self.last_active = datetime.now()

    def add_message(self, role, content):
        if not self.chat_context or self.chat_context[0]["role"] != "system":
            self.initialize_chat_context(self.get_first_name())
        self.chat_context.append({"role": role, "content": content})
        self.total_messages += 1
        if len(self.chat_context) > 20:
            self.chat_context = [self.chat_context[0]] + self.chat_context[-19:]

    def get_chat_context(self):
        if not self.chat_context or self.chat_context[0]["role"] != "system":
            self.initialize_chat_context(self.get_first_name())
        return self.chat_context

def get_user_session(user_id: str):
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    session = user_sessions[user_id]
    session.update_activity()
    return session

def cleanup_old_sessions():
    now = datetime.now()
    expired = [uid for uid, s in user_sessions.items() if (now - s.last_active).total_seconds() > 7200]
    for uid in expired:
        del user_sessions[uid]

def enhance_prompt_with_context(user_prompt: str, session: UserSession):
    tenders = get_embedded_table()
    prefs = session.get_user_preferences()

    relevant = get_relevant_tenders(user_prompt, tenders, prefs)
    relevant_context = "RELEVANT TENDERS (USE ONLY THESE):\n\n" + "\n\n".join(
        format_tender_with_links(rec['tender']) + f" ✅ {', '.join(rec['match_reasons'])}"
        for rec in relevant
    ) if relevant else "No relevant tenders found."

    database_context = format_embedded_table_for_ai(tenders, prefs)

    return f"""
User: {session.get_first_name()}
Message: {user_prompt}

DATABASE CONTEXT:
{database_context}

{relevant_context}

INSTRUCTIONS:
- Respond to {session.get_first_name()}
- Use only provided data
- Never invent links
- Document rule: ONLY `link` field
"""

# --- API Endpoints ---
@app.get("/")
async def root():
    tenders = get_embedded_table()
    return {
        "message": "B-Max AI Assistant",
        "status": "healthy" if ollama_available else "degraded",
        "embedded_tenders": len(tenders) if tenders else 0,
        "active_sessions": len(user_sessions),
        "available_agencies": len(available_agencies),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    cleanup_old_sessions()
    tenders = get_embedded_table()
    return {
        "status": "ok",
        "embedded_tenders": len(tenders) if tenders else 0,
        "active_sessions": len(user_sessions),
        "available_agencies": len(available_agencies),
        "ollama_available": ollama_available,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/agencies")
async def get_agencies():
    return {
        "agencies": sorted(list(available_agencies)),
        "count": len(available_agencies),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not ollama_available:
            raise HTTPException(status_code=503, detail="AI service unavailable")

        should_respond, msg = content_filter.should_respond(request.prompt)
        if not should_respond:
            return {"response": msg, "filtered": True, **_base_response(request)}

        session = get_user_session(request.user_id)
        enhanced = enhance_prompt_with_context(request.prompt, session)
        session.add_message("user", enhanced)

        response = client.chat('deepseek-v3.1:671b-cloud', messages=session.get_chat_context())
        reply = response['message']['content']
        session.add_message("assistant", reply)

        return {"response": reply, "filtered": False, **_base_response(request, session)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _base_response(request, session=None):
    session = session or get_user_session(request.user_id)
    return {
        "user_id": request.user_id,
        "username": session.get_first_name(),
        "full_name": session.get_first_name(),
        "timestamp": datetime.now().isoformat(),
        "session_active": True,
        "total_messages": session.total_messages
    }

@app.get("/session-info/{user_id}")
async def get_session_info(user_id: str):
    session = user_sessions.get(user_id)
    if not session:
        return {"error": "Session not found"}
    return {
        "user_id": user_id,
        "first_name": session.get_first_name(),
        "total_messages": session.total_messages,
        "context_length": len(session.chat_context),
        "last_active": session.last_active.isoformat()
    }

@app.on_event("startup")
async def startup_event():
    print("Initializing embedded tender table...")
    embed_tender_table()
    print("Startup complete")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting B-Max AI Assistant on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
