import os
import uvicorn
import boto3
import time
import re
import difflib
from datetime import datetime
from typing import List, Dict, Any
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

# Ollama
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

# In-memory
user_sessions = {}
embedded_tender_table = None
last_table_update = None
available_agencies = set()

class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "guest"

# --- Content Filter ---
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
            'kill myself', 'end my life', 'suicide', 'self harm'
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
        for keyword in self.inappropriate_keywords:
            if keyword in text_lower:
                return True
        return False

    def is_tender_related(self, text):
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in self.tender_keywords if keyword in text_lower)
        if keyword_count >= 1:
            return True
        ai_related = any(phrase in text_lower for phrase in [
            'who are you', 'what are you', 'your name', 'your purpose',
            'hello', 'hi ', 'hey ', 'good morning', 'good afternoon', 'good evening',
            'help', 'assist', 'support', 'thank', 'thanks', 'bye', 'goodbye'
        ])
        return ai_related

    def should_respond(self, prompt):
        if self.contains_inappropriate_content(prompt):
            return False, "I apologize, but I cannot respond to that type of content. I'm here to help with tender-related questions and business opportunities."
        if not self.is_tender_related(prompt):
            return False, "I'm sorry, but I'm specifically designed to assist with tender-related questions and business opportunities through TenderConnect. I can help you find tender information, document links, categories, and recommendations."
        return True, None

content_filter = ContentFilter()

# --- DynamoDB Helpers ---
def dd_to_py(item):
    if not item:
        return {}
    result = {}
    for k, v in item.items():
        if 'S' in v: result[k] = v['S']
        elif 'N' in v:
            n = v['N']
            result[k] = int(n) if n.isdigit() else float(n)
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
        response = cognito.admin_get_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=username
        )
        user_attributes = {}
        for attr in response.get('UserAttributes', []):
            user_attributes[attr['Name']] = attr['Value']
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

# --- Document Links ---
def extract_document_links(tender):
    links = []
    if 'link' in tender and tender['link']:
        link_value = tender['link'].strip()
        if link_value and (link_value.startswith('http://') or link_value.startswith('https://')):
            links.append({'type': 'Primary Document', 'url': link_value, 'is_primary': True})
        elif link_value and link_value not in ['', 'null', 'None']:
            links.append({'type': 'Primary Document', 'url': link_value, 'is_primary': True})

    link_fields = ['documentLink', 'documents', 'tenderDocuments', 'bidDocuments',
                   'attachmentLinks', 'relatedDocuments', 'document_url',
                   'bid_documents', 'tender_documents', 'attachments']
    for field in link_fields:
        if field in tender and tender[field]:
            field_value = tender[field]
            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, str) and item.strip():
                        item = item.strip()
                        if item.startswith(('http://', 'https://')):
                            links.append({'type': field, 'url': item, 'is_primary': False})
            elif isinstance(field_value, str) and field_value.strip():
                field_value = field_value.strip()
                if field_value.startswith(('http://', 'https://')):
                    links.append({'type': field, 'url': field_value, 'is_primary': False})

    text_fields = ['description', 'title', 'additionalInfo', 'noticeDetails', 'details']
    for field in text_fields:
        if field in tender and tender[field]:
            found_links = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', str(tender[field]))
            for link in found_links:
                if not link.startswith(('http://', 'https://')):
                    link = 'https://' + link
                if link not in [l['url'] for l in links]:
                    links.append({'type': f'found_in_{field}', 'url': link, 'is_primary': False})
    return links

def format_tender_with_links(tender):
    title = tender.get('title', 'No title')
    reference = tender.get('referenceNumber', 'N/A')
    category = tender.get('Category', 'Unknown')
    agency = tender.get('sourceAgency', 'Unknown')
    closing_date = tender.get('closingDate', 'Unknown')
    status = tender.get('status', 'Unknown')

    document_links = extract_document_links(tender)
    primary_links = [l for l in document_links if l.get('is_primary')]
    secondary_links = [l for l in document_links if not l.get('is_primary')]

    response = f"**{title}**\n\n"
    response += f"• **Reference**: `{reference}`\n"
    response += f"• **Category**: {category}\n"
    response += f"• **Agency**: {agency}\n"
    response += f"• **Closing Date**: {closing_date}\n"
    response += f"• **Status**: {status}\n\n"

    if primary_links:
        response += "**Document Links**\n"
        for link in primary_links:
            url = link['url']
            response += f"**PRIMARY DOCUMENT**: [Download Tender Documents]({url})\n"
        if secondary_links:
            response += "\n"
    elif secondary_links:
        response += "**Document Links**\n"

    for i, link in enumerate(secondary_links, 1):
        link_type = link['type'].replace('_', ' ').title()
        url = link['url']
        response += f"{i}. [{link_type}]({url})\n"

    if not document_links:
        response += "• **Document Links**: No direct links available\n\n"

    source_url = tender.get('sourceUrl')
    if source_url and source_url not in [l['url'] for l in document_links]:
        response += f"• **Source Page**: [View Original Tender]({source_url})\n"

    response += "\n" + "─" * 40 + "\n"
    return response

# --- Agency & Embed ---
def extract_available_agencies(tenders):
    global available_agencies
    agencies = {t.get('sourceAgency', '').strip() for t in tenders if t.get('sourceAgency')}
    available_agencies = agencies
    return agencies

def embed_tender_table():
    global embedded_tender_table, last_table_update
    try:
        if not dynamodb:
            return None
        print("Embedding ProcessedTender table...")
        all_tenders = []
        last_evaluated_key = None
        while True:
            resp = dynamodb.scan(TableName=DYNAMODB_TABLE_TENDERS, ExclusiveStartKey=last_evaluated_key) if last_evaluated_key else dynamodb.scan(TableName=DYNAMODB_TABLE_TENDERS)
            items = resp.get('Items', [])
            for item in items:
                all_tenders.append(dd_to_py(item))
            last_evaluated_key = resp.get('LastEvaluatedKey')
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
    if (embedded_tender_table is None or last_table_update is None or
        (datetime.now() - last_table_update).total_seconds() > 1800):
        return embed_tender_table()
    return embedded_tender_table

# --- Advanced Search ---
def advanced_search(user_prompt: str, tenders: List[Dict], user_preferences: Dict) -> List[Dict]:
    prompt_low = user_prompt.lower()
    words = [w for w in prompt_low.split() if len(w) > 2]
    pref_cats = {c.lower() for c in user_preferences.get("preferredCategories", [])}
    pref_sites = {s.lower() for s in user_preferences.get("preferredSites", [])}

    scored = []
    for tender in tenders:
        title = (tender.get("title") or "").lower()
        ref = (tender.get("referenceNumber") or "").lower()
        cat = (tender.get("Category") or "").lower()
        agency = (tender.get("sourceAgency") or "").lower()
        source_url = (tender.get("sourceUrl") or "").lower()
        desc = (tender.get("description") or "").lower()

        score = 0
        reasons = []

        if any(a in agency for a in words) or any(a in prompt_low for a in agency.split()):
            score += 30; reasons.append("Agency match")
        elif words and any(difflib.SequenceMatcher(None, w, agency).ratio() > 0.7 for w in words):
            score += 25; reasons.append("Fuzzy agency")

        if cat in pref_cats: score += 15; reasons.append(f"Preferred: {cat.title()}")
        if any(w in cat for w in words): score += 12; reasons.append("Category keyword")

        if any(w in title for w in words): score += 10; reasons.append("Title keyword")
        if words:
            best = max((difflib.SequenceMatcher(None, w, title).ratio() for w in words), default=0)
            if best > 0.6: score += int(best * 10); reasons.append("Fuzzy title")

        if any(w in ref for w in words): score += 8; reasons.append("Reference match")
        if desc and any(w in desc for w in words): score += 6; reasons.append("Description keyword")
        if any(s in source_url for s in pref_sites): score += 7; reasons.append("Preferred source")

        primary = [l for l in extract_document_links(tender) if l.get("is_primary")]
        if primary: score += 9; reasons.append("Primary document")
        elif extract_document_links(tender): score += 3; reasons.append("Has document")

        cd = tender.get("closingDate", "")
        if cd and cd != "Unknown":
            try:
                dt = datetime.fromisoformat(cd.replace("Z", "+00:00"))
                if 0 <= (dt - datetime.now()).days <= 7:
                    score += 5; reasons.append("Closing soon")
            except: pass

        if score > 0:
            scored.append({"tender": tender, "score": score, "reasons": reasons})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:6]

# --- Table Summary ---
def format_embedded_table_for_ai(tenders, user_preferences=None):
    if not tenders:
        return "No tender data available"
    total = len(tenders)
    categories = {}
    agencies = {}
    with_links = sum(1 for t in tenders if extract_document_links(t))
    for t in tenders:
        cat = t.get('Category', 'Unknown')
        agency = t.get('sourceAgency', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
        agencies[agency] = agencies.get(agency, 0) + 1

    summary = f"**TENDER DATABASE** ({total} tenders)\n\n"
    summary += f"• **With Documents**: {with_links} ({with_links/total*100:.1f}%)\n"
    summary += f"• **Categories**: {len(categories)}\n"
    summary += f"• **Agencies**: {len(agencies)}\n\n"

    if available_agencies:
        summary += "**Available Agencies**\n"
        for a in sorted(list(available_agencies))[:15]:
            summary += f"• {a}\n"
        if len(available_agencies) > 15:
            summary += f"• ...and {len(available_agencies)-15} more\n"
        summary += "\n"

    if user_preferences and user_preferences.get('preferredCategories'):
        summary += f"**Your Preferred Categories**\n"
        for c in user_preferences['preferredCategories']:
            summary += f"• {c}\n"
        summary += "\n"

    summary += "**Top Categories**\n"
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
        summary += f"• {cat}: {count}\n"
    summary += "\n"
    summary += "**Rules**\n"
    summary += "• Only use data from this table\n"
    summary += "• Use `link` field for downloads\n"
    summary += "• Never invent tenders\n"
    return summary

# --- Session ---
class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = None
        self.chat_context = []
        self.last_active = datetime.now()
        self.total_messages = 0
        self.load_user_profile()
        self.initialize_chat_context(self.get_first_name())

    def initialize_chat_context(self, first_name: str):
        tenders = get_embedded_table()
        prefs = self.get_user_preferences()
        table_ctx = format_embedded_table_for_ai(tenders, prefs) if tenders else "No data"
        system_prompt = f"""You are B-Max, a professional AI assistant for TenderConnect.

RULES:
1. Address user as **{first_name}**
2. Use **only** the embedded tender database
3. **Never** invent data
4. Format cleanly: **Title**, • Bullet, [Link](url)
5. If no match: "No matching tenders found"
6. Document links: **Only** from `link` field

USER:
- Name: {first_name}
- Company: {self.user_profile.get('companyName', 'Not specified') if self.user_profile else 'Not specified'}

DATABASE:
{table_ctx}

RESPONSE STYLE:
- Professional
- Structured
- No fluff
- Use exact DB values
"""
        if not self.chat_context:
            self.chat_context = [{"role": "system", "content": system_prompt}]
        else:
            self.chat_context[0] = {"role": "system", "content": system_prompt}

    def load_user_profile(self):
        try:
            if not dynamodb:
                self.user_profile = {'firstName': 'User'}
                return
            profile = get_user_profile_by_user_id(self.user_id)
            if not profile and '@' in self.user_id:
                profile = get_user_profile_by_email(self.user_id)
            if not profile:
                cog_user = get_cognito_user_by_username(self.user_id)
                if cog_user and cog_user.get('user_id'):
                    profile = get_user_profile_by_user_id(cog_user['user_id'])
            self.user_profile = profile or {'firstName': 'User'}
        except Exception as e:
            self.user_profile = {'firstName': 'User'}

    def get_first_name(self):
        return self.user_profile.get('firstName', 'User') if self.user_profile else "User"

    def get_user_preferences(self):
        return {
            'preferredCategories': self.user_profile.get('preferredCategories', []) if self.user_profile else [],
            'preferredSites': self.user_profile.get('preferredSites', []) if self.user_profile else []
        }

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

def get_user_session(user_id: str) -> UserSession:
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

# --- Prompt Enhancement ---
def enhance_prompt_with_context(user_prompt: str, session: UserSession) -> str:
    tenders = get_embedded_table()
    prefs = session.get_user_preferences()
    search_results = advanced_search(user_prompt, tenders, prefs) if tenders else []

    if search_results:
        personalized_context = "**Recommended Tenders**\n\n"
        for i, rec in enumerate(search_results, 1):
            tender_formatted = format_tender_with_links(rec["tender"])
            reasons = rec["reasons"]
            personalized_context += tender_formatted
            if reasons:
                personalized_context += f"**Why this tender?** {', '.join(reasons)}\n\n"
        personalized_context = personalized_context.strip()
    else:
        personalized_context = (
            "**No matching tenders found.**\n\n"
            "Try using keywords from:\n"
            "• Tender title\n"
            "• Reference number\n"
            "• Agency name\n"
            "• Category (e.g., IT Services, Construction)\n\n"
            "Example: _'eTenders construction'_"
        )

    database_context = format_embedded_table_for_ai(tenders, prefs) if tenders else "No data"

    return f"""
User: {session.get_first_name()}
Message: {user_prompt}

DATABASE:
{database_context}

RECOMMENDATIONS:
{personalized_context}

INSTRUCTIONS:
- Use only data above
- Format professionally
- Never invent
"""

# --- Endpoints ---
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
    tenders = get_embedded_table()
    cleanup_old_sessions()
    return {
        "status": "ok",
        "service": "B-Max AI Assistant",
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
        try:
            resp = client.chat('deepseek-v3.1:671b-cloud', messages=session.get_chat_context())
            response_text = resp['message']['content']
        except Exception as e:
            response_text = f"I apologize {session.get_first_name()}, but I'm having technical issues. Please try again."
        session.add_message("assistant", response_text)
        return {"response": response_text, "filtered": False, **_base_response(request, session)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _base_response(request, session=None):
    session = session or get_user_session(request.user_id)
    return {
        "user_id": request.user_id,
        "username": session.get_first_name(),
        "full_name": f"{session.user_profile.get('firstName', '')} {session.user_profile.get('lastName', '')}".strip() or "User",
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
    embed_tender_table()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
