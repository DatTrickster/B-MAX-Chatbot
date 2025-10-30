import os
import uvicorn
import time
import boto3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "af-south-1")
DYNAMODB_TABLE_TENDERS = os.getenv("DYNAMODB_TABLE_DEST", "ProcessedTender")
DYNAMODB_TABLE_USERS = os.getenv("DYNAMODB_TABLE_USERS", "UserProfiles")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")

# ----------------------------------------------------------------------
# AWS Clients
# ----------------------------------------------------------------------
try:
    dynamodb = boto3.client(
        "dynamodb",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )
    cognito = (
        boto3.client(
            "cognito-idp",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION,
        )
        if COGNITO_USER_POOL_ID
        else None
    )
except Exception as e:
    print(f"AWS init failed: {e}")
    dynamodb = None
    cognito = None

# ----------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------
try:
    from ollama import Client
    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
    ) if os.getenv("OLLAMA_API_KEY") else None
    ollama_available = client is not None
except Exception:
    ollama_available = False
    client = None

# ----------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# In-memory
# ----------------------------------------------------------------------
user_sessions = {}
embedded_tender_table = None
last_table_update = None
available_agencies = set()

class ChatRequest(BaseModel):
    prompt: str
    user_id: str  # Can be Cognito username OR UUID

# ----------------------------------------------------------------------
# DynamoDB Helper
# ----------------------------------------------------------------------
def dd_to_py(item):
    if not item:
        return {}
    res = {}
    for k, v in item.items():
        if "S" in v: res[k] = v["S"]
        elif "N" in v: res[k] = int(v["N"]) if v["N"].isdigit() else float(v["N"])
        elif "BOOL" in v: res[k] = v["BOOL"]
        elif "SS" in v: res[k] = v["SS"]
        elif "M" in v: res[k] = dd_to_py(v["M"])
        elif "L" in v: res[k] = [dd_to_py(i) for i in v["L"]]
    return res

# ----------------------------------------------------------------------
# User Profile Resolver (Cognito → UUID → Profile)
# ----------------------------------------------------------------------
def resolve_user_profile(user_id: str):
    """
    1. If user_id is UUID → direct lookup
    2. Else → treat as Cognito username → get sub → lookup by userId
    3. Fallback → default profile
    """
    if not dynamodb:
        return {"firstName": "User", "preferredCategories": [], "preferredSites": []}

    # Case 1: Direct UUID lookup
    if len(user_id) == 36 and user_id.count("-") == 4:
        resp = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="userId = :uid",
            ExpressionAttributeValues={":uid": {"S": user_id}}
        )
        items = resp.get("Items")
        if items:
            return dd_to_py(items[0])

    # Case 2: Cognito username → get sub (UUID)
    if cognito and COGNITO_USER_POOL_ID:
        try:
            resp = cognito.admin_get_user(
                UserPoolId=COGNITO_USER_POOL_ID,
                Username=user_id
            )
            sub = resp.get("UserSub")  # This is the UUID
            if sub:
                resp2 = dynamodb.scan(
                    TableName=DYNAMODB_TABLE_USERS,
                    FilterExpression="userId = :uid",
                    ExpressionAttributeValues={":uid": {"S": sub}}
                )
                items = resp2.get("Items")
                if items:
                    return dd_to_py(items[0])
        except Exception as e:
            print(f"Cognito lookup failed for {user_id}: {e}")

    # Fallback
    return {"firstName": "User", "preferredCategories": [], "preferredSites": []}

# ----------------------------------------------------------------------
# Document Links: ONLY `link`
# ----------------------------------------------------------------------
def extract_document_links(tender):
    url = tender.get("link", "").strip()
    return [{"url": url}] if url else []

def format_tender_with_links(tender):
    out = f"**{tender.get('title', 'No title')}**\n"
    out += f"• **Reference**: {tender.get('referenceNumber', 'N/A')}\n"
    out += f"• **Category**: {tender.get('Category', 'Unknown')}\n"
    out += f"• **Agency**: {tender.get('sourceAgency', 'Unknown')}\n"
    out += f"• **Closing Date**: {tender.get('closingDate', 'Unknown')}\n"
    links = extract_document_links(tender)
    if links:
        out += f"• **Document**: [Download Tender Documents]({links[0]['url']})\n"
    else:
        out += "• **Document Links**: No direct links available\n"
    return out

# ----------------------------------------------------------------------
# Load Tenders
# ----------------------------------------------------------------------
def embed_tender_table():
    global embedded_tender_table, last_table_update, available_agencies
    if not dynamodb:
        return
    print("Loading tenders...")
    tenders = []
    lek = None
    while True:
        params = {"TableName": DYNAMODB_TABLE_TENDERS}
        if lek: params["ExclusiveStartKey"] = lek
        resp = dynamodb.scan(**params)
        for i in resp.get("Items", []):
            tenders.append(dd_to_py(i))
        lek = resp.get("LastEvaluatedKey")
        if not lek: break
    embedded_tender_table = tenders
    last_table_update = datetime.now()
    available_agencies = {t.get("sourceAgency", "").strip() for t in tenders if t.get("sourceAgency")}
    print(f"Loaded {len(tenders)} tenders, {len(available_agencies)} agencies")

def get_tenders():
    global embedded_tender_table, last_table_update
    if not embedded_tender_table or (datetime.now() - last_table_update).total_seconds() > 1800:
        embed_tender_table()
    return embedded_tender_table or []

# ----------------------------------------------------------------------
# Context & Recommendations
# ----------------------------------------------------------------------
def build_context(tenders, profile):
    cats = {}
    for t in tenders:
        c = t.get("Category", "Unknown")
        cats[c] = cats.get(c, 0) + 1

    ctx = "=== TENDER DATABASE ===\n"
    ctx += f"Total: {len(tenders)}\n"
    ctx += f"Agencies: {len(available_agencies)}\n\n"

    ctx += "ALL AGENCIES:\n"
    for a in sorted(available_agencies):
        ctx += f"• {a}\n"
    ctx += "\n"

    ctx += "USER PREFERENCES:\n"
    pc = profile.get("preferredCategories", [])
    ps = profile.get("preferredSites", [])
    ctx += f"• Categories: {', '.join(pc) if pc else 'None'}\n"
    ctx += f"• Sites: {len(ps)}\n\n"

    ctx += "TOP CATEGORIES:\n"
    for c, n in sorted(cats.items(), key=lambda x: x[1], reverse=True)[:6]:
        ctx += f"• {c}: {n}\n"
    ctx += "\n"

    ctx += "DOCUMENT POLICY:\n"
    ctx += "- ONLY `link` field = downloadable PDF\n"
    ctx += "- NEVER show `sourceUrl`\n"
    ctx += "- Format: [Download Tender Documents](<link>)\n"

    return ctx

def recommend_tenders(prompt: str, tenders: list, profile: dict):
    pref_cats = [c.lower() for c in profile.get("preferredCategories", [])]
    pref_sites = [s.lower() for s in profile.get("preferredSites", [])]
    prompt_low = prompt.lower()

    scored = []
    for t in tenders:
        score = 0
        reasons = []

        if t.get("Category", "").lower() in pref_cats:
            score += 15
            reasons.append("Matches preferred category")

        if any(ps in t.get("sourceUrl", "").lower() for ps in pref_sites):
            score += 12
            reasons.append("From preferred source")

        for f in ["title", "description", "Category", "sourceAgency"]:
            val = t.get(f, "").lower()
            if any(w in val for w in prompt_low.split() if len(w) > 3):
                score += 5
                reasons.append(f"Keyword match in {f}")

        if extract_document_links(t):
            score += 8
            reasons.append("Has document link")

        cd = t.get("closingDate", "")
        if cd:
            try:
                dt = datetime.fromisoformat(cd.replace("Z", "+00:00"))
                days = (dt - datetime.now()).days
                if 0 <= days <= 7:
                    score += 7
                    reasons.append("Closing soon")
            except:
                pass

        if score > 0:
            scored.append({"tender": t, "score": score, "reasons": reasons})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:6]

# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------
class Session:
    def __init__(self, user_id):
        self.user_id = user_id
        self.profile = resolve_user_profile(user_id)  # ← NOW RESOLVES COGNITO → UUID → PROFILE
        self.chat = []
        self.last_active = datetime.now()
        self.msg_count = 0
        self.rebuild_system()

    def first_name(self):
        return self.profile.get("firstName", "User")

    def rebuild_system(self):
        tenders = get_tenders()
        ctx = build_context(tenders, self.profile)
        name = self.first_name()

        sys_prompt = f"""You are B-Max, a tender assistant.

RULES:
1. ALWAYS start reply with: "{name},"
2. Use only data below.
3. For recommendations: use USER PREFERENCES.
4. Document links: ONLY `link` field → [Download Tender Documents](<link>)
5. NEVER show `sourceUrl`.
6. If no match: "No matching tenders found."

USER: {name}
PREFERENCES:
- Categories: {', '.join(self.profile.get('preferredCategories', [])) or 'None'}
- Sites: {len(self.profile.get('preferredSites', []))}

DATABASE:
{ctx}
"""
        if not self.chat or self.chat[0]["role"] != "system":
            self.chat = [{"role": "system", "content": sys_prompt}]
        else:
            self.chat[0]["content"] = sys_prompt

    def add(self, role, content):
        self.chat.append({"role": role, "content": content})
        self.msg_count += 1
        if len(self.chat) > 25:
            self.chat = [self.chat[0]] + self.chat[-24:]
        self.last_active = datetime.now()

    def context(self):
        return self.chat

def get_session(uid):
    if uid not in user_sessions:
        user_sessions[uid] = Session(uid)
    s = user_sessions[uid]
    s.last_active = datetime.now()
    return s

# ----------------------------------------------------------------------
# Prompt Builder
# ----------------------------------------------------------------------
def build_prompt(user_prompt: str, session: Session):
    tenders = get_tenders()
    relevant = recommend_tenders(user_prompt, tenders, session.profile)

    rel_block = "RELEVANT TENDERS:\n\n"
    if relevant:
        for i, r in enumerate(relevant, 1):
            rel_block += f"{i}. {format_tender_with_links(r['tender'])}\n"
            rel_block += f"   Reasons: {', '.join(r['reasons'])}\n\n"
    else:
        rel_block += "No relevant tenders found.\n"

    return f"""
User: {session.first_name()}
Message: {user_prompt}

{rel_block}

Use only above data. Follow all rules.
"""

# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
@app.get("/")
async def root():
    t = get_tenders()
    return {
        "message": "B-Max Ready",
        "tenders": len(t),
        "agencies": len(available_agencies),
        "sessions": len(user_sessions),
        "ts": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    if not ollama_available:
        raise HTTPException(503, "AI unavailable")

    session = get_session(req.user_id)
    prompt = build_prompt(req.prompt, session)
    session.add("user", prompt)

    try:
        resp = client.chat("deepseek-v3.1:671b-cloud", messages=session.context())
        reply = resp["message"]["content"]
    except Exception as e:
        reply = f"{session.first_name()}, sorry, I had an issue. Try again."

    session.add("assistant", reply)
    return {
        "response": reply,
        "user": session.first_name(),
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def start():
    embed_tender_table()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
