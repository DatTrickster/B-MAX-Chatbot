import os
import uvicorn
import boto3
import time
import re
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
app = FastAPI(title="B-Max AI Assistant", version="1.0.0")
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
    user_id: str = "guest"

# ----------------------------------------------------------------------
# Content Filter (unchanged)
# ----------------------------------------------------------------------
class ContentFilter:
    def __init__(self):
        self.bad = [
            "nigger", "nigga", "chink", "spic", "kike", "raghead", "towelhead",
            "cracker", "honky", "wetback", "gook", "dyke", "fag", "faggot",
            "tranny", "retard", "midget", "fuck", "shit", "asshole", "bitch",
            "cunt", "pussy", "dick", "cock", "whore", "slut", "motherfucker",
            "bastard", "douchebag", "shithead", "dipshit", "kill you", "hurt you",
            "rape", "murder", "suicide", "bomb", "terrorist", "naked", "nude",
            "porn", "sex", "sexual", "fuck you", "suck my", "kill myself",
            "end my life", "self harm"
        ]
        self.tender_words = [
            "tender", "bid", "proposal", "procurement", "contract", "rfp", "rfq",
            "government", "municipal", "supply", "service", "construction", "it",
            "engineering", "consulting", "maintenance", "logistics", "healthcare",
            "document", "deadline", "closing", "submission", "requirements",
            "specification", "evaluation", "award", "vendor", "supplier",
            "category", "agency", "department", "opportunity", "business",
            "company", "industry", "sector", "project", "work", "job",
            "price", "quotation", "estimate", "budget", "cost",
            "download", "link", "pdf", "attachment", "contact", "email", "phone"
        ]

    def should_respond(self, prompt: str):
        low = prompt.lower()
        if any(w in low for w in self.bad):
            return False, "I apologize, but I cannot respond to that type of content. I'm here to help with tender-related questions."
        if any(w in low for w in self.tender_words):
            return True, None
        if any(p in low for p in ["who are you", "hello", "hi ", "hey ", "help", "thank"]):
            return True, None
        return False, "I'm a tender assistant only. Ask about tenders, bids, or documents."

content_filter = ContentFilter()

# ----------------------------------------------------------------------
# DynamoDB Helpers
# ----------------------------------------------------------------------
def dd_to_py(item):
    if not item: return {}
    res = {}
    for k, v in item.items():
        if "S" in v: res[k] = v["S"]
        elif "N" in v: res[k] = int(v["N"]) if v["N"].isdigit() else float(v["N"])
        elif "BOOL" in v: res[k] = v["BOOL"]
        elif "SS" in v: res[k] = v["SS"]
        elif "M" in v: res[k] = dd_to_py(v["M"])
        elif "L" in v: res[k] = [dd_to_py(i) for i in v["L"]]
    return res

def get_user_profile_by_user_id(uid: str):
    try:
        r = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="userId = :u",
            ExpressionAttributeValues={":u": {"S": uid}},
        )
        return dd_to_py(r.get("Items", [{}])[0]) if r.get("Items") else None
    except Exception: return None

def get_cognito_user(username: str):
    if not cognito or not COGNITO_USER_POOL_ID: return None
    try:
        resp = cognito.admin_get_user(UserPoolId=COGNITO_USER_POOL_ID, Username=username)
        attrs = {a["Name"]: a["Value"] for a in resp.get("UserAttributes", [])}
        return {"user_id": resp.get("UserSub"), "email": attrs.get("email")}
    except Exception: return None

# ----------------------------------------------------------------------
# Document Links — ONLY `link` field
# ----------------------------------------------------------------------
def extract_primary_link(tender):
    url = tender.get("link", "").strip()
    return url if url and (url.startswith("http://") or url.startswith("https://")) else None

def format_tender_block(tender, reasons=None):
    title = tender.get("title", "No title")
    ref = tender.get("referenceNumber", "N/A")
    cat = tender.get("Category", "Unknown")
    agency = tender.get("sourceAgency", "Unknown")
    close = tender.get("closingDate", "Unknown").split("T")[0] if tender.get("closingDate") else "Unknown"
    link = extract_primary_link(tender)

    block = f"**{title}**\n"
    block += f"• **Reference**: {ref}\n"
    block += f"• **Category**: {cat}\n"
    block += f"• **Agency**: {agency}\n"
    block += f"• **Closing Date**: {close}\n"
    if link:
        block += f"• **Document**: [Download Tender Documents]({link})\n"
    else:
        block += "• **Document Links**: No direct links available\n"

    if reasons:
        block += f"\n   Reasons: {', '.join(reasons)}\n"
    return block

# ----------------------------------------------------------------------
# Load Tenders + Agencies
# ----------------------------------------------------------------------
def embed_tender_table():
    global embedded_tender_table, last_table_update, available_agencies
    if not dynamodb: return
    print("Embedding tenders...")
    tenders = []
    lek = None
    while True:
        params = {"TableName": DYNAMODB_TABLE_TENDERS}
        if lek: params["ExclusiveStartKey"] = lek
        resp = dynamodb.scan(**params)
        for i in resp.get("Items", []): tenders.append(dd_to_py(i))
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
# Context Builder
# ----------------------------------------------------------------------
def build_system_context(name: str, prefs: dict, agencies: set):
    pc = ', '.join(prefs.get("preferredCategories", [])) or "None"
    ps = len(prefs.get("preferredSites", []))

    ctx = f"""You are B-Max, a professional tender assistant.

CRITICAL RULES:
1. ALWAYS start reply with: "{name},"
2. NEVER mention database or technical details.
3. ONLY use data below.
4. DOCUMENT LINKS:
   - Use ONLY the `link` field.
   - NEVER show `sourceUrl`.
   - Format: [Download Tender Documents](<link>)
   - If missing → "No direct document link available"
5. RECOMMENDATIONS:
   - Use USER PREFERENCES.
   - Max 6 tenders.
   - Include reasons.
6. If no match → "No matching tenders found."

USER: {name}
PREFERENCES:
• Categories: {pc}
• Sites: {ps}

AVAILABLE AGENCIES:
{', '.join(sorted(agencies)[:30])}{"..." if len(agencies) > 30 else ""}

DATABASE: Full access to all tenders."""
    return ctx

# ----------------------------------------------------------------------
# Recommendation Engine
# ----------------------------------------------------------------------
def recommend_tenders(prompt: str, tenders: list, prefs: dict):
    low = prompt.lower()
    pcats = [c.lower() for c in prefs.get("preferredCategories", [])]
    psites = [s.lower() for s in prefs.get("preferredSites", [])]

    scored = []
    for t in tenders:
        score = 0
        reasons = []

        cat = t.get("Category", "").lower()
        if any(c in cat for c in pcats):
            score += 15
            reasons.append("Preferred category")

        src = t.get("sourceUrl", "").lower()
        if any(s in src for s in psites):
            score += 10
            reasons.append("Preferred source")

        for field in ["title", "description", "Category"]:
            val = t.get(field, "").lower()
            if any(w in val for w in low.split() if len(w) > 3):
                score += 5
                reasons.append(f"Keyword in {field}")

        if extract_primary_link(t):
            score += 8
            reasons.append("Has document")

        cd = t.get("closingDate")
        if cd:
            try:
                dt = datetime.fromisoformat(cd.replace("Z", "+00:00"))
                days = (dt - datetime.now()).days
                if 0 <= days <= 7:
                    score += 6
                    reasons.append("Closing soon")
            except: pass

        if score:
            scored.append({"tender": t, "score": score, "reasons": reasons})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:6]

# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------
class Session:
    def __init__(self, uid):
        self.user_id = uid
        self.profile = self._load_profile()
        self.chat = []
        self.last_active = datetime.now()
        self.msg_count = 0
        self._rebuild_system()

    def _load_profile(self):
        if not dynamodb:
            return {"firstName": "User", "preferredCategories": [], "preferredSites": []}

        # Direct UUID
        p = get_user_profile_by_user_id(self.user_id)
        if p: return p

        # Cognito username → UUID
        c = get_cognito_user(self.user_id)
        if c and c.get("user_id"):
            p = get_user_profile_by_user_id(c["user_id"])
            if p: return p

        return {"firstName": "User", "preferredCategories": [], "preferredSites": []}

    def name(self):
        return self.profile.get("firstName", "User")

    def prefs(self):
        return {
            "preferredCategories": self.profile.get("preferredCategories", []),
            "preferredSites": self.profile.get("preferredSites", []),
        }

    def _rebuild_system(self):
        ctx = build_system_context(self.name(), self.prefs(), available_agencies)
        sys = {"role": "system", "content": ctx}
        if not self.chat or self.chat[0]["role"] != "system":
            self.chat = [sys]
        else:
            self.chat[0] = sys

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
# Prompt + Relevant Tenders
# ----------------------------------------------------------------------
def build_prompt(prompt: str, sess: Session):
    tenders = get_tenders()
    recs = recommend_tenders(prompt, tenders, sess.prefs())

    rel_block = ""
    if recs:
        rel_block = "RELEVANT TENDERS (USE THESE FIRST):\n\n"
        for i, r in enumerate(recs, 1):
            rel_block += f"{i}. {format_tender_block(r['tender'], r['reasons'])}\n"
    else:
        rel_block = "No relevant tenders found.\n"

    return f"""
User: {sess.name()}
Message: {prompt}

{rel_block}

INSTRUCTIONS:
- Respond naturally
- Start with "{sess.name()},"
- Use only data above
- Follow all rules in system prompt
"""

# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "B-Max Ready",
        "tenders": len(get_tenders()),
        "sessions": len(user_sessions),
        "agencies": len(available_agencies),
        "ts": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    if not ollama_available:
        raise HTTPException(503, "AI unavailable")

    ok, msg = content_filter.should_respond(req.prompt)
    if not ok:
        return {"response": msg, "filtered": True, "user_id": req.user_id, "username": "User"}

    sess = get_session(req.user_id)
    prompt = build_prompt(req.prompt, sess)
    sess.add("user", prompt)

    try:
        resp = client.chat("deepseek-v3.1:671b-cloud", messages=sess.context())
        answer = resp["message"]["content"]
    except Exception:
        answer = f"{sess.name()}, sorry, I had an issue. Try again."

    sess.add("assistant", answer)
    return {
        "response": answer,
        "filtered": False,
        "user_id": req.user_id,
        "username": sess.name(),
        "full_name": sess.name(),
        "timestamp": datetime.now().isoformat(),
        "total_messages": sess.msg_count
    }

@app.on_event("startup")
async def start():
    embed_tender_table()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
