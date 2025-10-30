import os
import uvicorn
import time
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import boto3

load_dotenv()

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "af-south-1")
DYNAMODB_TABLE_TENDERS = os.getenv("DYNAMODB_TABLE_DEST", "ProcessedTender")
DYNAMODB_TABLE_USERS   = os.getenv("DYNAMODB_TABLE_USERS", "UserProfiles")
COGNITO_USER_POOL_ID   = os.getenv("COGNITO_USER_POOL_ID")

# ----------------------------------------------------------------------
# AWS clients
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
    print("AWS Clients (DynamoDB + Cognito) initialized")
except Exception as e:
    print(f"AWS init error: {e}")
    dynamodb = None
    cognito = None

# ----------------------------------------------------------------------
# Ollama (optional)
# ----------------------------------------------------------------------
try:
    from ollama import Client
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
    client = (
        Client(host="https://ollama.com", headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})
        if OLLAMA_API_KEY
        else None
    )
    ollama_available = client is not None
    print("Ollama ready" if ollama_available else "Ollama not configured")
except Exception as e:
    ollama_available = False
    print(f"Ollama error: {e}")

# ----------------------------------------------------------------------
# FastAPI app
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
# In-memory caches
# ----------------------------------------------------------------------
user_sessions = {}
embedded_tender_table = None
last_table_update = None
available_agencies = set()


# ----------------------------------------------------------------------
# Request model
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "guest"


# ----------------------------------------------------------------------
# Content filter (unchanged)
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
            return False, "I cannot respond to inappropriate content."
        if any(w in low for w in self.tender_words):
            return True, None
        if any(p in low for p in ["who are you", "hello", "hi ", "hey ", "help", "thank"]):
            return True, None
        return False, "I’m a tender-assistant only. Ask about tenders, bids, or documents."

content_filter = ContentFilter()


# ----------------------------------------------------------------------
# DynamoDB helpers
# ----------------------------------------------------------------------
def dd_to_py(item):
    if not item:
        return {}
    res = {}
    for k, v in item.items():
        if "S" in v:
            res[k] = v["S"]
        elif "N" in v:
            res[k] = int(v["N"]) if v["N"].isdigit() else float(v["N"])
        elif "BOOL" in v:
            res[k] = v["BOOL"]
        elif "SS" in v:
            res[k] = v["SS"]
        elif "M" in v:
            res[k] = dd_to_py(v["M"])
        elif "L" in v:
            res[k] = [dd_to_py(i) for i in v["L"]]
    return res


def get_user_profile_by_user_id(uid: str):
    try:
        r = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="userId = :u",
            ExpressionAttributeValues={":u": {"S": uid}},
        )
        return dd_to_py(r.get("Items", [{}])[0]) if r.get("Items") else None
    except Exception as e:
        print(f"User profile scan error: {e}")
        return None


def get_cognito_user(username: str):
    if not cognito or not COGNITO_USER_POOL_ID:
        return None
    try:
        resp = cognito.admin_get_user(UserPoolId=COGNITO_USER_POOL_ID, Username=username)
        attrs = {a["Name"]: a["Value"] for a in resp.get("UserAttributes", [])}
        return {"user_id": resp.get("UserSub"), "email": attrs.get("email")}
    except Exception as e:
        print(f"Cognito error: {e}")
        return None


# ----------------------------------------------------------------------
# Document link handling – ONLY the `link` field
# ----------------------------------------------------------------------
def extract_document_links(tender):
    """Return **only** the `link` field if present."""
    if tender.get("link"):
        url = tender["link"].strip()
        if url:
            return [{"type": "Primary Document", "url": url, "is_primary": True}]
    return []


def format_tender_with_links(tender):
    title = tender.get("title", "No title")
    ref   = tender.get("referenceNumber", "N/A")
    cat   = tender.get("Category", "Unknown")
    agency = tender.get("sourceAgency", "Unknown")
    close = tender.get("closingDate", "Unknown")
    stat  = tender.get("status", "Unknown")

    out = f"**{title}**\n"
    out += f"• **Reference**: {ref}\n"
    out += f"• **Category**: {cat}\n"
    out += f"• **Agency**: {agency}\n"
    out += f"• **Closing Date**: {close}\n"
    out += f"• **Status**: {stat}\n"

    links = extract_document_links(tender)
    if links:
        out += f"• **Document**: [Download Tender Documents]({links[0]['url']})\n"
    else:
        out += "• **Document Links**: No direct links available\n"
    return out


# ----------------------------------------------------------------------
# Load all tenders + agencies
# ----------------------------------------------------------------------
def embed_tender_table():
    global embedded_tender_table, last_table_update, available_agencies
    if not dynamodb:
        return None
    print("Embedding tenders...")
    tenders = []
    lek = None
    while True:
        params = {"TableName": DYNAMODB_TABLE_TENDERS}
        if lek:
            params["ExclusiveStartKey"] = lek
        resp = dynamodb.scan(**params)
        for i in resp.get("Items", []):
            tenders.append(dd_to_py(i))
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break

    embedded_tender_table = tenders
    last_table_update = datetime.now()
    available_agencies = {t.get("sourceAgency", "").strip() for t in tenders if t.get("sourceAgency")}
    print(f"Embedded {len(tenders)} tenders, {len(available_agencies)} agencies")
    return tenders


def get_embedded_table():
    global embedded_tender_table, last_table_update
    if (
        embedded_tender_table is None
        or last_table_update is None
        or (datetime.now() - last_table_update).total_seconds() > 1800
    ):
        embed_tender_table()
    return embedded_tender_table or []


# ----------------------------------------------------------------------
# Context for the LLM
# ----------------------------------------------------------------------
def format_embedded_table_for_ai(tenders, prefs=None):
    if not tenders:
        return "No tender data."

    total = len(tenders)
    cat_cnt = {}
    agency_cnt = {}
    stat_cnt = {}
    with_doc = 0
    for t in tenders:
        cat_cnt[t.get("Category", "Unknown")] = cat_cnt.get(t.get("Category", "Unknown"), 0) + 1
        agency_cnt[t.get("sourceAgency", "Unknown")] = agency_cnt.get(t.get("sourceAgency", "Unknown"), 0) + 1
        stat_cnt[t.get("status", "Unknown")] = stat_cnt.get(t.get("status", "Unknown"), 0) + 1
        if extract_document_links(t):
            with_doc += 1

    txt = "=== TENDER DATABASE CONTEXT ===\n\n"
    txt += f"Total tenders: {total}\n"
    txt += f"Categories: {len(cat_cnt)}\n"
    txt += f"Agencies: {len(agency_cnt)}\n"
    txt += f"Statuses: {len(stat_cnt)}\n"
    txt += f"Tenders with document link: {with_doc}\n\n"

    txt += "ALL AGENCIES:\n"
    for a in sorted(available_agencies):
        txt += f"• {a}\n"
    txt += "\n"

    if prefs:
        pc = prefs.get("preferredCategories", [])
        ps = prefs.get("preferredSites", [])
        txt += f"USER PREFERENCES:\n"
        txt += f"• Categories: {', '.join(pc) if pc else 'None'}\n"
        txt += f"• Sites: {len(ps)} source(s)\n\n"

    txt += "TOP CATEGORIES:\n"
    for c, n in sorted(cat_cnt.items(), key=lambda x: x[1], reverse=True)[:6]:
        txt += f"• {c}: {n}\n"
    txt += "\n"

    txt += "DOCUMENT LINK POLICY (MANDATORY):\n"
    txt += "- ONLY the `link` field is a downloadable document.\n"
    txt += "- NEVER show `sourceUrl` or any other URL.\n"
    txt += "- If `link` missing → “No direct document link available”.\n"
    txt += "- Format: [Download Tender Documents](<exact-link>)\n"

    return txt


# ----------------------------------------------------------------------
# Recommendation engine (uses preferences)
# ----------------------------------------------------------------------
def get_relevant_tenders(prompt: str, tenders: list, prefs: dict):
    if not tenders:
        return []

    low_prompt = prompt.lower()
    pref_cats = [c.lower() for c in prefs.get("preferredCategories", [])]
    pref_sites = [s.lower() for s in prefs.get("preferredSites", [])]

    scored = []

    for t in tenders:
        score = 0
        reasons = []

        # ---- category preference ----
        tcat = t.get("Category", "").lower()
        if any(pc in tcat for pc in pref_cats):
            score += 12
            reasons.append("Matches preferred category")

        # ---- site preference ----
        tsrc = t.get("sourceUrl", "").lower()
        if any(ps in tsrc for ps in pref_sites):
            score += 8
            reasons.append("From preferred source")

        # ---- keyword match in key fields ----
        for field in ["title", "description", "Category", "sourceAgency"]:
            val = t.get(field, "").lower()
            matches = sum(1 for w in low_prompt.split() if len(w) > 3 and w in val)
            if matches:
                score += matches * 4
                reasons.append(f"{matches} keyword(s) in {field}")

        # ---- has document link ----
        if extract_document_links(t):
            score += 6
            reasons.append("Has document link")

        # ---- urgency ----
        cd = t.get("closingDate")
        if cd:
            try:
                dt = datetime.fromisoformat(cd.replace("Z", "+00:00"))
                days = (dt - datetime.now()).days
                if 0 <= days <= 7:
                    score += 5
                    reasons.append("Closing soon")
            except:
                pass

        if score:
            scored.append({"tender": t, "score": score, "reasons": reasons})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:12]  # generous but still manageable


# ----------------------------------------------------------------------
# Session handling (name + preferences)
# ----------------------------------------------------------------------
class UserSession:
    def __init__(self, uid):
        self.user_id = uid
        self.profile = None
        self.chat = []
        self.last_active = datetime.now()
        self.msg_count = 0
        self.load_profile()
        self.rebuild_system_prompt()

    # --------------------------------------------------------------
    def load_profile(self):
        """Cognito username → sub (UUID) → UserProfiles"""
        if not dynamodb:
            self.profile = {"firstName": "User", "preferredCategories": [], "preferredSites": []}
            return

        # 1. Direct UUID?
        p = get_user_profile_by_user_id(self.user_id)
        if p:
            self.profile = p
            return

        # 2. Cognito → UUID
        c = get_cognito_user(self.user_id)
        if c and c.get("user_id"):
            p = get_user_profile_by_user_id(c["user_id"])
            if p:
                self.profile = p
                return

        # 3. Email fallback
        if "@" in self.user_id:
            # (optional – you already have get_user_profile_by_email if you want)
            pass

        self.profile = {"firstName": "User", "preferredCategories": [], "preferredSites": []}

    # --------------------------------------------------------------
    def first0(self):
        return self.profile.get("firstName", "User")

    # --------------------------------------------------------------
    def preferences(self):
        return {
            "preferredCategories": self.profile.get("preferredCategories", []),
            "preferredSites": self.profile.get("preferredSites", []),
        }

    # --------------------------------------------------------------
    def rebuild_system_prompt(self):
        tenders = get_embedded_table()
        ctx = format_embedded_table_for_ai(tenders, self.preferences())
        name = self.first0()

        sys = f"""You are B-Max, a friendly tender assistant.

RULES YOU MUST FOLLOW:
1. ALWAYS start every reply with the user's first name: "{name}".
2. NEVER mention that you have a database or any technical details.
3. ONLY use the data that appears in the sections below.
4. DOCUMENT LINKS:
   • ONLY the `link` field is a downloadable file.
   • NEVER show `sourceUrl` or any other URL.
   • If `link` missing → say “No direct document link available”.
   • Format: [Download Tender Documents](<exact-link>)
5. RECOMMENDATIONS:
   • When the user asks for recommendations, use the USER PREFERENCES section.
   • Prefer tenders that match preferred categories / sites.
   • List up to 6 best matches with reasons.
6. If nothing matches → “No matching tenders found in the database.”

USER NAME: {name}
USER PREFERENCES:
{ctx.split('USER PREFERENCES:')[1].split('TOP CATEGORIES:')[0] if 'USER PREFERENCES:' in ctx else ''}

FULL DATABASE CONTEXT:
{ctx}
"""
        if not self.chat or self.chat[0]["role"] != "system":
            self.chat = [{"role": "system", "content": sys}]
        else:
            self.chat[0] = {"role": "system", "content": sys}

    # --------------------------------------------------------------
    def add(self, role, content):
        self.chat.append({"role": role, "content": content})
        self.msg_count += 1
        if len(self.chat) > 22:  # keep system + 21 recent
            self.chat = [self.chat[0]] + self.chat[-21:]
        self.last_active = datetime.now()

    # --------------------------------------------------------------
    def context(self):
        return self.chat


def get_session(uid: str) -> UserSession:
    if uid not in user_sessions:
        user_sessions[uid] = UserSession(uid)
    sess = user_sessions[uid]
    sess.last_active = datetime.now()
    return sess


def cleanup_sessions():
    now = datetime.now()
    dead = [uid for uid, s in user_sessions.items() if (now - s.last_active).total_seconds() > 7200]
    for uid in dead:
        del user_sessions[uid]


# ----------------------------------------------------------------------
# Prompt augmentation (adds relevant tenders)
# ----------------------------------------------------------------------
def augment_prompt(prompt: str, sess: UserSession) -> str:
    tenders = get_embedded_table()
    prefs = sess.preferences()
    relevant = get_relevant_tenders(prompt, tenders, prefs)

    rel_block = ""
    if relevant:
        rel_block = "RELEVANT TENDERS (USE THESE FIRST):\n\n"
        for i, r in enumerate(relevant, 1):
            rel_block += f"{i}. {format_tender_with_links(r['tender'])}\n"
            rel_block += f"   Reasons: {', '.join(r['reasons'])}\n\n"
    else:
        rel_block = "No relevant tenders found.\n"

    return f"""
User name: {sess.first0()}
User message: {prompt}

{rel_block}

FULL DATABASE CONTEXT:
{format_embedded_table_for_ai(tenders, prefs)}
"""


# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------
@app.get("/")
async def root():
    t = get_embedded_table()
    return {
        "message": "B-Max AI Assistant",
        "status": "ok" if ollama_available else "degraded",
        "tenders": len(t),
        "sessions": len(user_sessions),
        "agencies": len(available_agencies),
        "ts": datetime.now().isoformat(),
    }


@app.get("/health")
async def health():
    cleanup_sessions()
    t = get_embedded_table()
    return {
        "status": "ok",
        "tenders": len(t),
        "sessions": len(user_sessions),
        "agencies": len(available_agencies),
        "ollama": ollama_available,
        "ts": datetime.now().isoformat(),
    }


@app.get("/agencies")
async def agencies():
    return {
        "agencies": sorted(list(available_agencies)),
        "count": len(available_agencies),
        "ts": datetime.now().isoformat(),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if not ollama_available:
        raise HTTPException(503, "AI service unavailable")

    ok, msg = content_filter.should_respond(req.prompt)
    if not ok:
        return {"response": msg, "filtered": True, **_base(req)}

    sess = get_session(req.user_id)
    augmented = augment_prompt(req.prompt, sess)
    sess.add("user", augmented)

    try:
        resp = client.chat("deepseek-v3.1:671b-cloud", messages=sess.context())
        answer = resp["message"]["content"]
    except Exception as e:
        answer = f"Sorry {sess.first0()}, I hit a snag. Try again in a moment."

    sess.add("assistant", answer)
    return {"response": answer, "filtered": False, **_base(req, sess)}


def _base(req: ChatRequest, sess: UserSession | None = None):
    sess = sess or get_session(req.user_id)
    return {
        "user_id": req.user_id,
        "username": sess.first0(),
        "full_name": sess.first0(),
        "timestamp": datetime.now().isoformat(),
        "session_active": True,
        "total_messages": sess.msg_count,
    }


@app.get("/session-info/{uid}")
async def session_info(uid: str):
    s = user_sessions.get(uid)
    if not s:
        return {"error": "not found"}
    return {
        "user_id": uid,
        "first_name": s.first0(),
        "messages": s.msg_count,
        "context_len": len(s.chat),
        "last_active": s.last_active.isoformat(),
    }


@app.on_event("startup")
async def start():
    print("Loading tenders at startup...")
    embed_tender_table()
    print("Ready!")


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
