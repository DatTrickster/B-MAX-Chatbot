import os
import json
import boto3
import uvicorn
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from difflib import SequenceMatcher

# ========== ENV + CONFIG ==========
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "af-south-1")
DYNAMODB_TABLE_TENDERS = os.getenv("DYNAMODB_TABLE_DEST", "ProcessedTender")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# ========== INIT FASTAPI ==========
app = FastAPI(
    title="B-Max AI Assistant",
    description="AI-powered tender analysis and categorization system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== INIT AWS CLIENTS ==========
try:
    dynamodb = boto3.client(
        "dynamodb",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )

    cognito = None
    if COGNITO_USER_POOL_ID:
        cognito = boto3.client(
            "cognito-idp",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION,
        )
    print("✅ AWS Clients (DynamoDB + Cognito) initialized successfully")
except Exception as e:
    print(f"❌ AWS initialization error: {e}")
    dynamodb = None
    cognito = None

# ========== INIT OLLAMA CLIENT ==========
try:
    from ollama import Client

    if OLLAMA_API_KEY:
        ollama = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        )
        ollama_available = True
        print("✅ Ollama client initialized successfully")
    else:
        ollama_available = False
        ollama = None
        print("⚠️ Ollama API key not found")
except Exception as e:
    ollama_available = False
    ollama = None
    print(f"❌ Ollama initialization error: {e}")

# ========== GLOBAL MEMORY ==========
user_sessions = {}

# ========== HELPERS ==========
def dd_to_py(item):
    """Convert DynamoDB JSON to normal Python dict"""
    if not item:
        return {}
    result = {}
    for k, v in item.items():
        if "S" in v:
            result[k] = v["S"]
        elif "N" in v:
            n = v["N"]
            result[k] = int(n) if n.isdigit() else float(n)
        elif "BOOL" in v:
            result[k] = v["BOOL"]
        elif "SS" in v:
            result[k] = v["SS"]
        elif "M" in v:
            result[k] = dd_to_py(v["M"])
        elif "L" in v:
            result[k] = [dd_to_py(el) for el in v["L"]]
    return result


def scan_all_tenders():
    """Scan all tenders from DynamoDB"""
    try:
        if not dynamodb:
            return []

        all_tenders = []
        last_evaluated_key = None

        while True:
            params = {"TableName": DYNAMODB_TABLE_TENDERS, "Limit": 100}
            if last_evaluated_key:
                params["ExclusiveStartKey"] = last_evaluated_key

            response = dynamodb.scan(**params)
            items = response.get("Items", [])
            for item in items:
                all_tenders.append(dd_to_py(item))

            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break

        print(f"📊 Scanned {len(all_tenders)} total tenders from database")
        return all_tenders
    except Exception as e:
        print(f"❌ Error scanning tenders: {e}")
        return []


# ========== MODELS ==========
class ChatRequest(BaseModel):
    user_id: str
    prompt: str


# ========== ENDPOINTS ==========
@app.get("/")
async def root():
    return {
        "message": "B-Max AI Assistant API is live 🚀",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Check service + DB health"""
    db_status = "connected" if dynamodb else "disconnected"
    tender_count = 0
    if dynamodb:
        tenders = scan_all_tenders()
        tender_count = len(tenders)

    return {
        "status": "ok",
        "service": "B-Max Chatbot",
        "dynamodb": db_status,
        "tenders": tender_count,
        "ollama": "connected" if ollama_available else "disabled",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/categories")
async def get_categories():
    """Return all unique categories"""
    try:
        if not dynamodb:
            raise HTTPException(status_code=503, detail="Database not connected")

        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS, ProjectionExpression="Category"
        )
        items = response.get("Items", [])
        categories = set()

        for item in items:
            cat = item.get("Category", {}).get("S", "").strip()
            if cat:
                categories.add(cat)

        while "LastEvaluatedKey" in response:
            response = dynamodb.scan(
                TableName=DYNAMODB_TABLE_TENDERS,
                ProjectionExpression="Category",
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            for item in response.get("Items", []):
                cat = item.get("Category", {}).get("S", "").strip()
                if cat:
                    categories.add(cat)

        return {"categories": sorted(categories)}
    except Exception as e:
        print(f"❌ Category fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tenders/{category}")
async def get_tenders_by_category(category: str, limit: int = 5):
    """Get tenders by category"""
    try:
        all_tenders = scan_all_tenders()
        filtered = [
            t for t in all_tenders if category.lower() in t.get("Category", "").lower()
        ]
        if not filtered:
            return {"message": f"No tenders found for '{category}'"}
        return {"category": category, "results": filtered[:limit]}
    except Exception as e:
        print(f"❌ Tender fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{user_id}")
async def get_session(user_id: str):
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "name": "User",
            "messages": [],
            "created_at": datetime.now().isoformat(),
        }
        print(f"🎯 Session created for {user_id}")
    return user_sessions[user_id]


@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat prompts"""
    try:
        user_id = request.user_id.strip()
        prompt = request.prompt.strip()

        print(f"💬 Chat request received - user_id: {user_id}, prompt: {prompt}")

        # Create session if missing
        if user_id not in user_sessions:
            user_sessions[user_id] = {
                "name": "User",
                "messages": [],
                "created_at": datetime.now().isoformat(),
            }

        session = user_sessions[user_id]

        # Detect tender-related queries
        if any(
            kw in prompt.lower()
            for kw in ["tender", "rfq", "bid", "project", "available"]
        ):
            print("🔍 Processing tender-related query")
            all_tenders = scan_all_tenders()

            # Find category match
            found_category = None
            for t in all_tenders:
                if SequenceMatcher(None, prompt.lower(), t.get("Category", "").lower()).ratio() > 0.5:
                    found_category = t.get("Category", "")
                    break

            if not found_category:
                found_category = "IT Services"

            print(f"🔍 Searching tenders for category: {found_category}")
            filtered = [
                t for t in all_tenders if found_category.lower() in t.get("Category", "").lower()
            ]

            response_text = f"Here are {len(filtered[:4])} recent tenders for **{found_category}**:\n\n"
            for t in filtered[:4]:
                response_text += f"• **{t.get('title', 'Untitled')}** ({t.get('referenceNumber', 'N/A')})\n  🔗 {t.get('link', '')}\n  🏛️ {t.get('sourceAgency', '')}\n  📅 Closes: {t.get('closingDate', '')}\n\n"

        else:
            response_text = (
                "I'm here to help you find and analyze tenders. Try asking about a specific category or keyword!"
            )

        # Add messages
        session["messages"].append({"role": "user", "content": prompt})
        session["messages"].append({"role": "assistant", "content": response_text})

        print(f"✅ Response sent to {user_id}")
        return {"response": response_text, "session": session}

    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== MAIN ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("🚀 Starting B-Max Chatbot Server...")
    print("📡 API running at: http://0.0.0.0:" + str(port))
    uvicorn.run(app, host="0.0.0.0", port=port)
