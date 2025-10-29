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

# In-memory session storage with embedded table data
user_sessions = {}
embedded_tender_table = None
last_table_update = None

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

def embed_tender_table():
    """Embed the entire ProcessedTender table into memory for AI context"""
    global embedded_tender_table, last_table_update
    
    try:
        if not dynamodb:
            print("❌ DynamoDB client not available")
            return None
        
        print("🧠 Embedding entire ProcessedTender table into AI context...")
        all_tenders = []
        last_evaluated_key = None
        
        # Paginate through all results
        while True:
            if last_evaluated_key:
                response = dynamodb.scan(
                    TableName=DYNAMODB_TABLE_TENDERS,
                    ExclusiveStartKey=last_evaluated_key
                )
            else:
                response = dynamodb.scan(
                    TableName=DYNAMODB_TABLE_TENDERS
                )
            
            items = response.get('Items', [])
            for item in items:
                tender = dd_to_py(item)
                all_tenders.append(tender)
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        embedded_tender_table = all_tenders
        last_table_update = datetime.now()
        
        print(f"✅ Embedded {len(all_tenders)} tenders from ProcessedTender table into AI context")
        
        # Log table statistics
        if all_tenders:
            categories = {}
            agencies = {}
            statuses = {}
            
            for tender in all_tenders:
                category = tender.get('Category', 'Unknown')
                agency = tender.get('sourceAgency', 'Unknown')
                status = tender.get('status', 'Unknown')
                
                categories[category] = categories.get(category, 0) + 1
                agencies[agency] = agencies.get(agency, 0) + 1
                statuses[status] = statuses.get(status, 0) + 1
            
            print(f"📊 ProcessedTender Table Stats - Categories: {len(categories)}, Agencies: {len(agencies)}, Statuses: {len(statuses)}")
            print(f"🏷️ Top Categories: {list(categories.keys())[:5]}")
        
        return all_tenders
        
    except Exception as e:
        print(f"❌ Error embedding ProcessedTender table: {e}")
        return None

def get_embedded_table():
    """Get the embedded tender table, refresh if stale"""
    global embedded_tender_table, last_table_update
    
    # Refresh table every 30 minutes or if not loaded
    if (embedded_tender_table is None or 
        last_table_update is None or 
        (datetime.now() - last_table_update).total_seconds() > 1800):
        return embed_tender_table()
    
    return embedded_tender_table

def format_embedded_table_for_ai(tenders):
    """Format the entire embedded ProcessedTender table for AI consumption"""
    if not tenders:
        return "EMBEDDED PROCESSEDTENDER TABLE: No data available"
    
    # Create a comprehensive summary of the table
    table_summary = "COMPLETE PROCESSEDTENDER DATABASE EMBEDDED IN CONTEXT:\n\n"
    
    # Table statistics
    total_tenders = len(tenders)
    categories = {}
    agencies = {}
    statuses = {}
    
    for tender in tenders:
        category = tender.get('Category', 'Unknown')
        agency = tender.get('sourceAgency', 'Unknown')
        status = tender.get('status', 'Unknown')
        
        categories[category] = categories.get(category, 0) + 1
        agencies[agency] = agencies.get(agency, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
    
    table_summary += f"📊 PROCESSEDTENDER DATABASE OVERVIEW:\n"
    table_summary += f"• Total Tenders: {total_tenders}\n"
    table_summary += f"• Categories: {len(categories)}\n"
    table_summary += f"• Agencies: {len(agencies)}\n"
    table_summary += f"• Statuses: {len(statuses)}\n\n"
    
    # Top categories
    table_summary += "🏷️ TOP CATEGORIES:\n"
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:8]:
        table_summary += f"• {category}: {count} tenders\n"
    
    table_summary += "\n"
    
    # Recent tenders sample with ALL fields from ProcessedTender
    table_summary += "📋 SAMPLE OF AVAILABLE TENDERS (with all ProcessedTender fields):\n"
    for i, tender in enumerate(tenders[:10], 1):
        title = tender.get('title', 'No title')
        category = tender.get('Category', 'Unknown')
        agency = tender.get('sourceAgency', 'Unknown')
        closing_date = tender.get('closingDate', 'Unknown')
        status = tender.get('status', 'Unknown')
        reference_number = tender.get('referenceNumber', 'N/A')
        contact_name = tender.get('contactName', 'N/A')
        contact_email = tender.get('contactEmail', 'N/A')
        contact_number = tender.get('contactNumber', 'N/A')
        link = tender.get('link', 'N/A')
        source_url = tender.get('sourceUrl', 'N/A')
        
        table_summary += f"🚀 TENDER #{i}:\n"
        table_summary += f"   📋 Title: {title}\n"
        table_summary += f"   🏷️ Reference: {reference_number}\n"
        table_summary += f"   📊 Category: {category}\n"
        table_summary += f"   🏢 Agency: {agency}\n"
        table_summary += f"   📅 Closing: {closing_date}\n"
        table_summary += f"   📈 Status: {status}\n"
        table_summary += f"   👤 Contact: {contact_name} | {contact_email} | {contact_number}\n"
        table_summary += f"   🔗 Documents: {link}\n"
        table_summary += f"   🌐 Source: {source_url}\n\n"
    
    table_summary += "💡 You have COMPLETE access to ALL ProcessedTender data including titles, references, categories, agencies, closing dates, contacts, and document links. Use this comprehensive information to provide accurate, detailed answers and recommendations."
    
    return table_summary

def get_intelligent_insights(user_prompt: str, tenders: list):
    """Generate intelligent insights based on the embedded ProcessedTender table"""
    if not tenders:
        return "No data available for analysis."
    
    insights = []
    user_prompt_lower = user_prompt.lower()
    
    # Category distribution insights
    categories = {}
    for tender in tenders:
        category = tender.get('Category', 'Unknown')
        categories[category] = categories.get(category, 0) + 1
    
    # Urgency insights
    urgent_tenders = []
    for tender in tenders:
        closing_date = tender.get('closingDate', '')
        if closing_date:
            try:
                closing_dt = datetime.fromisoformat(closing_date.replace('Z', '+00:00'))
                days_until_close = (closing_dt - datetime.now()).days
                if 0 <= days_until_close <= 7:
                    urgent_tenders.append(tender)
            except:
                pass
    
    # Agency insights
    agencies = {}
    for tender in tenders:
        agency = tender.get('sourceAgency', 'Unknown')
        agencies[agency] = agencies.get(agency, 0) + 1
    
    # Contact availability insights
    tenders_with_contacts = [t for t in tenders if t.get('contactName') != 'N/A' and t.get('contactName')]
    
    # Generate insights based on user query
    if any(word in user_prompt_lower for word in ['category', 'categories', 'type']):
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        insights.append(f"📊 Category Distribution: {', '.join([f'{cat} ({count})' for cat, count in top_categories])}")
    
    if any(word in user_prompt_lower for word in ['urgent', 'soon', 'closing', 'deadline']):
        insights.append(f"⏰ Urgent Opportunities: {len(urgent_tenders)} tenders closing within 7 days")
    
    if any(word in user_prompt_lower for word in ['agency', 'department', 'municipality']):
        top_agencies = sorted(agencies.items(), key=lambda x: x[1], reverse=True)[:5]
        insights.append(f"🏢 Top Agencies: {', '.join([f'{agency} ({count})' for agency, count in top_agencies])}")
    
    if any(word in user_prompt_lower for word in ['contact', 'email', 'phone', 'person']):
        insights.append(f"📞 Contact Info Available: {len(tenders_with_contacts)} tenders have contact details")
    
    # General insights
    if not insights:
        insights.append(f"📈 Database contains {len(tenders)} tenders across {len(categories)} categories")
        insights.append(f"🏢 {len(agencies)} different agencies publishing opportunities")
        if urgent_tenders:
            insights.append(f"⏰ {len(urgent_tenders)} tenders closing within 7 days")
        if tenders_with_contacts:
            insights.append(f"📞 {len(tenders_with_contacts)} tenders with contact information")
    
    return " | ".join(insights)

def find_relevant_tenders(user_prompt: str, tenders: list):
    """Find the most relevant tenders based on user query"""
    if not tenders:
        return []
    
    user_prompt_lower = user_prompt.lower()
    relevant_tenders = []
    
    # Score tenders based on relevance to query
    for tender in tenders:
        score = 0
        
        # Title relevance
        title = tender.get('title', '').lower()
        if any(word in title for word in user_prompt_lower.split()):
            score += 3
        
        # Category relevance
        category = tender.get('Category', '').lower()
        if any(word in category for word in user_prompt_lower.split()):
            score += 2
        
        # Agency relevance
        agency = tender.get('sourceAgency', '').lower()
        if any(word in agency for word in user_prompt_lower.split()):
            score += 1
        
        # Status relevance
        status = tender.get('status', '').lower()
        if 'active' in status or 'open' in status:
            score += 1
        
        if score > 0:
            relevant_tenders.append((tender, score))
    
    # Sort by relevance score and return top 5
    relevant_tenders.sort(key=lambda x: x[1], reverse=True)
    return [tender for tender, score in relevant_tenders[:5]]

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = None
        self.chat_context = []
        self.last_active = datetime.now()
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
        # Get the embedded table for system context
        tenders = get_embedded_table()
        table_context = format_embedded_table_for_ai(tenders) if tenders else "ProcessedTender database not currently available."
        
        return f"""You are B-Max, an AI assistant for TenderConnect with COMPLETE ACCESS to the entire ProcessedTender database.

CRITICAL RULES - FOLLOW THESE EXACTLY:
1. You have the ENTIRE ProcessedTender database embedded in your context with ALL fields
2. Use the complete table data to provide accurate, comprehensive answers
3. ALWAYS address the user by their first name "{first_name}"
4. Provide data-driven insights and recommendations using REAL data
5. Reference specific tenders, categories, agencies, contacts, and deadlines from the embedded table
6. Never invent or create fake tender data - you have access to all real ProcessedTender data
7. Be proactive in suggesting opportunities based on complete database knowledge

AVAILABLE TENDER FIELDS IN PROCESSEDTENDER TABLE:
- title, Category, sourceAgency, closingDate, status, referenceNumber
- contactName, contactEmail, contactNumber, link, sourceUrl
- And all other fields from the ProcessedTender table

YOUR CAPABILITIES WITH EMBEDDED PROCESSEDTENDER TABLE:
- Access to ALL tenders with complete field information
- Ability to analyze patterns and trends across the entire database
- Provide statistical insights and data-driven recommendations
- Compare and contrast different opportunities with full details
- Identify urgent opportunities and provide contact information
- Reference specific tender details like reference numbers and document links

EMBEDDED PROCESSEDTENDER DATABASE CONTEXT:
{table_context}

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMPORTANT: 
- You have COMPLETE knowledge of all available tenders in ProcessedTender table
- Use this comprehensive knowledge to provide the best recommendations
- Reference actual data points, contact details, and specific tender information
- Be specific about what's available in the database including reference numbers and links"""

    def load_user_profile(self):
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
        
        # Keep reasonable context length but preserve system message
        if len(self.chat_context) > 16:
            system_message = self.chat_context[0]
            recent_messages = self.chat_context[-15:]
            self.chat_context = [system_message] + recent_messages

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
    """Enhance prompt with embedded ProcessedTender table context and intelligent insights"""
    
    # Get the embedded table
    tenders = get_embedded_table()
    
    if not tenders:
        database_context = "⚠️ ProcessedTender database not currently available."
        insights = "No insights available."
        relevant_tenders_context = ""
    else:
        # Generate intelligent insights based on the embedded data
        insights = get_intelligent_insights(user_prompt, tenders)
        
        # Find relevant tenders for the specific query
        relevant_tenders = find_relevant_tenders(user_prompt, tenders)
        
        # Format relevant tenders context
        if relevant_tenders:
            relevant_tenders_context = "🎯 RELEVANT TENDERS FOR YOUR QUERY:\n\n"
            for i, tender in enumerate(relevant_tenders, 1):
                title = tender.get('title', 'No title')
                category = tender.get('Category', 'Unknown')
                agency = tender.get('sourceAgency', 'Unknown')
                closing_date = tender.get('closingDate', 'Unknown')
                status = tender.get('status', 'Unknown')
                reference_number = tender.get('referenceNumber', 'N/A')
                
                relevant_tenders_context += f"{i}. {title}\n"
                relevant_tenders_context += f"   📊 {category} | 🏢 {agency}\n"
                relevant_tenders_context += f"   📅 {closing_date} | 📈 {status}\n"
                relevant_tenders_context += f"   🏷️ Ref: {reference_number}\n\n"
        else:
            relevant_tenders_context = "No specific tenders match your query, but I have access to all tenders in the database."
        
        # For specific queries, provide focused context
        if any(word in user_prompt.lower() for word in ['stat', 'analytics', 'overview', 'summary']):
            database_context = format_embedded_table_for_ai(tenders)
        else:
            # Provide a concise context for regular queries
            database_context = f"COMPLETE PROCESSEDTENDER DATABASE ACCESS: {len(tenders)} tenders available with full details including contacts, references, and document links."
    
    user_first_name = session.get_first_name()
    
    enhanced_prompt = f"""
User: {user_first_name}
Message: {user_prompt}

PROCESSEDTENDER DATABASE CONTEXT:
{database_context}

INTELLIGENT INSIGHTS:
{insights}

{relevant_tenders_context}

INSTRUCTIONS:
- You have COMPLETE access to the ProcessedTender database with ALL fields
- Provide data-driven, specific recommendations with reference numbers
- Include contact information and document links when available
- Reference actual tenders from the embedded table
- Use your full knowledge of the database to provide comprehensive answers
- Be proactive in suggesting relevant opportunities with specific details
"""
    return enhanced_prompt

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    tenders = get_embedded_table()
    tender_count = len(tenders) if tenders else 0
    
    return {
        "message": "B-Max AI Assistant with Embedded ProcessedTender Database",
        "status": "healthy" if ollama_available else "degraded",
        "embedded_tenders": tender_count,
        "last_update": last_table_update.isoformat() if last_table_update else None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    tenders = get_embedded_table()
    tender_count = len(tenders) if tenders else 0
    
    # Database statistics
    categories = set()
    agencies = set()
    if tenders:
        for tender in tenders:
            categories.add(tender.get('Category', 'Unknown'))
            agencies.add(tender.get('sourceAgency', 'Unknown'))
    
    return {
        "status": "ok",
        "service": "B-Max AI Assistant with Embedded ProcessedTender DB",
        "embedded_data": {
            "total_tenders": tender_count,
            "unique_categories": len(categories),
            "unique_agencies": len(agencies),
            "last_updated": last_table_update.isoformat() if last_table_update else None
        },
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
        
        print(f"🎯 Using session with embedded ProcessedTender table - First name: {user_first_name}")
        
        # Enhance prompt with embedded table context
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
        
        print(f"✅ Response sent using embedded ProcessedTender database context")
        
        return {
            "response": response_text,
            "user_id": request.user_id,
            "username": user_first_name,
            "full_name": session.get_display_name(),
            "timestamp": datetime.now().isoformat(),
            "session_active": True,
            "embedded_data_used": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/refresh-embedded-data")
async def refresh_embedded_data():
    """Force refresh of the embedded ProcessedTender table"""
    try:
        tenders = embed_tender_table()
        return {
            "message": "Embedded ProcessedTender data refreshed successfully",
            "tenders_embedded": len(tenders) if tenders else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing embedded data: {str(e)}")

# Initialize embedded table on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing embedded ProcessedTender table...")
    embed_tender_table()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("🚀 Starting B-Max AI Assistant with Embedded ProcessedTender Database...")
    print("💬 Endpoint: POST /chat")
    print("🔧 Health: GET /health")
    print("🔄 Refresh: POST /refresh-embedded-data")
    print("📊 Database:", "Connected" if dynamodb else "Disconnected")
    print("🤖 Ollama:", "Connected" if ollama_available else "Disconnected")
    print(f"🌐 Server running on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
