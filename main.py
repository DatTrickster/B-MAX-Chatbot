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

def get_user_profile_by_user_id(user_id: str):
    """Find user profile by userId (UUID)"""
    try:
        resp = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="userId = :uid",
            ExpressionAttributeValues={":uid": {"S": user_id}}
        )
        items = resp.get("Items", [])
        return dd_to_py(items[0]) if items else None
    except Exception as e:
        print(f"❌ Error scanning for user profile: {e}")
        return None

def get_user_profile_by_email(email: str):
    """Find user profile by email"""
    try:
        resp = dynamodb.scan(
            TableName=DYNAMODB_TABLE_USERS,
            FilterExpression="email = :email",
            ExpressionAttributeValues={":email": {"S": email}}
        )
        items = resp.get("Items", [])
        return dd_to_py(items[0]) if items else None
    except Exception as e:
        print(f"❌ Error scanning for user by email: {e}")
        return None

def get_cognito_user_by_username(username: str):
    """Get user details from Cognito by username"""
    try:
        if not cognito or not COGNITO_USER_POOL_ID:
            print("❌ Cognito not configured")
            return None
            
        response = cognito.admin_get_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=username
        )
        
        user_attributes = {}
        for attr in response.get('UserAttributes', []):
            user_attributes[attr['Name']] = attr['Value']
        
        # The UserSub is the UUID we need - this is the main identifier
        user_sub = response.get('UserSub')
        
        cognito_user = {
            'username': response.get('Username'),
            'user_id': user_sub,  # This is the UUID that matches userId in UserProfiles
            'email': user_attributes.get('email'),
            'email_verified': user_attributes.get('email_verified', 'false') == 'true',
            'status': response.get('UserStatus'),
            'enabled': response.get('Enabled', False),
            'created': response.get('UserCreateDate'),
            'modified': response.get('UserLastModifiedDate'),
            'attributes': user_attributes
        }
        
        print(f"✅ Found Cognito user: {username} -> UUID: {user_sub}")
        return cognito_user
        
    except Exception as e:
        print(f"❌ Error fetching Cognito user {username}: {e}")
        return None

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

def format_embedded_table_for_ai(tenders, user_preferences=None):
    """Format the entire embedded ProcessedTender table for AI consumption"""
    if not tenders:
        return "EMBEDDED PROCESSEDTENDER TABLE: No data available"
    
    # Create a comprehensive summary of the table
    table_summary = "COMPLETE TENDER DATABASE CONTEXT:\n\n"
    
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
    
    table_summary += f"📊 DATABASE OVERVIEW:\n"
    table_summary += f"• Total Tenders: {total_tenders}\n"
    table_summary += f"• Categories: {len(categories)}\n"
    table_summary += f"• Agencies: {len(agencies)}\n"
    table_summary += f"• Statuses: {len(statuses)}\n\n"
    
    # Add user preferences context if available
    if user_preferences:
        preferred_categories = user_preferences.get('preferredCategories', [])
        preferred_sites = user_preferences.get('preferredSites', [])
        
        if preferred_categories:
            table_summary += f"🎯 USER PREFERENCES:\n"
            table_summary += f"• Preferred Categories: {', '.join(preferred_categories)}\n"
            if preferred_sites:
                table_summary += f"• Preferred Sites: {len(preferred_sites)} sources\n"
            table_summary += "\n"
    
    # Top categories
    table_summary += "🏷️ TOP CATEGORIES:\n"
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:6]:
        table_summary += f"• {category}: {count} tenders\n"
    
    table_summary += "\n"
    
    # Sample tenders with key information
    table_summary += "📋 SAMPLE TENDERS:\n"
    for i, tender in enumerate(tenders[:8], 1):
        title = tender.get('title', 'No title')[:80] + '...' if len(tender.get('title', '')) > 80 else tender.get('title', 'No title')
        category = tender.get('Category', 'Unknown')
        agency = tender.get('sourceAgency', 'Unknown')
        closing_date = tender.get('closingDate', 'Unknown')
        status = tender.get('status', 'Unknown')
        reference_number = tender.get('referenceNumber', 'N/A')
        
        table_summary += f"{i}. {title}\n"
        table_summary += f"   📊 {category} | 🏢 {agency}\n"
        table_summary += f"   📅 {closing_date} | 📈 {status}\n"
        table_summary += f"   🏷️ {reference_number}\n\n"
    
    table_summary += "💡 You have access to all tender data including titles, references, categories, agencies, closing dates, contacts, and document links."
    
    return table_summary

def get_personalized_recommendations(user_prompt: str, tenders: list, user_preferences: dict):
    """Get personalized recommendations based on user preferences"""
    if not tenders:
        return []
    
    user_prompt_lower = user_prompt.lower()
    preferred_categories = user_preferences.get('preferredCategories', [])
    preferred_sites = user_preferences.get('preferredSites', [])
    
    scored_tenders = []
    
    for tender in tenders:
        score = 0
        match_reasons = []
        
        # Category preference matching (highest weight)
        tender_category = tender.get('Category', '')
        if tender_category in preferred_categories:
            score += 10
            match_reasons.append(f"Matches your preferred category: {tender_category}")
        
        # Site preference matching
        tender_source = tender.get('sourceUrl', '')
        for site in preferred_sites:
            if site in tender_source:
                score += 5
                match_reasons.append("From your preferred source")
                break
        
        # Query relevance
        title = tender.get('title', '').lower()
        if any(word in title for word in user_prompt_lower.split()):
            score += 3
            match_reasons.append("Matches your search query")
        
        # Urgency scoring
        closing_date = tender.get('closingDate', '')
        if closing_date:
            try:
                closing_dt = datetime.fromisoformat(closing_date.replace('Z', '+00:00'))
                days_until_close = (closing_dt - datetime.now()).days
                if 0 <= days_until_close <= 7:
                    score += 4
                    match_reasons.append("Closing soon")
            except:
                pass
        
        if score > 0:
            scored_tenders.append({
                'tender': tender,
                'score': score,
                'match_reasons': match_reasons
            })
    
    # Sort by score and return top recommendations
    scored_tenders.sort(key=lambda x: x['score'], reverse=True)
    return scored_tenders[:6]

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = None
        self.cognito_user = None
        self.chat_context = []
        self.last_active = datetime.now()
        self.total_messages = 0
        self.session_id = f"{user_id}_{int(time.time())}"
        
        print(f"🎯 Creating NEW session for user_id: {user_id}")
        self.load_user_profile()
        
        first_name = self.get_first_name()
        self.initialize_chat_context(first_name)
        
        print(f"✅ Session created - Name: {first_name}, Profile loaded: {self.user_profile is not None}")

    def initialize_chat_context(self, first_name: str):
        """Initialize or reinitialize chat context with system prompt"""
        # Get the embedded table for system context
        tenders = get_embedded_table()
        user_preferences = self.get_user_preferences()
        
        table_context = format_embedded_table_for_ai(tenders, user_preferences) if tenders else "Tender database not currently available."
        
        system_prompt = f"""You are B-Max, an AI assistant for TenderConnect. You have complete access to the tender database.

CRITICAL RULES - FOLLOW THESE EXACTLY:
1. ALWAYS address the user by their first name "{first_name}" in EVERY response
2. NEVER introduce yourself or mention that you have database access
3. Provide natural, conversational responses
4. Use the embedded database to give accurate, specific information
5. Focus on the user's preferences and needs
6. Format responses clearly with proper spacing and emojis for readability
7. Be warm, professional, and helpful

USER PROFILE:
- First Name: {first_name}
- Preferred Categories: {', '.join(user_preferences.get('preferredCategories', [])) if user_preferences.get('preferredCategories') else 'Not specified'}
- Company: {self.user_profile.get('companyName', 'Not specified') if self.user_profile else 'Not specified'}

DATABASE CONTEXT:
{table_context}

RESPONSE GUIDELINES:
- Keep responses natural and conversational
- Use proper formatting with line breaks for readability
- Focus on providing valuable information without self-references
- Personalize recommendations based on user preferences
- Use emojis sparingly to enhance readability
- Never mention your capabilities or database access"""

        # Set the chat context with system message only if empty
        if not self.chat_context:
            self.chat_context = [{"role": "system", "content": system_prompt}]
        else:
            # Update system message if context exists but system prompt might be outdated
            self.chat_context[0] = {"role": "system", "content": system_prompt}

    def load_user_profile(self):
        """Load user profile using Cognito to get UUID, then UserProfiles table"""
        try:
            if not dynamodb:
                self.user_profile = self.create_default_profile()
                return

            print(f"🔍 Loading profile for: {self.user_id}")
            
            # Strategy 1: Check if user_id is already a UUID (starts with Cognito pattern)
            if self.user_id.startswith(('us-east-', 'us-west-', 'af-south-')) or len(self.user_id) > 20:
                # This might already be a UUID, try direct lookup
                profile = get_user_profile_by_user_id(self.user_id)
                if profile:
                    self.user_profile = profile
                    print(f"✅ Profile found via direct UUID: {self.user_id}")
                    print(f"   firstName: {profile.get('firstName', 'NOT FOUND')}")
                    return
            
            # Strategy 2: Query Cognito to get UUID from username
            print(f"🔍 Querying Cognito for username: {self.user_id}")
            self.cognito_user = get_cognito_user_by_username(self.user_id)
            
            if self.cognito_user and self.cognito_user['user_id']:
                cognito_uuid = self.cognito_user['user_id']
                print(f"✅ Found Cognito UUID: {cognito_uuid} for username: {self.user_id}")
                
                # Now lookup in UserProfiles using the Cognito UUID
                profile = get_user_profile_by_user_id(cognito_uuid)
                if profile:
                    self.user_profile = profile
                    print(f"✅ Profile found via Cognito UUID: {cognito_uuid}")
                    print(f"   firstName: {profile.get('firstName', 'NOT FOUND')}")
                    return
                else:
                    print(f"❌ No UserProfiles entry found for Cognito UUID: {cognito_uuid}")
            
            # Strategy 3: Try direct email lookup in UserProfiles
            if '@' in self.user_id:
                profile = get_user_profile_by_email(self.user_id)
                if profile:
                    self.user_profile = profile
                    print(f"✅ Profile found via email: {self.user_id}")
                    return
            
            # Strategy 4: If we have Cognito user but no profile, try email from Cognito
            if self.cognito_user and self.cognito_user.get('email'):
                cognito_email = self.cognito_user['email']
                profile = get_user_profile_by_email(cognito_email)
                if profile:
                    self.user_profile = profile
                    print(f"✅ Profile found via Cognito email: {cognito_email}")
                    return
            
            # If no profile found, use default
            print(f"❌ No profile found for: {self.user_id}")
            self.user_profile = self.create_default_profile()
                
        except Exception as e:
            print(f"❌ Error loading user profile: {e}")
            self.user_profile = self.create_default_profile()

    def create_default_profile(self):
        """Create a default profile when user not found"""
        default_profile = {
            'firstName': 'User',
            'lastName': '',
            'companyName': 'Unknown',
            'position': 'User',
            'location': 'Unknown',
            'preferredCategories': []
        }
        print("⚠️ Using default profile")
        return default_profile

    def get_user_preferences(self):
        """Get user preferences from profile"""
        if not self.user_profile:
            return {}
        
        return {
            'preferredCategories': self.user_profile.get('preferredCategories', []),
            'preferredSites': self.user_profile.get('preferredSites', []),
            'companyName': self.user_profile.get('companyName', ''),
            'position': self.user_profile.get('position', '')
        }

    def get_display_name(self):
        if self.user_profile:
            first_name = self.user_profile.get('firstName', 'User')
            last_name = self.user_profile.get('lastName', '')
            return f"{first_name} {last_name}".strip()
        return "User"

    def get_first_name(self):
        if self.user_profile:
            first_name = self.user_profile.get('firstName', 'User')
            return first_name
        return "User"

    def update_activity(self):
        self.last_active = datetime.now()
        print(f"🕒 Session activity updated for {self.user_id}")

    def add_message(self, role, content):
        """Add message to chat context with proper management"""
        # Ensure system message is always first
        if not self.chat_context or self.chat_context[0]["role"] != "system":
            self.initialize_chat_context(self.get_first_name())
        
        self.chat_context.append({"role": role, "content": content})
        self.total_messages += 1
        
        # Keep reasonable context length but preserve system message
        if len(self.chat_context) > 20:  # Increased context window
            system_message = self.chat_context[0]
            recent_messages = self.chat_context[-19:]  # Keep 19 most recent + system
            self.chat_context = [system_message] + recent_messages
        
        print(f"💬 Added {role} message. Total messages: {self.total_messages}, Context length: {len(self.chat_context)}")

    def get_chat_context(self):
        """Get the current chat context, ensuring system message is present"""
        if not self.chat_context or self.chat_context[0]["role"] != "system":
            self.initialize_chat_context(self.get_first_name())
        return self.chat_context

def get_user_session(user_id: str) -> UserSession:
    """Get or create user session with improved session management"""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
        print(f"🆕 Created new session for {user_id}. Total active sessions: {len(user_sessions)}")
    else:
        print(f"🔄 Reusing existing session for {user_id}")
    
    session = user_sessions[user_id]
    session.update_activity()
    return session

def cleanup_old_sessions():
    """Clean up sessions that haven't been active for 2 hours"""
    current_time = datetime.now()
    expired_users = []
    
    for user_id, session in user_sessions.items():
        inactivity_time = (current_time - session.last_active).total_seconds()
        if inactivity_time > 7200:  # 2 hours
            expired_users.append(user_id)
            print(f"🧹 Cleaning up expired session for {user_id} (inactive for {inactivity_time:.0f}s)")
    
    for user_id in expired_users:
        del user_sessions[user_id]
    
    if expired_users:
        print(f"🧹 Cleaned up {len(expired_users)} expired sessions. Remaining: {len(user_sessions)}")

def enhance_prompt_with_context(user_prompt: str, session: UserSession) -> str:
    """Enhance prompt with embedded table context and personalization"""
    
    # Get the embedded table
    tenders = get_embedded_table()
    user_preferences = session.get_user_preferences()
    
    if not tenders:
        database_context = "No tender data available."
        personalized_context = ""
    else:
        # Get personalized recommendations
        personalized_recommendations = get_personalized_recommendations(user_prompt, tenders, user_preferences)
        
        if personalized_recommendations:
            personalized_context = "🎯 PERSONALIZED RECOMMENDATIONS:\n\n"
            for i, rec in enumerate(personalized_recommendations, 1):
                tender = rec['tender']
                reasons = rec['match_reasons']
                
                title = tender.get('title', 'No title')
                category = tender.get('Category', 'Unknown')
                agency = tender.get('sourceAgency', 'Unknown')
                closing_date = tender.get('closingDate', 'Unknown')
                reference_number = tender.get('referenceNumber', 'N/A')
                
                personalized_context += f"{i}. **{title}**\n"
                personalized_context += f"   📊 {category} | 🏢 {agency}\n"
                personalized_context += f"   📅 {closing_date} | 🏷️ {reference_number}\n"
                if reasons:
                    personalized_context += f"   ✅ {', '.join(reasons)}\n"
                personalized_context += "\n"
        else:
            personalized_context = ""
        
        database_context = format_embedded_table_for_ai(tenders, user_preferences)
    
    user_first_name = session.get_first_name()
    
    enhanced_prompt = f"""
User: {user_first_name}
Message: {user_prompt}

DATABASE CONTEXT:
{database_context}

{personalized_context}

INSTRUCTIONS:
- Respond naturally to {user_first_name}
- Use the database context to provide accurate information
- Personalize responses based on user preferences
- Format responses clearly with proper spacing
- Never mention database access or your capabilities
- Focus on being helpful and conversational
"""
    return enhanced_prompt

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    tenders = get_embedded_table()
    tender_count = len(tenders) if tenders else 0
    
    return {
        "message": "B-Max AI Assistant",
        "status": "healthy" if ollama_available else "degraded",
        "embedded_tenders": tender_count,
        "active_sessions": len(user_sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    tenders = get_embedded_table()
    tender_count = len(tenders) if tenders else 0
    
    # Perform session cleanup on health check
    cleanup_old_sessions()
    
    return {
        "status": "ok",
        "service": "B-Max AI Assistant",
        "embedded_tenders": tender_count,
        "active_sessions": len(user_sessions),
        "ollama_available": ollama_available,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not ollama_available:
            raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
        
        print(f"💬 Chat request received - user_id: {request.user_id}, prompt: {request.prompt}")
        
        session = get_user_session(request.user_id)
        user_first_name = session.get_first_name()
        
        print(f"🎯 Using session - First name: {user_first_name}, Total messages: {session.total_messages}")
        print(f"📊 Current context length: {len(session.chat_context)}")
        
        # Enhance prompt with embedded table context
        enhanced_prompt = enhance_prompt_with_context(request.prompt, session)
        
        # Add enhanced user message to context
        session.add_message("user", enhanced_prompt)
        
        # Get the current chat context for the AI
        chat_context = session.get_chat_context()
        
        # Get AI response
        try:
            response = client.chat(
                'deepseek-v3.1:671b-cloud', 
                messages=chat_context
            )
            response_text = response['message']['content']
        except Exception as e:
            print(f"❌ Ollama API error: {e}")
            response_text = f"I apologize {user_first_name}, but I'm having trouble processing your request right now. Please try again in a moment."
        
        # Add assistant response to context
        session.add_message("assistant", response_text)
        
        print(f"✅ Response sent to {user_first_name}. Total session messages: {session.total_messages}")
        
        return {
            "response": response_text,
            "user_id": request.user_id,
            "username": user_first_name,
            "full_name": session.get_display_name(),
            "timestamp": datetime.now().isoformat(),
            "session_active": True,
            "total_messages": session.total_messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/session-info/{user_id}")
async def get_session_info(user_id: str):
    """Debug endpoint to check session state"""
    if user_id in user_sessions:
        session = user_sessions[user_id]
        return {
            "user_id": user_id,
            "first_name": session.get_first_name(),
            "total_messages": session.total_messages,
            "context_length": len(session.chat_context),
            "last_active": session.last_active.isoformat(),
            "session_id": session.session_id
        }
    else:
        return {"error": "Session not found"}

# Initialize embedded table on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing embedded tender table...")
    embed_tender_table()
    print("✅ Startup complete")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("🚀 Starting B-Max AI Assistant...")
    print("💬 Endpoint: POST /chat")
    print("🔧 Health: GET /health")
    print("🐛 Session Debug: GET /session-info/{user_id}")
    print("📊 Database:", "Connected" if dynamodb else "Disconnected")
    print("🤖 Ollama:", "Connected" if ollama_available else "Disconnected")
    print(f"🌐 Server running on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
