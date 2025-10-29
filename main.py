import os
import uvicorn
import json
import boto3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "af-south-1")
DYNAMODB_TABLE_TENDERS = os.getenv("DYNAMODB_TABLE_DEST", "ProcessedTender")
DYNAMODB_TABLE_USERS = os.getenv("DYNAMODB_TABLE_USERS", "UserProfiles")
DYNAMODB_TABLE_BOOKMARKS = os.getenv("DYNAMODB_TABLE_BOOKMARKS", "UserBookmarks")

# Initialize AWS clients
try:
    dynamodb = boto3.client(
        'dynamodb',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=AWS_REGION
    )
    print("✅ AWS Clients initialized successfully")
except Exception as e:
    print(f"❌ AWS Client initialization error: {e}")
    dynamodb = None

# Ollama Client
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

app = FastAPI(title="B-Max AI Assistant", version="1.0.0")

# In-memory session storage (in production, use Redis or database)
user_sessions = {}

class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "guest"  # Default user ID

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_profile = None
        self.chat_context = []
        self.last_active = datetime.now()
        self.greeted = False
        self.bookmark_patterns = None
        
        # Load user profile first
        self.load_user_profile()
        
        # Set up system prompt with user's actual name
        username = self.get_display_name()
        self.chat_context = [
            {"role": "system", "content": f"""You are B-Max, an AI assistant for TenderConnect. 

IMPORTANT RULES:
1. Always be respectful, professional, and helpful
2. Remember context from previous messages in this conversation
3. You can access tender database information when relevant
4. Personalize responses using the user's name: {username}
5. If you don't know something, be honest and say so
6. Keep responses concise but informative
7. Use emojis occasionally to make conversations friendly
8. When users ask about their preferences or bookmarks, analyze their bookmark patterns
9. Always greet the user by their first name when starting a conversation

Your capabilities:
- Answer questions about tenders, procurement, and the TenderConnect platform
- Provide information about tender categories, deadlines, and processes
- Help users understand how to use the platform
- Access real tender data from the database when needed
- Analyze user bookmark patterns and provide personalized recommendations
- Access user profile information

User Profile:
- Name: {username}
- Company: {self.user_profile.get('companyName', 'Unknown') if self.user_profile else 'Unknown'}
- Position: {self.user_profile.get('position', 'Unknown') if self.user_profile else 'Unknown'}
- Location: {self.user_profile.get('location', 'Unknown') if self.user_profile else 'Unknown'}

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Start by greeting the user warmly using their first name and introducing yourself."""}
        ]

    def load_user_profile(self):
        """Load user profile from UserProfiles table"""
        try:
            if dynamodb:
                response = dynamodb.get_item(
                    TableName=DYNAMODB_TABLE_USERS,
                    Key={'userId': {'S': self.user_id}}
                )
                if 'Item' in response:
                    self.user_profile = dynamodb_to_python(response['Item'])
                    print(f"📋 Loaded profile for user: {self.user_id}")
                    print(f"   👤 Name: {self.get_display_name()}")
                    print(f"   🏢 Company: {self.user_profile.get('companyName', 'Unknown')}")
                    print(f"   📍 Location: {self.user_profile.get('location', 'Unknown')}")
                else:
                    print(f"❌ No profile found for user: {self.user_id}")
                    # Create a default profile for unknown users
                    self.user_profile = {
                        'firstName': 'User',
                        'lastName': '',
                        'companyName': 'Unknown',
                        'position': 'User',
                        'location': 'Unknown'
                    }
        except Exception as e:
            print(f"❌ Error loading user profile: {e}")
            # Create a default profile on error
            self.user_profile = {
                'firstName': 'User',
                'lastName': '',
                'companyName': 'Unknown',
                'position': 'User',
                'location': 'Unknown'
            }

    def get_display_name(self):
        """Get user's display name from profile"""
        if self.user_profile:
            first_name = self.user_profile.get('firstName', 'User')
            last_name = self.user_profile.get('lastName', '')
            if last_name:
                return f"{first_name} {last_name}"
            return first_name
        return "User"

    def get_first_name(self):
        """Get user's first name only"""
        if self.user_profile:
            return self.user_profile.get('firstName', 'User')
        return "User"

    def update_activity(self):
        self.last_active = datetime.now()

    def add_message(self, role, content):
        self.chat_context.append({"role": role, "content": content})
        # Keep only last 20 messages to manage context length
        if len(self.chat_context) > 20:
            self.chat_context = [self.chat_context[0]] + self.chat_context[-19:]

def get_user_session(user_id: str) -> UserSession:
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
        # Load bookmarks for new session
        load_user_bookmarks(user_sessions[user_id])
    user_sessions[user_id].update_activity()
    return user_sessions[user_id]

def load_user_bookmarks(session: UserSession):
    """Load user bookmark data"""
    try:
        if dynamodb:
            # Get user bookmarks
            bookmarks_response = dynamodb.query(
                TableName=DYNAMODB_TABLE_BOOKMARKS,
                KeyConditionExpression='userId = :uid',
                ExpressionAttributeValues={':uid': {'S': session.user_id}}
            )
            
            if 'Items' in bookmarks_response and bookmarks_response['Items']:
                bookmarks = [dynamodb_to_python(item) for item in bookmarks_response['Items']]
                session.bookmark_patterns = analyze_bookmark_patterns(bookmarks, session.user_id)
                print(f"🔖 Loaded {len(bookmarks)} bookmarks for user: {session.user_id}")
            else:
                print(f"📝 No bookmarks found for user: {session.user_id}")
                session.bookmark_patterns = "No bookmarks found"
                
    except Exception as e:
        print(f"❌ Error loading user bookmarks: {e}")
        session.bookmark_patterns = f"Error loading bookmarks: {str(e)}"

def analyze_bookmark_patterns(bookmarks, user_id):
    """Analyze user's bookmark patterns to understand preferences"""
    if not bookmarks:
        return "No bookmarks found for this user."
    
    try:
        # Get tender details for bookmarked tenders
        tender_categories = []
        tender_agencies = []
        
        for bookmark in bookmarks:
            tender_id = bookmark.get('tenderId')
            if tender_id:
                # Get tender details from ProcessedTender table
                tender_response = dynamodb.get_item(
                    TableName=DYNAMODB_TABLE_TENDERS,
                    Key={'tender_id': {'S': tender_id}}
                )
                if 'Item' in tender_response:
                    tender = dynamodb_to_python(tender_response['Item'])
                    category = tender.get('Category')
                    agency = tender.get('sourceAgency')
                    
                    if category:
                        tender_categories.append(category)
                    if agency:
                        tender_agencies.append(agency)
        
        # Analyze patterns
        from collections import Counter
        category_counter = Counter(tender_categories)
        agency_counter = Counter(tender_agencies)
        
        analysis = {
            "total_bookmarks": len(bookmarks),
            "top_categories": category_counter.most_common(3),
            "top_agencies": agency_counter.most_common(3),
            "preferred_categories": [cat for cat, count in category_counter.most_common(5)],
            "preferred_agencies": [agency for agency, count in agency_counter.most_common(5)]
        }
        
        return analysis
        
    except Exception as e:
        return f"Error analyzing bookmarks: {str(e)}"

def get_personalized_recommendations(user_id):
    """Get personalized tender recommendations based on bookmark patterns"""
    try:
        if not dynamodb:
            return "Database unavailable for recommendations"
        
        # Get user's bookmark patterns
        session = user_sessions.get(user_id)
        if not session or not session.bookmark_patterns:
            return "No bookmark data available for recommendations"
        
        patterns = session.bookmark_patterns
        if isinstance(patterns, str):  # Error message
            return patterns
        
        # Get tenders matching user's preferred categories
        preferred_categories = patterns.get('preferred_categories', [])
        if not preferred_categories:
            return "No preferred categories found for recommendations"
        
        # Scan for tenders in preferred categories
        all_tenders = []
        response = dynamodb.scan(
            TableName=DYNAMODB_TABLE_TENDERS,
            FilterExpression='attribute_exists(Category)',
            ProjectionExpression='tender_id, title, Category, closingDate, sourceAgency'
        )
        all_tenders.extend(response.get('Items', []))
        
        while 'LastEvaluatedKey' in response:
            response = dynamodb.scan(
                TableName=DYNAMODB_TABLE_TENDERS,
                FilterExpression='attribute_exists(Category)',
                ProjectionExpression='tender_id, title, Category, closingDate, sourceAgency',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            all_tenders.extend(response.get('Items', []))
        
        # Filter and rank tenders
        recommended_tenders = []
        for tender_item in all_tenders:
            tender = dynamodb_to_python(tender_item)
            category = tender.get('Category')
            
            # Score based on category preference and recency
            if category in preferred_categories:
                score = preferred_categories.index(category) + 1
                tender['recommendation_score'] = score
                recommended_tenders.append(tender)
        
        # Sort by recommendation score and get top 5
        recommended_tenders.sort(key=lambda x: x.get('recommendation_score', 0))
        top_recommendations = recommended_tenders[:5]
        
        return format_recommendations_for_ai(top_recommendations, patterns)
        
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def format_recommendations_for_ai(tenders, patterns):
    """Format recommendations for AI consumption"""
    if not tenders:
        return "No recommendations available at this time."
    
    formatted = "Personalized Recommendations based on your bookmarks:\n\n"
    formatted += f"Your top categories: {', '.join(patterns.get('preferred_categories', []))}\n"
    formatted += f"Your top agencies: {', '.join(patterns.get('preferred_agencies', []))}\n\n"
    
    for i, tender in enumerate(tenders, 1):
        title = tender.get('title', 'Unknown Title')
        category = tender.get('Category', 'Uncategorized')
        closing_date = tender.get('closingDate', 'Unknown')
        agency = tender.get('sourceAgency', 'Unknown Agency')
        
        formatted += f"{i}. {title}\n"
        formatted += f"   📊 Category: {category} (matches your interests)\n"
        formatted += f"   🏢 Agency: {agency}\n"
        formatted += f"   📅 Closes: {closing_date}\n\n"
    
    return formatted

def get_tender_information(query: str = None, user_id: str = None):
    """Get relevant tender information from database"""
    try:
        if not dynamodb:
            return "Database currently unavailable"
        
        # Check if user wants personalized recommendations
        if user_id and query and any(keyword in query.lower() for keyword in ['my', 'recommend', 'suggest', 'prefer', 'bookmark', 'interest']):
            return get_personalized_recommendations(user_id)
        
        # Build query based on user's question
        if query and any(keyword in query.lower() for keyword in ['recent', 'latest', 'new']):
            # Get latest tenders
            response = dynamodb.scan(
                TableName=DYNAMODB_TABLE_TENDERS,
                Limit=5,
                Select='SPECIFIC_ATTRIBUTES',
                ProjectionExpression='title, Category, closingDate, sourceAgency'
            )
        elif query and any(keyword in query.lower() for keyword in ['category', 'categories', 'type']):
            # Get tenders by category
            response = dynamodb.scan(
                TableName=DYNAMODB_TABLE_TENDERS,
                Select='SPECIFIC_ATTRIBUTES',
                ProjectionExpression='title, Category, closingDate'
            )
        else:
            # General tender count and stats
            response = dynamodb.scan(
                TableName=DYNAMODB_TABLE_TENDERS,
                Select='COUNT'
            )
            return f"Total tenders in database: {response.get('Count', 0)}"
        
        items = response.get('Items', [])
        if not items:
            return "No tenders found in the database."
        
        # Format the response
        tenders_info = []
        for item in items[:5]:  # Limit to 5 tenders
            tender = {}
            for key, value in item.items():
                if 'S' in value:
                    tender[key] = value['S']
                elif 'N' in value:
                    tender[key] = value['N']
            tenders_info.append(tender)
        
        return format_tenders_for_ai(tenders_info)
        
    except Exception as e:
        return f"Error accessing database: {str(e)}"

def format_tenders_for_ai(tenders):
    """Format tender data for AI consumption"""
    if not tenders:
        return "No tender data available."
    
    formatted = "Recent Tenders:\n"
    for i, tender in enumerate(tenders, 1):
        title = tender.get('title', 'Unknown Title')
        category = tender.get('Category', 'Uncategorized')
        closing_date = tender.get('closingDate', 'Unknown')
        agency = tender.get('sourceAgency', 'Unknown Agency')
        
        formatted += f"{i}. {title}\n"
        formatted += f"   Category: {category}\n"
        formatted += f"   Agency: {agency}\n"
        formatted += f"   Closes: {closing_date}\n\n"
    
    return formatted

def enhance_prompt_with_context(user_prompt: str, session: UserSession) -> str:
    """Enhance user prompt with context and database information"""
    
    # Check if we need to query database
    database_context = ""
    tender_keywords = ['tender', 'tenders', 'procurement', 'bid', 'bids', 'contract', 'rfp', 'rfi', 'category', 'categories', 'recommend', 'suggest', 'bookmark', 'interest']
    
    if any(keyword in user_prompt.lower() for keyword in tender_keywords):
        database_context = get_tender_information(user_prompt, session.user_id)
    
    user_first_name = session.get_first_name()
    
    enhanced_prompt = f"""
User: {user_first_name}
Message: {user_prompt}

Current Context:
- Conversation history available in chat context
- User is asking about: {user_prompt}
- User's company: {session.user_profile.get('companyName', 'Unknown') if session.user_profile else 'Unknown'}
- User's position: {session.user_profile.get('position', 'Unknown') if session.user_profile else 'Unknown'}

{database_context if database_context else "No additional database context needed for this query."}

Please respond naturally while:
1. Using the user's first name: {user_first_name}
2. Maintaining conversation context
3. Being helpful and professional
4. Using database information if provided above
5. Personalizing responses based on user's profile and preferences
"""
    return enhanced_prompt

def dynamodb_to_python(item):
    """Convert DynamoDB item to Python dict"""
    result = {}
    for key, value in item.items():
        if 'S' in value:
            result[key] = value['S']
        elif 'N' in value:
            result[key] = float(value['N']) if '.' in value['N'] else int(value['N'])
        elif 'BOOL' in value:
            result[key] = value['BOOL']
        elif 'NULL' in value:
            result[key] = None
        elif 'M' in value:
            result[key] = dynamodb_to_python(value['M'])
        elif 'L' in value:
            result[key] = [dynamodb_to_python({'item': v})['item'] for v in value['L']]
        elif 'SS' in value:
            result[key] = value['SS']
        elif 'NS' in value:
            result[key] = [float(n) if '.' in n else int(n) for n in value['NS']]
    return result

@app.get("/")
async def root():
    return {
        "message": "B-Max AI Assistant API is running!",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health",
            "session_info": "/session/{user_id}"
        }
    }

@app.get("/health")
async def health_check():
    db_status = "healthy" if dynamodb else "unavailable"
    try:
        # Test Ollama connection
        test_response = ""
        for part in client.chat('deepseek-v3.1:671b-cloud', messages=[{"role": "user", "content": "Say OK"}], stream=True):
            test_response += part['message']['content']
        ollama_status = "healthy"
    except Exception as e:
        ollama_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok",
        "assistant": "B-Max AI",
        "database": db_status,
        "ollama": ollama_status,
        "active_sessions": len(user_sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint with context memory and database access"""
    try:
        # Clean up old sessions periodically
        cleanup_old_sessions()
        
        # Get or create user session
        session = get_user_session(request.user_id)
        user_first_name = session.get_first_name()
        
        # Enhance prompt with context and database info
        enhanced_prompt = enhance_prompt_with_context(request.prompt, session)
        
        # Add user message to context
        session.add_message("user", enhanced_prompt)
        
        # Generate response
        response_text = ""
        print(f"💬 {user_first_name}: {request.prompt}")
        print("🤖 B-Max thinking...", end="", flush=True)
        
        # Stream response from AI
        for part in client.chat('deepseek-v3.1:671b-cloud', messages=session.chat_context, stream=True):
            response_text += part['message']['content']
            print(".", end="", flush=True)
        
        print()  # New line after thinking dots
        
        # Add AI response to context
        session.add_message("assistant", response_text)
        
        # Mark as greeted after first interaction
        if not session.greeted:
            session.greeted = True
        
        print(f"✅ B-Max: {response_text[:100]}...")
        
        return {
            "response": response_text,
            "user_id": request.user_id,
            "username": user_first_name,
            "timestamp": datetime.now().isoformat(),
            "session_active": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

def cleanup_old_sessions():
    """Remove sessions older than 2 hours"""
    current_time = datetime.now()
    expired_users = []
    for user_id, session in user_sessions.items():
        if (current_time - session.last_active).total_seconds() > 7200:  # 2 hours
            expired_users.append(user_id)
    for user_id in expired_users:
        del user_sessions[user_id]

@app.get("/session/{user_id}")
async def get_session_info(user_id: str):
    """Get information about a user's session"""
    if user_id not in user_sessions:
        return {"error": "Session not found"}
    
    session = user_sessions[user_id]
    return {
        "user_id": session.user_id,
        "username": session.get_display_name(),
        "first_name": session.get_first_name(),
        "company": session.user_profile.get('companyName', 'Unknown') if session.user_profile else 'Unknown',
        "position": session.user_profile.get('position', 'Unknown') if session.user_profile else 'Unknown',
        "message_count": len(session.chat_context) - 1,  # Exclude system message
        "last_active": session.last_active.isoformat(),
        "greeted": session.greeted,
        "bookmark_count": session.bookmark_patterns.get('total_bookmarks', 0) if isinstance(session.bookmark_patterns, dict) else 0
    }

@app.delete("/session/{user_id}")
async def clear_session(user_id: str):
    """Clear a user's session"""
    if user_id in user_sessions:
        del user_sessions[user_id]
        return {"message": "Session cleared successfully"}
    return {"message": "Session not found"}

@app.get("/tenders")
async def get_tenders_preview():
    """Get a preview of tenders for context"""
    try:
        tender_info = get_tender_information("recent")
        return {
            "tenders_preview": tender_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to get tender preview: {str(e)}"}

if __name__ == "__main__":
    print("🚀 Starting B-Max AI Assistant...")
    print("💬 Endpoint: POST /chat")
    print("🔧 Health: GET /health")
    print("📊 Tenders Preview: GET /tenders")
    print("👤 User Profiles: Integrated with UserProfiles table")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
