# B-Max AI Assistant API Documentation

**Version:** 1.0.0  
**Base URL:** `http://your-domain:8000`

## Overview

B-Max is an AI-powered assistant for TenderConnect that helps users find and interact with government tender opportunities. The API provides intelligent search, personalized recommendations, and conversational assistance for tender-related queries.

---

## Authentication

Currently, the API uses user identification through `user_id` parameters. Authentication is handled through AWS Cognito integration (optional).

---

## Endpoints

### 1. Root Endpoint

Get service status and basic information.

**Endpoint:** `GET /`

**Response:**
```json
{
  "message": "B-Max AI Assistant",
  "status": "healthy",
  "embedded_tenders": 1250,
  "active_sessions": 15,
  "available_agencies": 45,
  "timestamp": "2025-10-31T10:30:00"
}
```

**Status Values:**
- `healthy` - All services operational
- `degraded` - Ollama AI service unavailable

---

### 2. Health Check

Monitor service health and database status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok",
  "service": "B-Max AI Assistant",
  "embedded_tenders": 1250,
  "active_sessions": 15,
  "available_agencies": 45,
  "ollama_available": true,
  "timestamp": "2025-10-31T10:30:00"
}
```

**Use Case:** System monitoring and load balancing decisions.

---

### 3. Get Available Agencies

Retrieve list of all government agencies with active tenders.

**Endpoint:** `GET /agencies`

**Response:**
```json
{
  "agencies": [
    "City of Cape Town",
    "City of Johannesburg",
    "Department of Health",
    "Eskom Holdings SOC Ltd"
  ],
  "count": 45,
  "timestamp": "2025-10-31T10:30:00"
}
```

**Use Case:** Populate dropdown filters, autocomplete fields, or agency selection UI.

---

### 4. Chat with B-Max

Submit queries and receive intelligent tender recommendations.

**Endpoint:** `POST /chat`

**Request Body:**
```json
{
  "prompt": "Show me IT tenders in Cape Town",
  "user_id": "john.doe@example.com"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | User's query or message |
| `user_id` | string | No | User identifier (email, username, or UUID). Defaults to "guest" |

**Response:**
```json
{
  "response": "I found 3 matching tenders for you:\n\n**IT Infrastructure Upgrade**\n• Reference: CT-2025-001...",
  "user_id": "john.doe@example.com",
  "username": "John",
  "full_name": "John Doe",
  "timestamp": "2025-10-31T10:30:00",
  "session_active": true,
  "total_messages": 5,
  "filtered": false
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | AI-generated response with tender details and markdown formatting |
| `user_id` | string | User identifier used for the request |
| `username` | string | User's first name |
| `full_name` | string | User's full name |
| `timestamp` | string | ISO 8601 timestamp |
| `session_active` | boolean | Whether user session is active |
| `total_messages` | integer | Total messages in current session |
| `filtered` | boolean | Whether content was filtered/blocked |

**Error Responses:**

| Status Code | Description |
|-------------|-------------|
| `503` | AI service temporarily unavailable |
| `500` | Internal server error |

---

### 5. Session Information

Retrieve details about a user's current session.

**Endpoint:** `GET /session-info/{user_id}`

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | string | User identifier |

**Response:**
```json
{
  "user_id": "john.doe@example.com",
  "first_name": "John",
  "total_messages": 12,
  "context_length": 15,
  "last_active": "2025-10-31T10:25:00",
  "session_id": "john.doe@example.com_1730371500"
}
```

**Not Found Response:**
```json
{
  "error": "Session not found"
}
```

---

## Features

### Content Filtering

The API automatically filters inappropriate content and off-topic queries:

- **Blocked Content:** Profanity, hate speech, violence, sexual content
- **Off-Topic Handling:** Redirects non-tender queries with helpful message
- **Response:** Returns `filtered: true` with explanation message

### Intelligent Search

B-Max uses advanced matching algorithms:

- **Fuzzy Matching:** Handles typos and variations in agency names
- **Keyword Scoring:** Prioritizes tenders matching multiple search terms
- **User Preferences:** Weights results based on user profile preferences
- **Document Links:** Extracts and prioritizes tenders with downloadable documents

### Session Management

- **Automatic Creation:** Sessions created on first interaction
- **Context Retention:** Maintains conversation history (up to 20 messages)
- **Auto-Cleanup:** Sessions expire after 2 hours of inactivity
- **User Profiles:** Integrates with DynamoDB and AWS Cognito for personalization

---

## Example Queries

### Search by Category
```json
{
  "prompt": "Show me construction tenders",
  "user_id": "user123"
}
```

### Search by Agency
```json
{
  "prompt": "What tenders does City of Cape Town have?",
  "user_id": "user123"
}
```

### Search by Location
```json
{
  "prompt": "Find healthcare tenders in Johannesburg",
  "user_id": "user123"
}
```

### Get Documents
```json
{
  "prompt": "Show me tenders with downloadable bid documents",
  "user_id": "user123"
}
```

### Closing Soon
```json
{
  "prompt": "Which tenders are closing this week?",
  "user_id": "user123"
}
```

---

## Response Format

Tender details are returned in markdown format:

```markdown
**IT Infrastructure Upgrade Project**
• **Reference**: `CT-2025-IT-001`
• **Category**: IT Services
• **Agency**: City of Cape Town
• **Closing Date**: 2025-11-15
• **Status**: Open

**Document Links**
**PRIMARY DOCUMENT**: [Download Tender Documents](https://example.com/docs.pdf)

• **Source Page**: [View Original Tender](https://example.com/tender)
────────────────────────────────────────
```

---

## Rate Limiting

- **Sessions:** Maximum 2-hour lifetime, auto-cleanup on inactivity
- **Context:** Limited to 20 messages per session to manage memory
- **Database Refresh:** Tender database refreshes every 30 minutes

---

## Error Handling

All endpoints return appropriate HTTP status codes:

| Code | Description |
|------|-------------|
| `200` | Success |
| `503` | Service unavailable (AI offline) |
| `500` | Internal server error |

Error response format:
```json
{
  "detail": "Error description"
}
```

---

## AWS Integration

### Required Environment Variables

```bash
AWS_REGION=af-south-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
DYNAMODB_TABLE_DEST=ProcessedTender
DYNAMODB_TABLE_USERS=UserProfiles
DYNAMODB_TABLE_BOOKMARKS=UserBookmarks
COGNITO_USER_POOL_ID=your_pool_id
OLLAMA_API_KEY=your_ollama_key
PORT=8000
```

### DynamoDB Tables

**ProcessedTender:**
- Stores all tender opportunities
- Scanned on startup and every 30 minutes
- Fields: title, referenceNumber, Category, sourceAgency, closingDate, link, etc.

**UserProfiles:**
- User preferences and company information
- Fields: userId, firstName, lastName, companyName, preferredCategories

---

## CORS Configuration

API is configured with permissive CORS for development:

```python
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

**Production Recommendation:** Restrict origins to your frontend domain.

---

## Support

For issues or questions about the B-Max AI Assistant API, contact your system administrator or refer to the TenderConnect documentation.
