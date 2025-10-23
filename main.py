import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)
chat_context = [
    {"role": "system", "content": "Hi, I'm B-Max! How can I assist?"}
]
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "AI Chatbot API is running!"}


class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: ChatRequest):
    # Add user's message to context
    chat_context.append({"role": "user", "content": request.prompt})

    response_text = ""
    # Send all messages so far to the AI
    for part in client.chat('deepseek-v3.1:671b-cloud', messages=chat_context, stream=True):
        response_text += part['message']['content']

    # Add AI's response to the context
    chat_context.append({"role": "assistant", "content": response_text})

    return {"response": response_text}


