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

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [{'role': 'user', 'content': request.prompt}]
    response_text = ""

    for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
        response_text += part['message']['content']

    return {"response": response_text}


