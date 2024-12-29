# src/api/models.py
from pydantic import BaseModel
from typing import List, Optional

class Source(BaseModel):
    file: str
    relevance_score: float
    context: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    processing_time: float

# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import RAGResponse
from src.core.assistant import RAGAssistant

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query", response_model=RAGResponse)
async def process_query(query: str):
    assistant = RAGAssistant()
    result = await assistant.process_query(query)
    
    # Format response like Perplexity
    sources = [
        Source(
            file=chunk['metadata'].get('source', 'Unknown'),
            relevance_score=1 - chunk.get('distance', 0),
            context=chunk['content'][:200]
        )
        for chunk in result.get('chunks', [])
    ]
    
    return RAGResponse(
        answer=result['answer'],
        sources=sources,
        processing_time=result.get('processing_time', 0)
    )