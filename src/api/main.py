from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from src.core.assistant import RAGAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("api")

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query")
async def process_query(request: QueryRequest):
    try:
        assistant = RAGAssistant()
        result = await assistant.process_query(request.query)
        
        # Process chunks to get clean, unique sources
        sources = []
        seen_content = set()
        
        for chunk in result.get('chunks', []):
            # Skip if distance too high
            if chunk.get('distance', 1) > 0.5:
                continue
                
            content_preview = chunk['content'][:100]
            if content_preview in seen_content:
                continue
            seen_content.add(content_preview)
            
            # Clean up source info
            source_path = chunk['metadata'].get('source', '')
            doc_name = os.path.basename(source_path)
            doc_name = os.path.splitext(doc_name)[0].replace('_', ' ').title()
            
            sources.append({
                "document": doc_name,
                "section": chunk['metadata'].get('section', 'General'),
                "relevance": round((1 - chunk.get('distance', 0)) * 100, 1),
                "context": chunk['content'].strip()
            })
        
        # Sort by relevance and limit to top 3
        sources = sorted(sources, key=lambda x: x['relevance'], reverse=True)[:3]
        
        return {
            "answer": result['answer'],
            "sources": sources,
            "processing_time": result.get('processing_time', 0)
        }

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "answer": "I apologize, but I encountered an error. Please try again in a moment.",
            "sources": [],
            "processing_time": 0
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)