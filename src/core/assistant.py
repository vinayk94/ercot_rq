from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import logging
import psycopg2
from typing import List, Dict
import time
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAssistant:
    def __init__(self):
        load_dotenv()
        
        # Initialize API key managers
        jina_keys = [
            key.strip() 
            for key in os.getenv("JINA_API_KEYS", "").split(",")
            if key.strip()
        ]
        class JinaKeyManager:
            def __init__(self, api_keys: List[str]):
                self.api_keys = api_keys
                self.current_key = api_keys[0] if api_keys else None
            
            def get_current_key(self):
                return self.current_key
                
            def rotate_key(self):
                return self.current_key

        self.key_manager = JinaKeyManager(jina_keys)


        # Initialize embeddings
        from .embeddings import ERCOTEmbeddings
        self.embeddings = ERCOTEmbeddings(
            key_manager=self.key_manager,
            model_name="jina-embeddings-v3"
        )
        
        # Connect to database
        self.conn = psycopg2.connect(os.getenv("POSTGRESQL_URI"))
        
        # Setup LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=1024
        )

    def create_prompt(self):
        template = """You are an expert assistant helping users understand ERCOT registration and qualification processes. 
        Use the provided context to answer questions about ERCOT participant types (QSE, LSE, RE, etc.), forms, and procedures.
        
        Guidelines:
        1. If the context contains relevant information, provide a detailed answer with:
           - Specific form names and references
           - Clear steps or requirements
           - Citations to source documents
        2. If the context doesn't contain relevant information, say:
           "I don't find specific information about this in the ERCOT documentation provided. 
           You may want to check ERCOT's official website or contact ERCOT directly for this information."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        return template

    async def vector_search(self, query: str, k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            cur = self.conn.cursor()
            cur.execute("""
                SELECT 
                    tc.chunk as content,
                    tc.metadata as metadata,
                    (ce.embedding <=> %s::vector) as distance
                FROM chunk_embeddings ce
                JOIN text_chunks tc ON ce.chunk_id = tc.id
                ORDER BY distance ASC
                LIMIT %s;
            """, (embedding_str, k))
            
            chunks = []
            for row in cur:
                chunks.append({
                    'content': row[0],
                    'metadata': row[1],
                    'distance': float(row[2])
                })
            
            logger.info(f"Found {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    async def process_query(self, query: str) -> Dict:
        try:
            start_time = time.time()
            chunks = await self.vector_search(query)
            
            if not chunks:
                return {
                    'answer': "I couldn't find any relevant information in the documents.",
                    'chunks': [],
                    'processing_time': 0
                }

            # Format context for LLM including the URLs
            context = "\n\n".join(
                f"Source: {chunk['metadata'].get('title', 'Unknown')} ({chunk['metadata'].get('url', '')})\n{chunk['content']}"
                for chunk in chunks
            )
            
            response = await self.llm.ainvoke(
                f"""You are an expert assistant helping users understand ERCOT registration and qualification processes.
                When referring to documents or forms, include their URLs in your answer using [text](url) format.
                Use the URLs provided in the context.
                
                Context: {context}
                Question: {query}"""
            )
            
            # Use the original metadata from scraping
            formatted_chunks = []
            for chunk in chunks:
                metadata = chunk['metadata']
                formatted_chunks.append({
                    'content': chunk['content'],
                    'title': metadata.get('title', 'Unknown Document'),
                    'url': metadata.get('url', ''),
                    'section': metadata.get('section', 'General'),
                    'distance': chunk.get('distance', 1.0)
                })
            
            return {
                'answer': response.content,
                'chunks': formatted_chunks,
                'processing_time': round(time.time() - start_time, 2)
            }
                
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            raise

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass