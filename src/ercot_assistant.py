from langchain_community.vectorstores import PGVector  # Use community version
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict
from collections import deque
import time
import asyncio
from sqlalchemy import create_engine, text
from urllib.parse import urlparse
from embeddings import ERCOTEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def create_prompt():
    """Create an ERCOT-specific prompt template for RAG"""
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
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def get_connection_string(uri: str) -> str:
    """Convert URI to proper SQLAlchemy format"""
    parsed = urlparse(uri)
    # Replace 'postgres://' with 'postgresql://' if needed
    if parsed.scheme == 'postgres':
        return uri.replace('postgres://', 'postgresql://', 1)
    return uri

class JinaKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = deque(api_keys)
        self.current_key = self.api_keys[0]
        self.failed_keys = set()
    
    def get_current_key(self):
        return self.current_key
    
    def rotate_key(self):
        if len(self.api_keys) > 1:
            self.api_keys.rotate(-1)
            self.current_key = self.api_keys[0]
            logging.info("Rotated to next Jina API key")
        return self.current_key
    
    def mark_key_failed(self, key: str, reason: str):
        if key not in self.failed_keys:
            self.failed_keys.add(key)
            logging.warning(f"Jina API key marked as failed: {reason}")
        self.rotate_key()
    
    def has_available_keys(self):
        return len(self.failed_keys) < len(self.api_keys)

class RAGAssistant:
    def __init__(self):
        load_dotenv()
        
        # Initialize API key managers
        jina_keys = [
            key.strip() 
            for key in os.getenv("JINA_API_KEYS", "").split(",")
            if key.strip()
        ]
        self.key_manager = JinaKeyManager(jina_keys)
        
        # Setup embeddings with key rotation
        self.embeddings = ERCOTEmbeddings(
            key_manager=self.key_manager,
            model_name="jina-embeddings-v3"
        )
        
        # Get and format connection string
        raw_connection_string = os.getenv("POSTGRESQL_URI")
        if not raw_connection_string:
            raise ValueError("POSTGRESQL_URI not found in environment variables")
            
        CONNECTION_STRING = get_connection_string(raw_connection_string)
        logging.info(f"Using connection scheme: {urlparse(CONNECTION_STRING).scheme}")
        
        # Create engine with proper dialect
        self.engine = create_engine(
            CONNECTION_STRING,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
        
        # Test connection and inspect table
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.commit()
                
                # Let's inspect the table structure
                table_info = self.inspect_table()
                logging.info(f"Table structure: {table_info}")
                
            logging.info("Database connection test successful")
        except Exception as e:
            logging.error(f"Database connection test failed: {e}")
            raise
        
        # Setup vector store with proper connection
        self.vectorstore = PGVector.from_existing_index(
            embedding=self.embeddings,
            connection_string=CONNECTION_STRING,
            collection_name="chunk_embeddings",
            distance_strategy="cosine"
        )
        
        # Setup Groq LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=1024
        )
        
        # Setup retriever with specific parameters
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "fetch_k": 20,  # Fetch more candidates initially
                "score_threshold": 0.5
            }
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": create_prompt(),
            }
        )
        
        logging.info("RAG Assistant initialized successfully")

    def inspect_table(self) -> Dict:
        """Inspect the chunk_embeddings table structure"""
        with self.engine.connect() as conn:
            # Get table structure
            columns = conn.execute(text("""
                SELECT 
                    column_name, 
                    data_type,
                    is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'chunk_embeddings'
                ORDER BY ordinal_position;
            """)).fetchall()
            
            # Get index information
            indexes = conn.execute(text("""
                SELECT
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE tablename = 'chunk_embeddings';
            """)).fetchall()
            
            # Get sample row
            sample = conn.execute(text("""
                SELECT *
                FROM chunk_embeddings
                LIMIT 1;
            """)).fetchone()
            
            return {
                'columns': [(col[0], col[1], col[2]) for col in columns],
                'indexes': [(idx[0], idx[1]) for idx in indexes],
                'has_data': sample is not None
            }

    async def get_relevant_chunks(self, query: str) -> List[Dict]:
        """Get and debug relevant chunks for a query"""
        try:
            # Get query embedding directly
            query_embedding = self.embeddings.embed_query(query)
            
            # Use raw SQL to get similar chunks with scores
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        tc.chunk as content,
                        tc.metadata as metadata,
                        1 - (ce.embedding <=> %(embedding)s::vector) as similarity
                    FROM chunk_embeddings ce
                    JOIN text_chunks tc ON ce.chunk_id = tc.id
                    WHERE ce.embedding IS NOT NULL
                    ORDER BY ce.embedding <=> %(embedding)s::vector ASC
                    LIMIT 5;
                """), {"embedding": query_embedding})
                
                chunks = []
                for row in result:
                    chunks.append({
                        'content': row.content,
                        'metadata': row.metadata,
                        'similarity': float(row.similarity)
                    })
                
                return chunks
        except Exception as e:
            logging.error(f"Error getting relevant chunks: {e}")
            logging.error(f"Query embedding shape: {len(query_embedding)}")
            return []
        
    async def direct_vector_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search directly using SQL"""
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            with self.engine.connect() as conn:
                # First, let's verify we can access the embeddings
                verify = conn.execute(text("""
                    SELECT embedding[1:5] as sample
                    FROM chunk_embeddings 
                    LIMIT 1;
                """))
                logging.info(f"Sample embedding values: {verify.fetchone()}")
                
                # Now try the actual search
                result = conn.execute(text("""
                    SELECT 
                        tc.chunk as content,
                        tc.metadata as metadata,
                        1 - (ce.embedding <=> :embedding::vector) as similarity
                    FROM chunk_embeddings ce
                    JOIN text_chunks tc ON ce.chunk_id = tc.id
                    ORDER BY ce.embedding <=> :embedding::vector
                    LIMIT :k;
                """), {
                    "embedding": query_embedding,
                    "k": k
                })
                
                chunks = []
                for row in result:
                    chunks.append({
                        'content': row.content,
                        'metadata': row.metadata,
                        'similarity': float(row.similarity) if row.similarity else 0.0
                    })
                
                return chunks
                
        except Exception as e:
            logging.error(f"Error in direct vector search: {e}")
            logging.error(f"Query embedding shape: {len(query_embedding)}")
            return []
        
    async def basic_vector_search(self, query: str, k: int = 5) -> List[Dict]:
        """Most basic vector search possible"""
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            with self.engine.connect() as conn:
                # Convert embedding to string format PostgreSQL expects
                embedding_str = f"[{','.join(map(str, query_embedding))}]"
                
                # Use raw SQL with string formatting for vector
                result = conn.execute(text(f"""
                    SELECT 
                        tc.chunk as content,
                        tc.metadata as metadata,
                        (ce.embedding <=> '{embedding_str}'::vector) as distance
                    FROM chunk_embeddings ce
                    JOIN text_chunks tc ON ce.chunk_id = tc.id
                    ORDER BY distance ASC
                    LIMIT {k};
                """))
                
                chunks = []
                for row in result:
                    chunks.append({
                        'content': row.content,
                        'metadata': row.metadata,
                        'distance': float(row.distance) if row.distance else 1.0
                    })
                
                # Log what we found
                logging.info(f"Found {len(chunks)} chunks")
                if chunks:
                    logging.info(f"First chunk distance: {chunks[0]['distance']}")
                    logging.info(f"First chunk content preview: {chunks[0]['content'][:200]}")
                    logging.info(f"First chunk source: {chunks[0]['metadata'].get('source', 'Unknown')}")
                
                return chunks
                
        except Exception as e:
            logging.error(f"Error in vector search: {e}")
            logging.error(f"Query embedding shape: {len(query_embedding)}")
            return []  

    # Add test function
    async def test_vector_ops(self):
        """Test basic vector operations"""
        try:
            with self.engine.connect() as conn:
                # Test a basic vector operation
                logging.info("Testing vector operations...")
                
                # Try a simple comparison between two existing vectors
                result = conn.execute(text("""
                    SELECT e1.id, e2.id,
                        e1.embedding <=> e2.embedding as distance
                    FROM 
                        (SELECT id, embedding FROM chunk_embeddings LIMIT 1) e1,
                        (SELECT id, embedding FROM chunk_embeddings LIMIT 1 OFFSET 1) e2;
                """))
                
                row = result.fetchone()
                if row:
                    logging.info(f"Vector operation test successful")
                    logging.info(f"Distance between vectors {row[0]} and {row[1]}: {row[2]}")
                    return True
                else:
                    logging.error("No vectors found in test")
                    return False
                    
        except Exception as e:
            logging.error(f"Vector operation test failed: {e}")
            return False

    async def process_query(self, query: str) -> Dict:
        """Process a query using basic vector search"""
        try:
            start_time = time.time()
            
            # Test vector operations first
            test_result = await self.test_vector_ops()
            if not test_result:
                return {
                    'answer': "System error: Vector operations not working correctly",
                    'chunks': [],
                    'processing_time': 0
                }
            
            # Get relevant chunks
            relevant_chunks = await self.basic_vector_search(query)
            
            if not relevant_chunks:
                return {
                    'answer': "I couldn't find any relevant information in the documents.",
                    'chunks': [],
                    'processing_time': 0
                }
            
            # Format chunks for the LLM
            context = "\n\n".join([
                f"Document: {chunk['metadata'].get('source', 'Unknown')}\n"
                f"Content: {chunk['content']}\n"
                f"Distance: {chunk['distance']:.3f}"
                for chunk in relevant_chunks
            ])
            
            # Get LLM response with correct input format
            response = await self.qa_chain.ainvoke({
                "input_documents": relevant_chunks,
                "query": query
            })
            
            processing_time = round(time.time() - start_time, 2)
            
            return {
                'answer': response.get('result', response.get('output_text', '')),
                'chunks': relevant_chunks,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                'answer': "I apologize, but I encountered an error. Please try rephrasing your question.",
                'chunks': [],
                'processing_time': 0,
                'error': str(e)
            }
        
    async def check_collections(self) -> Dict:
        """Check available collections and their content"""
        try:
            with self.engine.connect() as conn:
                # Check tables
                tables = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                table_list = [row[0] for row in tables]
                
                # Get counts from chunk_embeddings
                count_result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM chunk_embeddings
                """))
                embedding_count = count_result.scalar()
                
                # Get sample embedding
                sample_result = conn.execute(text("""
                    SELECT chunk_id, metadata 
                    FROM chunk_embeddings 
                    LIMIT 1
                """))
                sample = sample_result.fetchone()
                
                return {
                    'tables': table_list,
                    'embedding_count': embedding_count,
                    'has_sample': sample is not None,
                    'sample_metadata': sample[1] if sample else None
                }
        except Exception as e:
            logging.error(f"Error checking collections: {e}")
            return {'error': str(e)}
    
    async def check_table_structure(self) -> Dict:
        """Check the structure of our tables"""
        try:
            with self.engine.connect() as conn:
                # Check chunk_embeddings structure
                result = conn.execute(text("""
                    SELECT 
                        column_name, 
                        data_type 
                    FROM information_schema.columns
                    WHERE table_name = 'chunk_embeddings'
                """))
                
                structure = {row[0]: row[1] for row in result}
                
                # Get a sample row
                sample = conn.execute(text("""
                    SELECT *
                    FROM chunk_embeddings ce
                    JOIN text_chunks tc ON ce.chunk_id = tc.id
                    LIMIT 1
                """))
                
                return {
                    'structure': structure,
                    'has_sample': sample.rowcount > 0
                }
                
        except Exception as e:
            logging.error(f"Error checking table structure: {e}")
            return {'error': str(e)}
        
    async def diagnose_embeddings(self) -> Dict:
        """Check where our embeddings actually are"""
        try:
            with self.engine.connect() as conn:
                # Check both tables
                result1 = conn.execute(text("""
                    SELECT COUNT(*), 
                        COUNT(DISTINCT chunk_id) as unique_chunks,
                        COUNT(NULLIF(embedding, NULL)) as non_null_embeddings
                    FROM chunk_embeddings;
                """))
                ce_stats = result1.fetchone()

                result2 = conn.execute(text("""
                    SELECT COUNT(*), 
                        COUNT(embedding) as non_null_embeddings
                    FROM langchain_pg_embedding;
                """))
                lpe_stats = result2.fetchone()

                # Get sample content
                result3 = conn.execute(text("""
                    SELECT ce.embedding IS NOT NULL as has_embedding,
                        tc.chunk as content,
                        tc.metadata->>'source' as source
                    FROM chunk_embeddings ce
                    JOIN text_chunks tc ON ce.chunk_id = tc.id
                    LIMIT 1;
                """))
                sample = result3.fetchone()

                return {
                    'chunk_embeddings_stats': {
                        'total_rows': ce_stats[0],
                        'unique_chunks': ce_stats[1],
                        'non_null_embeddings': ce_stats[2]
                    },
                    'langchain_embeddings_stats': {
                        'total_rows': lpe_stats[0],
                        'non_null_embeddings': lpe_stats[1]
                    },
                    'sample_chunk': {
                        'has_embedding': sample[0] if sample else None,
                        'content': sample[1][:200] if sample else None,
                        'source': sample[2] if sample else None
                    }
                }
        except Exception as e:
            logging.error(f"Error in diagnosis: {e}")
            return {'error': str(e)}

def main():
    """Example usage of the RAG Assistant"""
    assistant = RAGAssistant()

    # Run diagnostics first
    print("\nRunning Embedding Diagnostics...")
    diag_info = asyncio.run(assistant.diagnose_embeddings())
    print(f"Diagnostics: {diag_info}")

    # Check table structure first
    print("\nChecking Table Structure...")
    structure_info = asyncio.run(assistant.check_table_structure())
    print(f"Table structure: {structure_info}")
    
    # Check collections first
    print("\nChecking Vector Store Collections...")
    collection_info = asyncio.run(assistant.check_collections())
    print(f"Found tables: {collection_info['tables']}")
    print(f"Embedding count: {collection_info['embedding_count']}")
    if collection_info.get('sample_metadata'):
        print(f"Sample metadata: {collection_info['sample_metadata']}")
    
    test_queries = [
        "What forms do I need to submit for QSE registration?",
        "What is the process for changing a company's legal name in ERCOT?",
        "What are the steps to register as a Load Serving Entity?"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"Query: {query}")
        print("="*50)
        
        response = asyncio.run(assistant.process_query(query))
        
        print(f"\nAnswer: {response['answer']}")
        print("\nRelevant Chunks:")
        for i, chunk in enumerate(response.get('chunks', []), 1):
            print(f"\nChunk {i}:")
            print(f"Source: {os.path.basename(chunk['metadata'].get('source', 'Unknown'))}")
            print(f"Distance: {chunk['distance']:.3f}")
            print(f"Preview: {chunk['content'][:200]}...")
        
        print(f"\nProcessing Time: {response.get('processing_time', 0)} seconds")

if __name__ == "__main__":
    main()