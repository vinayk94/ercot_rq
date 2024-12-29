import os
import psycopg2
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_db_connection(postgres_uri):
    """Create database connection"""
    try:
        return psycopg2.connect(postgres_uri)
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

def setup_database():
    """
    Complete database setup:
    1. Enable pgvector extension
    2. Create necessary tables
    3. Create indexes
    """
    load_dotenv()
    postgres_uri = os.getenv("POSTGRESQL_URI")
    
    if not postgres_uri:
        raise ValueError("POSTGRESQL_URI not found in environment variables")
    
    conn = get_db_connection(postgres_uri)
    cur = conn.cursor()
    
    try:
        # 1. Enable pgvector extension
        logging.info("Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 2. Create tables
        logging.info("Creating tables...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                id BIGSERIAL PRIMARY KEY,
                chunk TEXT,
                metadata JSONB,
                embedding VECTOR(1024)
            );
            
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                id BIGSERIAL PRIMARY KEY,
                chunk_id BIGINT REFERENCES text_chunks(id),
                embedding vector(1024),
                metadata JSONB,
                model_version VARCHAR(50) DEFAULT 'jina-embeddings-v3',
                tokens_used INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 3. Create indexes
        logging.info("Creating indexes...")
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embedding_vector_idx 
                ON chunk_embeddings 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                
                CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id 
                ON chunk_embeddings(chunk_id);
                
                CREATE INDEX IF NOT EXISTS idx_embeddings_model 
                ON chunk_embeddings(model_version);
            """)
        except Exception as e:
            logging.warning(f"Could not create vector similarity index: {e}")
            logging.warning("You can create it later after inserting data.")
        
        # 4. Create monitoring view
        logging.info("Setting up monitoring views...")
        cur.execute("""
            CREATE OR REPLACE VIEW embedding_status AS
            WITH stats AS (
                SELECT 
                    COUNT(DISTINCT tc.id) as total_chunks,
                    COUNT(DISTINCT ce.chunk_id) as embedded_chunks,
                    COALESCE(SUM(ce.tokens_used), 0) as total_tokens_used
                FROM text_chunks tc
                LEFT JOIN chunk_embeddings ce ON tc.id = ce.chunk_id
            )
            SELECT 
                total_chunks,
                embedded_chunks,
                (total_chunks - embedded_chunks) as pending_chunks,
                total_tokens_used,
                CASE 
                    WHEN embedded_chunks > 0 THEN 
                        CAST((total_tokens_used::float / embedded_chunks) AS NUMERIC(10,2))
                    ELSE 0 
                END as avg_tokens_per_chunk,
                CAST((total_tokens_used::float / 1000000 * 100) AS NUMERIC(10,1)) 
                    as token_limit_used_percent
            FROM stats;
        """)
        
        conn.commit()
        logging.info("Database setup completed successfully!")
        
        # Show current status
        cur.execute("SELECT * FROM embedding_status")
        status = cur.fetchone()
        if status:
            logging.info(f"""
Current Status:
- Total chunks: {status[0]}
- Chunks with embeddings: {status[1]}
- Pending chunks: {status[2]}
- Total tokens used: {status[3]:,}
- Avg tokens per chunk: {status[4]:.2f}
- Token limit used: {status[5]:.1f}%
            """)
            
    except Exception as e:
        conn.rollback()
        logging.error(f"Error during database setup: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    try:
        setup_database()
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        raise