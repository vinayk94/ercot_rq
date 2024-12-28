import os
import psycopg2
import requests
import logging
from dotenv import load_dotenv
from psycopg2.extras import Json
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_db_connection(postgres_uri):
    return psycopg2.connect(postgres_uri)

def get_batch_embeddings(texts, api_key, retry_count=3):
    """Get embeddings from Jina AI API with retries"""
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    data = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "dimensions": 1024,
        "embedding_type": "float",
        "input": texts
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            return {
                'embeddings': [item["embedding"] for item in result["data"]],
                'tokens_used': result["usage"]["total_tokens"]
            }
        except Exception as e:
            if attempt == retry_count - 1:  # Last attempt
                logging.error(f"Error getting embeddings after {retry_count} attempts: {e}")
                if hasattr(response, 'text'):
                    logging.error(f"API Response: {response.text}")
                raise
            logging.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff

def check_token_usage(cur):
    """Check current token usage and determine if we can continue"""
    cur.execute("SELECT * FROM embedding_status")
    status = cur.fetchone()
    
    total_chunks = status[0]
    embedded_chunks = status[1]
    tokens_used = status[3]
    token_limit_used = status[5]
    
    return {
        'total_chunks': total_chunks,
        'embedded_chunks': embedded_chunks,
        'remaining_chunks': total_chunks - embedded_chunks,
        'tokens_used': tokens_used,
        'token_limit_used': token_limit_used
    }

def process_chunks(conn, api_key, batch_size=500, token_limit_percent=95):
    cur = conn.cursor()
    
    try:
        while True:
            # Check current usage
            usage = check_token_usage(cur)
            
            if usage['token_limit_used'] >= token_limit_percent:
                logging.info(f"Stopping: Token usage ({usage['token_limit_used']}%) approaching limit ({token_limit_percent}%)")
                break
                
            # Get next batch of unprocessed chunks
            cur.execute("""
                SELECT tc.id, tc.chunk, tc.metadata 
                FROM text_chunks tc
                LEFT JOIN chunk_embeddings ce ON tc.id = ce.chunk_id
                WHERE ce.id IS NULL
                LIMIT %s
            """, (batch_size,))
            
            chunks = cur.fetchall()
            if not chunks:
                logging.info("No more chunks to process")
                break
            
            chunk_ids = [c[0] for c in chunks]
            texts = [c[1] for c in chunks]
            
            logging.info(f"Processing batch of {len(chunks)} chunks...")
            batch_start = time.time()
            
            # Get embeddings
            result = get_batch_embeddings(texts, api_key)
            embeddings = result['embeddings']
            tokens_used = result['tokens_used']
            
            # Store embeddings
            for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
                cur.execute("""
                    INSERT INTO chunk_embeddings 
                        (chunk_id, embedding, metadata, model_version, tokens_used)
                    VALUES 
                        (%s, %s, %s, %s, %s)
                """, (
                    chunk_id, 
                    embedding, 
                    Json({"source": chunks[i][2].get('source')}),
                    'jina-embeddings-v3',
                    tokens_used // len(chunks)
                ))
            
            conn.commit()
            
            batch_duration = time.time() - batch_start
            logging.info(f"Batch processed in {batch_duration:.2f} seconds")
            
            # Show updated status
            usage = check_token_usage(cur)
            logging.info(f"""
Batch Complete! Current Status:
- Chunks embedded: {usage['embedded_chunks']} / {usage['total_chunks']}
- Remaining chunks: {usage['remaining_chunks']}
- Tokens used: {usage['tokens_used']:,}
- Token limit used: {usage['token_limit_used']}%
            """)
            
            # Add a small delay between batches
            time.sleep(1)
            
    except Exception as e:
        conn.rollback()
        logging.error(f"Error processing chunks: {e}")
        raise
    finally:
        cur.close()

def main():
    load_dotenv()
    postgres_uri = os.getenv("POSTGRESQL_URI")
    jina_api_key = os.getenv("JINA_API_KEY")
    
    if not postgres_uri or not jina_api_key:
        logging.error("Missing required environment variables")
        return
    
    conn = get_db_connection(postgres_uri)
    try:
        process_chunks(conn, jina_api_key, batch_size=500, token_limit_percent=95)
    finally:
        conn.close()

if __name__ == "__main__":
    main()