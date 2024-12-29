import psycopg2
import tiktoken
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from ..src.db.operations import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def count_tokens_in_chunks(postgres_uri, batch_size=1000):
    """
    Count tokens in all chunks from the database.
    Returns total tokens and detailed statistics.
    """
    # Initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoder
    
    conn = get_db_connection(postgres_uri)
    cur = conn.cursor()
    
    try:
        # Get total count first
        cur.execute("SELECT COUNT(*) FROM text_chunks")
        total_chunks = cur.fetchone()[0]
        logging.info(f"Total chunks to process: {total_chunks}")
        
        total_tokens = 0
        tokens_per_chunk = []
        processed = 0
        
        # Process in batches
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            while processed < total_chunks:
                cur.execute("""
                    SELECT chunk FROM text_chunks 
                    OFFSET %s LIMIT %s
                """, (processed, batch_size))
                
                chunks = cur.fetchall()
                if not chunks:
                    break
                    
                for chunk in chunks:
                    num_tokens = len(enc.encode(chunk[0]))
                    total_tokens += num_tokens
                    tokens_per_chunk.append(num_tokens)
                
                processed += len(chunks)
                pbar.update(len(chunks))
        
        # Calculate statistics
        avg_tokens = sum(tokens_per_chunk) / len(tokens_per_chunk)
        max_tokens = max(tokens_per_chunk)
        min_tokens = min(tokens_per_chunk)
        
        stats = {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "average_tokens_per_chunk": avg_tokens,
            "max_tokens_in_chunk": max_tokens,
            "min_tokens_in_chunk": min_tokens
        }
        
        return stats
        
    finally:
        cur.close()
        conn.close()

def main():
    load_dotenv()
    postgres_uri = os.getenv("POSTGRESQL_URI")
    
    if not postgres_uri:
        logging.error("PostgreSQL URI not found in environment variables")
        return
    
    try:
        stats = count_tokens_in_chunks(postgres_uri)
        
        # Print results
        print("\nToken Count Analysis Results:")
        print("-" * 40)
        print(f"Total Chunks: {stats['total_chunks']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Average Tokens per Chunk: {stats['average_tokens_per_chunk']:.2f}")
        print(f"Maximum Tokens in a Chunk: {stats['max_tokens_in_chunk']}")
        print(f"Minimum Tokens in a Chunk: {stats['min_tokens_in_chunk']}")
        
        # Print warning if approaching limit
        if stats['total_tokens'] > 800_000:  # 80% of 1M limit
            print("\n⚠️ WARNING: You are approaching the 1M token limit!")
            print(f"Currently using: {(stats['total_tokens']/1_000_000):.1%} of available tokens")
            
    except Exception as e:
        logging.error(f"Error during token counting: {e}")

if __name__ == "__main__":
    main()