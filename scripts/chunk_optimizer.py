import psycopg2
import tiktoken
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
from ..src.db.operations import get_db_connection 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_backup(conn):
    """Create a backup of the text_chunks table"""
    cur = conn.cursor()
    try:
        # Drop existing backup if exists
        cur.execute("DROP TABLE IF EXISTS text_chunks_backup")
        # Create backup table with timestamp
        cur.execute("""
            CREATE TABLE text_chunks_backup AS 
            SELECT * FROM text_chunks;
        """)
        conn.commit()
        logging.info("Created backup table: text_chunks_backup")
    finally:
        cur.close()

def is_mostly_nan(text):
    """Check if a chunk is mostly NaN values"""
    nan_count = text.lower().count('nan')
    words = text.split()
    return nan_count > len(words) * 0.5 if words else True

def analyze_small_chunks(conn, min_tokens=10):
    """Analyze and return IDs of small chunks without removing them"""
    cur = conn.cursor()
    enc = tiktoken.get_encoding("cl100k_base")
    
    try:
        cur.execute("SELECT id, chunk, metadata FROM text_chunks")
        chunks = cur.fetchall()
        
        small_chunk_ids = []
        examples = []  # Store examples
        
        for chunk_id, chunk_text, metadata in tqdm(chunks, desc="Analyzing chunk sizes"):
            token_count = len(enc.encode(chunk_text))
            if token_count < min_tokens:
                small_chunk_ids.append(chunk_id)
                if len(examples) < 5:
                    examples.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'tokens': token_count,
                        'source': metadata.get('source', 'Unknown'),
                        'page': metadata.get('page', 'N/A')
                    })
        
        return small_chunk_ids, examples
        
    finally:
        cur.close()

def analyze_similar_chunks(conn, similarity_threshold=0.95, batch_size=1000):
    """Analyze and return IDs of similar chunks without removing them"""
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT COUNT(*) FROM text_chunks")
        total_chunks = cur.fetchone()[0]
        
        similar_pairs = []
        processed = 0
        
        while processed < total_chunks:
            cur.execute("""
                SELECT id, chunk, metadata 
                FROM text_chunks 
                OFFSET %s LIMIT %s
            """, (processed, batch_size))
            
            chunks = cur.fetchall()
            if not chunks:
                break
                
            chunk_ids = [c[0] for c in chunks]
            texts = [c[1] for c in chunks]
            metadatas = [c[2] for c in chunks]
            
            # Skip chunks that are mostly NaN
            valid_indices = [i for i, text in enumerate(texts) 
                           if not is_mostly_nan(text)]
            
            if valid_indices:
                valid_texts = [texts[i] for i in valid_indices]
                
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(valid_texts)
                similarities = cosine_similarity(tfidf_matrix)
                
                for i in range(len(valid_indices)):
                    for j in range(i + 1, len(valid_indices)):
                        if similarities[i][j] > similarity_threshold:
                            idx1, idx2 = valid_indices[i], valid_indices[j]
                            # Only consider pairs from same source
                            if metadatas[idx1].get('source') == metadatas[idx2].get('source'):
                                similar_pairs.append({
                                    'id1': chunk_ids[idx1],
                                    'text1': texts[idx1],
                                    'metadata1': metadatas[idx1],
                                    'id2': chunk_ids[idx2],
                                    'text2': texts[idx2],
                                    'metadata2': metadatas[idx2],
                                    'similarity': similarities[i][j]
                                })
            
            processed += len(chunks)
            logging.info(f"Analyzed {processed}/{total_chunks} chunks")
        
        return similar_pairs
    finally:
        cur.close()

def remove_chunks(conn, chunk_ids):
    """Remove specified chunks from database"""
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM text_chunks WHERE id = ANY(%s) RETURNING id",
            (chunk_ids,)
        )
        removed = cur.fetchall()
        conn.commit()
        return len(removed)
    finally:
        cur.close()

def print_chunk_info(chunk_id, text, metadata, prefix=""):
    """Pretty print chunk information"""
    source = metadata.get('source', 'Unknown')
    page = metadata.get('page', 'N/A')
    print(f"{prefix}ID: {chunk_id}")
    print(f"{prefix}Source: {os.path.basename(source)}")
    print(f"{prefix}Page/Sheet: {page}")
    print(f"{prefix}Text: {text[:150]}..." if len(text) > 150 else f"{prefix}Text: {text}")
    print()

def main():
    load_dotenv()
    postgres_uri = os.getenv("POSTGRESQL_URI")
    
    if not postgres_uri:
        logging.error("PostgreSQL URI not found in environment variables")
        return
        
    conn = get_db_connection(postgres_uri)
    
    try:
        print("\nAnalyzing Database Chunks...")
        print("-" * 50)
        
        # Create backup first
        create_backup(conn)
        
        # Analyze small chunks
        print("\n1. Analyzing small chunks...")
        small_chunks, examples = analyze_small_chunks(conn, min_tokens=10)
        print(f"\nFound {len(small_chunks)} chunks with fewer than 10 tokens")
        if examples:
            print("\nExample small chunks:")
            for example in examples:
                print_chunk_info(
                    example['id'],
                    example['text'],
                    {'source': example['source'], 'page': example['page']},
                    prefix="  "
                )
        
        # Analyze similar chunks
        print("\n2. Analyzing similar chunks...")
        similar_pairs = analyze_similar_chunks(conn, similarity_threshold=0.95)
        print(f"\nFound {len(similar_pairs)} pairs of similar chunks")
        if similar_pairs:
            print("\nExample similar pairs (showing first 3):")
            for pair in similar_pairs[:3]:
                print("\nPair with similarity: {:.2f}".format(pair['similarity']))
                print("Chunk 1:")
                print_chunk_info(pair['id1'], pair['text1'], pair['metadata1'], prefix="  ")
                print("Chunk 2:")
                print_chunk_info(pair['id2'], pair['text2'], pair['metadata2'], prefix="  ")
        
        # Calculate potential token savings
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens_to_save = sum(len(enc.encode(pair['text1'])) 
                                 for pair in similar_pairs)
        
        # Summary and confirmation
        print("\nSummary of Proposed Changes:")
        print("-" * 50)
        print(f"Small chunks to remove: {len(small_chunks)}")
        print(f"Similar pairs found: {len(similar_pairs)}")
        print(f"Estimated tokens to be saved: {total_tokens_to_save:,}")
        print("\nA backup has been created as 'text_chunks_backup'")
        
        confirm = input("\nWould you like to proceed with removal? (yes/no): ")
        
        if confirm.lower() == 'yes':
            # Remove small chunks
            if small_chunks:
                removed = remove_chunks(conn, small_chunks)
                print(f"Removed {removed} small chunks")
            
            # Remove similar chunks (keeping one from each pair)
            if similar_pairs:
                to_remove = [pair['id1'] for pair in similar_pairs]
                removed = remove_chunks(conn, to_remove)
                print(f"Removed {removed} similar chunks")
                
            print("\nOptimization complete!")
            print("\nTo restore from backup if needed:")
            print("DELETE FROM text_chunks;")
            print("INSERT INTO text_chunks SELECT * FROM text_chunks_backup;")
        else:
            print("\nOperation cancelled. No changes made to the database.")
            print("The backup table 'text_chunks_backup' has been retained.")
            
    except Exception as e:
        logging.error(f"Error during optimization: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()