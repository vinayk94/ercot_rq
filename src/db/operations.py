import time
import psycopg2
from psycopg2.extras import Json
import logging

def get_db_connection(postgres_uri):
    try:
        return psycopg2.connect(postgres_uri)
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

def store_extracted_text_in_db(data, postgres_uri):
    conn = get_db_connection(postgres_uri)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS extracted_text (
            id BIGSERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_source ON extracted_text ((metadata->>'source'));
    """)
    conn.commit()

    new_count, skipped_count = 0, 0
    duplicate_files = set()

    for doc in data:
        cur.execute("""
            SELECT 1 FROM extracted_text
            WHERE metadata->>'source' = %s;
        """, (doc["metadata"]["source"],))
        exists = cur.fetchone()

        if not exists:
            cur.execute("""
                INSERT INTO extracted_text (content, metadata)
                VALUES (%s, %s);
            """, (doc["content"], Json(doc["metadata"])))
            conn.commit()
            new_count += 1
        else:
            skipped_count += 1
            source = doc["metadata"]["source"]
            if source not in duplicate_files:
                logging.debug(f"Found duplicate file: {source}")
                duplicate_files.add(source)

    cur.close()
    conn.close()
    return new_count, skipped_count

def store_chunks_in_db(chunks, postgres_uri):
    logging.info("Starting chunk storage...")
    conn = get_db_connection(postgres_uri)
    cur = conn.cursor()

    # Get initial count
    cur.execute("SELECT COUNT(*) FROM text_chunks")
    existing_count = cur.fetchone()[0]
    logging.info(f"Found {existing_count} existing chunks in database")

    # Process in batches
    new_count = 0
    skipped_count = 0
    batch_size = 500

    # Get all existing chunk/source combinations upfront
    cur.execute("SELECT chunk, metadata->>'source' FROM text_chunks")
    existing = set((row[0], row[1]) for row in cur.fetchall())
    logging.info(f"Loaded {len(existing)} existing chunk signatures")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_start = time.time()

        try:
            # Check against preloaded existing chunks
            to_insert = []
            for chunk in batch:
                if (chunk["chunk"], chunk["metadata"]["source"]) not in existing:
                    to_insert.append((chunk["chunk"], Json(chunk["metadata"])))
                    existing.add((chunk["chunk"], chunk["metadata"]["source"]))  # Update our set
                else:
                    skipped_count += 1

            if to_insert:
                cur.executemany("""
                    INSERT INTO text_chunks (chunk, metadata)
                    VALUES (%s, %s)
                """, to_insert)
                new_count += len(to_insert)
                conn.commit()

            batch_time = time.time() - batch_start
            logging.info(f"Batch {i//batch_size + 1}: {len(to_insert)} new, "
                        f"{len(batch) - len(to_insert)} duplicates "
                        f"(took {batch_time:.2f}s)")

        except Exception as e:
            logging.error(f"Error in batch {i//batch_size + 1}: {e}")
            conn.rollback()
            continue

    # Verify final count
    cur.execute("SELECT COUNT(*) FROM text_chunks")
    final_count = cur.fetchone()[0]

    logging.info(f"Final results:"
                f"\n  - Starting count: {existing_count}"
                f"\n  - Chunks processed: {len(chunks)}"
                f"\n  - New chunks added: {new_count}"
                f"\n  - Duplicates skipped: {skipped_count}"
                f"\n  - Ending count: {final_count}")

    cur.close()
    conn.close()
    return new_count, skipped_count

def cleanup_database(postgres_uri):
    logging.info("Starting database cleanup...")
    conn = get_db_connection(postgres_uri)
    cur = conn.cursor()

    try:
        # Get initial count
        cur.execute("SELECT COUNT(*) FROM text_chunks")
        initial_count = cur.fetchone()[0]
        logging.info(f"Initial chunk count: {initial_count}")

        # Create a temporary table with unique chunks
        cur.execute("""
            CREATE TEMP TABLE unique_chunks AS
            SELECT DISTINCT ON (chunk, metadata->>'source') 
                id, chunk, metadata
            FROM text_chunks;
        """)
        
        # Get count of unique chunks
        cur.execute("SELECT COUNT(*) FROM unique_chunks")
        unique_count = cur.fetchone()[0]
        logging.info(f"Unique chunks count: {unique_count}")

        # Clear main table and reinsert unique chunks
        cur.execute("TRUNCATE TABLE text_chunks")
        cur.execute("""
            INSERT INTO text_chunks (id, chunk, metadata)
            SELECT id, chunk, metadata FROM unique_chunks;
        """)
        
        # Get final count
        cur.execute("SELECT COUNT(*) FROM text_chunks")
        final_count = cur.fetchone()[0]
        
        conn.commit()
        logging.info(f"Cleanup complete:"
                    f"\n  - Initial count: {initial_count}"
                    f"\n  - Removed duplicates: {initial_count - unique_count}"
                    f"\n  - Final count: {final_count}")

    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()