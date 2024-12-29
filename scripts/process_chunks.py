import os
import time
import json  # For JSON serialization
from multiprocessing import Process
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database Monitoring Function
def monitor_database():
    postgres_uri = os.getenv("POSTGRESQL_URI")
    if not postgres_uri:
        logger.error("PostgreSQL URI is not set in .env")
        return

    try:
        conn = psycopg2.connect(postgres_uri)
        cur = conn.cursor()

        while True:
            logger.info("Monitoring database...")
            try:
                # Query for active connections
                cur.execute("SELECT COUNT(*) AS active_connections FROM pg_stat_activity")
                active_connections = cur.fetchone()[0]

                # Query for locks
                cur.execute("SELECT COUNT(*) AS locks FROM pg_locks")
                locks = cur.fetchone()[0]

                # Query for idle transactions
                cur.execute("""
                    SELECT COUNT(*) AS idle_transactions 
                    FROM pg_stat_activity 
                    WHERE state = 'idle in transaction'
                """)
                idle_transactions = cur.fetchone()[0]

                logger.info(f"Active connections: {active_connections}")
                logger.info(f"Locks: {locks}")
                logger.info(f"Idle transactions: {idle_transactions}")
            except Exception as monitor_error:
                logger.error(f"Error during database monitoring query: {monitor_error}")
            
            # Sleep for a monitoring interval (e.g., 10 seconds)
            time.sleep(10)

    except Exception as e:
        logger.error(f"Error during database monitoring setup: {e}")
    finally:
        cur.close()
        conn.close()
        logger.info("Database monitoring stopped.")

# Chunk Processing Function
def process_chunks():
    postgres_uri = os.getenv("POSTGRESQL_URI")
    if not postgres_uri:
        logger.error("PostgreSQL URI is not set in .env")
        return

    try:
        conn = psycopg2.connect(postgres_uri)
        cur = conn.cursor()

        # Sample chunks to insert (replace with actual chunk processing logic)
        chunks = [
            {"chunk_text": "Sample chunk 1", "metadata": {"key": "value1"}},
            {"chunk_text": "Sample chunk 2", "metadata": {"key": "value2"}}
        ]

        start_time = time.time()
        # Serialize metadata to JSON
        chunk_data = [(chunk['chunk_text'], json.dumps(chunk['metadata'])) for chunk in chunks]

        # Insert chunks
        execute_values(
            cur,
            """
            INSERT INTO text_chunks (chunk, metadata)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            chunk_data
        )
        conn.commit()

        elapsed_time = time.time() - start_time
        logger.info(f"Inserted {len(chunk_data)} chunks in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error during chunk insertion: {e}")
    finally:
        cur.close()
        conn.close()
        logger.info("Chunk processing completed.")

# Main Function to Run Both Processes
if __name__ == "__main__":
    try:
        # Create a separate process for monitoring the database
        monitor_process = Process(target=monitor_database)
        monitor_process.start()

        # Run the chunk processing in the main process
        process_chunks()

        # Wait for the monitoring process to finish if necessary (or terminate it)
        monitor_process.terminate()
        monitor_process.join()

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down processes.")
        monitor_process.terminate()
        monitor_process.join()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
