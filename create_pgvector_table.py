import os
import psycopg2
from dotenv import load_dotenv

def create_pgvector_table():
    """
    Creates a table for storing text chunks and embeddings.
    """
    # Load environment variables
    load_dotenv()
    postgres_uri = os.getenv("POSTGRESQL_URI")

    try:
        # Connect to the database
        conn = psycopg2.connect(postgres_uri)
        cur = conn.cursor()

        # Create the table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                id BIGSERIAL PRIMARY KEY,
                chunk TEXT,
                metadata JSONB,
                embedding VECTOR(1536)
            );
        """)
        conn.commit()

        print("Table 'text_chunks' created successfully!")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating table: {e}")

if __name__ == "__main__":
    create_pgvector_table()
