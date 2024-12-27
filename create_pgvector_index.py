import os
import psycopg2
from dotenv import load_dotenv

def create_pgvector_index():
    """
    Creates an index on the embedding column for similarity search.
    """
    # Load environment variables
    load_dotenv()
    postgres_uri = os.getenv("POSTGRESQL_URI")

    try:
        # Connect to the database
        conn = psycopg2.connect(postgres_uri)
        cur = conn.cursor()

        # Create the index
        cur.execute("""
            CREATE INDEX IF NOT EXISTS embedding_index
            ON text_chunks USING ivfflat (embedding)
            WITH (lists = 100);
        """)
        conn.commit()

        print("Index on 'embedding' created successfully!")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating index: {e}")

if __name__ == "__main__":
    create_pgvector_index()
