
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env
load_dotenv()
print(os.getenv("POSTGRESQL_URI")) 

def enable_pgvector():
    # Replace with your actual PostgreSQL URI from Aiven
    postgres_uri = os.getenv("POSTGRESQL_URI")

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(postgres_uri)
        cur = conn.cursor()

        # Enable the pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        print("pgvector extension enabled successfully!")
        cur.close()
        conn.close()
    except Exception as e:
        print("Error enabling pgvector:", e)

if __name__ == "__main__":
    enable_pgvector()
