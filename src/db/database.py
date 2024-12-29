from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse, quote_plus

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

def get_connection_strings():
    """Get properly formatted connection strings for different libraries"""
    DATABASE_URL = os.getenv("POSTGRESQL_URI")
    if not DATABASE_URL:
        raise EnvironmentError("POSTGRESQL_URI is required")

    # Parse the URL to get components
    parsed = urlparse(DATABASE_URL)
    
    # Create SQLAlchemy connection string
    sqlalchemy_url = f"postgresql://{parsed.username}:{quote_plus(parsed.password)}@{parsed.hostname}:{parsed.port}{parsed.path}"
    
    # Create direct PostgreSQL connection string (for PGVector)
    pg_connection = f"postgres://{parsed.username}:{quote_plus(parsed.password)}@{parsed.hostname}:{parsed.port}{parsed.path}"
    
    return sqlalchemy_url, pg_connection

def get_db_engine():
    """Create and return database engine with proper configuration"""
    sqlalchemy_url, _ = get_connection_strings()
    
    try:
        engine = create_engine(
            sqlalchemy_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise

# Create a single engine instance
engine = get_db_engine()