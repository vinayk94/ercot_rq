import logging
import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
from dotenv import load_dotenv

from src.db.operations import cleanup_database, store_chunks_in_db, store_extracted_text_in_db


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline_debug.log"),
        logging.StreamHandler()
    ]
)

load_dotenv()
postgres_uri = os.getenv("POSTGRESQL_URI")

class ExcelLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        try:
            logging.debug(f"Loading Excel file: {self.file_path}")
            sheets = pd.read_excel(self.file_path, sheet_name=None)
            for sheet_name, sheet_data in sheets.items():
                text = sheet_data.to_string(index=False)
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"sheet_name": sheet_name, "source": self.file_path}
                    )
                )
        except Exception as e:
            logging.error(f"Error loading Excel file {self.file_path}: {e}")
        return documents

def extract_text_from_docs(directory):
    processed_data = []
    skipped_files = 0
    processed_paths = set()  # Track already processed files

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            
            # Skip if already processed
            if file_path in processed_paths:
                continue
                
            processed_paths.add(file_path)
            ext = file.split('.')[-1].lower()

            try:
                if ext == 'docx':
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif ext == 'pdf':
                    loader = PyPDFLoader(file_path)
                elif ext in ['xls', 'xlsx']:
                    loader = ExcelLoader(file_path)
                else:
                    logging.info(f"Skipping unsupported file: {file}")
                    continue

                documents = loader.load()
                for doc in documents:
                    processed_data.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })

                logging.info(f"Processed file: {file}")
            except Exception as e:
                logging.error(f"Failed to process {file}: {e}")
                skipped_files += 1

    return processed_data

def chunk_text(data, chunk_size=500, chunk_overlap=50):  # Reduced overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunked_data = []

    for doc in data:
        chunks = splitter.split_text(doc["content"])
        for chunk in chunks:
            chunked_data.append({
                "chunk": chunk,
                "metadata": doc["metadata"]
            })

    return chunked_data

def main():
    try:
        directory = "./data/downloads"

        # Clean up database first
        cleanup_database(postgres_uri)

        # Extract text
        logging.info("Starting text extraction...")
        extracted_text = extract_text_from_docs(directory)
        new_docs, skipped_docs = store_extracted_text_in_db(extracted_text, postgres_uri)
        logging.info(f"Document processing complete: {new_docs} new, {skipped_docs} duplicates")

        # Chunk text
        logging.info("Starting chunking process...")
        chunked_text = chunk_text(extracted_text)
        logging.info(f"Generated {len(chunked_text)} chunks")

        # Store chunks
        new_chunks, skipped_chunks = store_chunks_in_db(chunked_text, postgres_uri)
        logging.info(f"Chunk storage complete: {new_chunks} new, {skipped_chunks} duplicates")

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
    finally:
        logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()