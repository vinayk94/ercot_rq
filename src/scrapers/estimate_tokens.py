from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    UnstructuredExcelLoader
)
import os

# Supported file loaders
loaders = {
    "docx": UnstructuredWordDocumentLoader,
    "pdf": UnstructuredPDFLoader,
    "xls": UnstructuredExcelLoader,
    "xlsx": UnstructuredExcelLoader,
}

# Token counter function
def count_tokens(text):
    return len(text.split())

# Process and estimate tokens
total_tokens = 0
data_dir = "./data/downloads"

for root, _, files in os.walk(data_dir):
    for file in files:
        ext = file.split('.')[-1].lower()
        if ext in loaders:
            try:
                filepath = os.path.join(root, file)
                loader = loaders[ext](filepath)
                documents = loader.load()
                
                # Use a text splitter to handle large files
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                chunks = splitter.split_documents(documents)
                
                # Count tokens in each chunk
                for chunk in chunks:
                    total_tokens += count_tokens(chunk.page_content)
            except Exception as e:
                print(f"Error processing {file}: {e}")

print(f"Estimated total tokens: {total_tokens}")
