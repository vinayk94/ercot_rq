from langchain.embeddings.base import Embeddings
import requests
import logging
from typing import List
import time

class ERCOTEmbeddings(Embeddings):
    """Custom embeddings class with key rotation"""
    
    def __init__(self, key_manager, model_name: str = "jina-embeddings-v3"):
        self.key_manager = key_manager
        self.model_name = model_name
        self.dimensions = 1024  # Jina's default dimension
        self.url = 'https://api.jina.ai/v1/embeddings'

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.key_manager.get_current_key()}'
        }
        
        data = {
            "model": self.model_name,
            "task": "text-matching",
            "dimensions": self.dimensions,
            "embedding_type": "float",
            "input": texts
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
        except Exception as e:
            if "token limit exceeded" in str(e).lower():
                self.key_manager.mark_key_failed(
                    self.key_manager.get_current_key(),
                    "Token limit exceeded"
                )
                if self.key_manager.has_available_keys():
                    # Retry with new key
                    time.sleep(1)  # Small delay before retry
                    return self._get_embeddings(texts)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        embeddings = self._get_embeddings([text])
        return embeddings[0]