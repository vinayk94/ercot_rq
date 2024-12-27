import requests
import logging

class EmbeddingGenerator:
    """
    A class for generating embeddings using the Galadriel API.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.galadriel.com/embeddings"
        logging.basicConfig(level=logging.INFO)

    def generate_embedding(self, text: str) -> list:
        """
        Generate an embedding for a given piece of text.

        Args:
            text (str): The input text for which to generate an embedding.

        Returns:
            list: The generated embedding as a list of floats, or an empty list on failure.
        """
        try:
            logging.info(f"Generating embedding for text: {text[:50]}...")  # Log a preview of the text
            response = requests.post(
                self.url,
                json={"text": text},
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return []
