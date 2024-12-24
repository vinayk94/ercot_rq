import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import aiohttp
from bs4 import BeautifulSoup
import asyncio
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ERCOTDocument:
    title: str
    url: str
    file_type: str
    section: str = "Unknown"

class ERCOTScraper:
    def __init__(self):
        self.base_url = "https://www.ercot.com"
        self.rq_url = f"{self.base_url}/services/rq"
        self.document_patterns = ['.docx', '.pdf', '.xls', '.xlsx']
        self.metadata = []  # Centralized metadata storage

    async def fetch_html(self, url: str, session: aiohttp.ClientSession) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML from a URL."""
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    return BeautifulSoup(html, 'html.parser')
                else:
                    logging.error(f"Failed to fetch {url}: {response.status}")
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
        return None

    async def get_sections(self, session: aiohttp.ClientSession) -> List[Dict[str, str]]:
        """Get all sections under the RQ page."""
        soup = await self.fetch_html(self.rq_url, session)
        if not soup:
            return []

        sections = []
        for link in soup.find_all('a', href=lambda x: x and '/services/rq/' in x):
            section_name = link.text.strip()
            section_url = f"{self.base_url}{link['href']}"
            sections.append({'name': section_name, 'url': section_url})
            logging.info(f"Discovered section: {section_name} ({section_url})")
        return sections

    async def scrape_documents(self, section: Dict[str, str], session: aiohttp.ClientSession) -> List[ERCOTDocument]:
        """Scrape documents from a specific section."""
        soup = await self.fetch_html(section['url'], session)
        if not soup:
            return []

        documents = []
        for link in soup.find_all('a', href=lambda x: x and any(x.endswith(ext) for ext in self.document_patterns)):
            title = link.text.strip()
            url = f"{self.base_url}{link['href']}" if not link['href'].startswith('http') else link['href']
            file_type = url.split('.')[-1]
            document = ERCOTDocument(title=title, url=url, file_type=file_type, section=section['name'])
            documents.append(document)
            self.metadata.append(document.__dict__)  # Store metadata for centralized saving
            logging.info(f"Found document: {title} ({file_type}) in section {section['name']}")
        return documents

    async def save_documents(self, documents: List[ERCOTDocument], save_path: str = "./data/downloads"):
        """Download and save documents locally."""
        async with aiohttp.ClientSession() as session:
            for doc in documents:
                try:
                    # Construct the file path
                    filepath = os.path.join(save_path, doc.section, f"{doc.title}.{doc.file_type}")
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    # Download the file
                    async with session.get(doc.url, timeout=10) as response:
                        if response.status == 200:
                            with open(filepath, 'wb') as f:
                                f.write(await response.read())
                            logging.info(f"Saved: {filepath}")
                except Exception as e:
                    logging.error(f"Failed to download {doc.title}: {e}")

    def save_metadata_to_json(self, output_file="./data/downloads/metadata.json"):
        """Save all metadata to a single JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        logging.info(f"Metadata saved to {output_file}")

async def main():
    scraper = ERCOTScraper()
    logging.info("Starting ERCOT Scraper...")

    # Step 1: Scrape all sections and collect metadata
    async with aiohttp.ClientSession() as session:
        sections = await scraper.get_sections(session)
        all_documents = []
        for section in sections:
            documents = await scraper.scrape_documents(section, session)
            all_documents.extend(documents)

    # Step 2: Save documents locally
    await scraper.save_documents(all_documents)

    # Step 3: Save metadata to a single JSON file
    scraper.save_metadata_to_json()

    logging.info("Scraping and saving complete.")

if __name__ == "__main__":
    asyncio.run(main())
