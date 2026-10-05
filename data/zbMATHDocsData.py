"""
Fetch document metadata from zbMATH Open API for a list of document IDs.

Usage:
    python zbMATHDocsData.py

Output:
    - zbMATHDocsData.csv: CSV with columns ['document_id', 'title', 'authors', 'year', 'msc', 'language', 'source']

Note:
    - Uses the zbMATH Open REST API: https://api.zbmath.org/v1/
    - Rate-limited (~1 request/sec recommended); includes delays.
"""

import pandas as pd
import requests
import time
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base URL for zbMATH Open API
BASE_URL = "https://api.zbmath.org/v1/document/"

# Output filename
OUTPUT_FILE = "zbMATHDocsData.csv"

def fetch_document(doc_id: str) -> Optional[Dict]:
    """
    Fetch a single document from zbMATH Open API.
    
    Returns:
        Dict with fields: document_id, title, authors, year, msc, language, source
        or None if failed.
    """
    url = f"{BASE_URL}{doc_id}.json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            result = data.get("result", {})
            
            # Extract fields
            title = result.get("title", "")
            authors = "; ".join([a.get("name", "") for a in result.get("author", [])])
            year = result.get("publication_year", "")
            msc = "; ".join(result.get("msc", []))
            language = result.get("language", {}).get("name", "") if result.get("language") else ""
            source = result.get("source", {}).get("journal", {}).get("name", "") if result.get("source") else ""
            
            return {
                "document_id": doc_id,
                "title": title,
                "authors": authors,
                "year": year,
                "msc": msc,
                "language": language,
                "source": source
            }
        else:
            logger.warning(f"Failed to fetch {doc_id}: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching {doc_id}: {e}")
        return None

def main():
    # Load document IDs from a CSV
    # df_ids = pd.read_csv("recommendations.csv")  # must have 'Recommendation' column
    # DOCUMENT_IDS = df_ids["Recommendation"].dropna().astype(str).unique().tolist()
    logger.info(f"Fetching data for {len(DOCUMENT_IDS)} document IDs...")

    records = []
    for i, doc_id in enumerate(DOCUMENT_IDS):
        logger.info(f"Processing ({i+1}/{len(DOCUMENT_IDS)}): {doc_id}")
        record = fetch_document(doc_id)
        if record:
            records.append(record)
        # Rate limit: 1 request per second
        time.sleep(1)

    if records:
        output_df = pd.DataFrame(records)
        output_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Saved {len(records)} records to {OUTPUT_FILE}")
    else:
        logger.error("No records fetched. Check document IDs and network.")

if __name__ == "__main__":
    main()