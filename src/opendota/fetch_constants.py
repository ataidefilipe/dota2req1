import argparse
import requests
import json
import os
import logging
from ..config import Config

logger = logging.getLogger(__name__)

def fetch_constants(output_dir='data'):
    """
    Fetches heroes and potentially other constants from OpenDota/dotaconstants CDN.
    Saves to data/heroes.json.
    """
    # Heroes
    url = "https://api.opendota.com/api/heroes"
    logger.info(f"Fetching heroes from {url}...")
    
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        heroes = resp.json()
        
        # We want to map Name -> ID and ID -> Name
        # heroes is a list of dicts: {'id': 1, 'name': 'npc_dota_hero_antimage', 'localized_name': 'Anti-Mage', ...}
        
        # Let's save the raw list
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'heroes.json')
        with open(path, 'w') as f:
            json.dump(heroes, f, indent=2)
            
        logger.info(f"Saved {len(heroes)} heroes to {path}")
        
    except Exception as e:
        logger.error(f"Failed to fetch heroes: {e}")

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    fetch_constants()
