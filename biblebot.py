import pandas as pd
import json
import logging
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import numpy as np
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BibleThemeExtractor:
    def __init__(self, ollama_url: str = "http://localhost:11434", max_retries: int = 10, retry_delay: int = 2):
        self.ollama_url = ollama_url
        self.model = "ALIENTELLIGENCE/holybible"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def create_theme_prompt(self, text: str, perspective: str) -> str:
        """Create different prompts for different perspectives of analysis"""
        prompts = {
            "theological": """Analyze this biblical text from a theological perspective. Extract single-word themes about:
            - Divine attributes (holiness, mercy, justice)
            - Spiritual concepts (salvation, redemption, grace)
            - Faith elements (belief, worship, prayer)
            - Moral teachings (righteousness, virtue, sin)
            Rate each word's theological significance from 0.0 to 1.0.""",
            
            "narrative": """Analyze this biblical text from a narrative perspective. Extract single-word themes about:
            - Key characters and their attributes
            - Actions and events
            - Locations and settings
            - Symbolic objects and elements
            Rate each word's narrative importance from 0.0 to 1.0.""",
            
            "emotional": """Analyze this biblical text from an emotional/psychological perspective. Extract single-word themes about:
            - Human emotions and feelings
            - Interpersonal relationships
            - Inner struggles and conflicts
            - Personal transformations
            Rate each word's emotional resonance from 0.0 to 1.0.""",
            
            "literary": """Analyze this biblical text from a literary perspective. Extract single-word themes about:
            - Literary motifs and patterns
            - Metaphors and symbols
            - Dramatic elements
            - Poetic imagery
            Rate each word's literary significance from 0.0 to 1.0."""
        }
        
        base_rules = """
        Rules:
        - Return ONLY single words (no spaces or hyphens)
        - Choose powerful, meaningful words
        - Each word should be unique
        - Format response as JSON array: [["word1", 0.9], ["word2", 0.8]]
        
        Text to analyze:
        """
        
        return prompts[perspective] + base_rules + text

    def validate_themes(self, themes):
        try:
            if not isinstance(themes, list):
                return False
            for item in themes:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    return False
                word, weight = item
                if not isinstance(word, str) or ' ' in word or '-' in word or len(word) < 2:
                    return False
                if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                    return False
            return True
        except Exception:
            return False

    def extract_themes_for_perspective(self, text: str, perspective: str) -> List:
        attempts = 0
        last_error = None
        
        while attempts < self.max_retries:
            try:
                logging.info(f"Extracting {perspective} themes - Attempt {attempts + 1}/{self.max_retries}")
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": self.create_theme_prompt(text, perspective),
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                response_text = result.get('response', '')
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                
                if start >= 0 and end > start:
                    themes = json.loads(response_text[start:end])
                    if self.validate_themes(themes):
                        logging.info(f"Successfully extracted {len(themes)} themes for {perspective}")
                        return themes
                    else:
                        logging.warning(f"Invalid theme format in attempt {attempts + 1}")
                        
            except Exception as e:
                last_error = str(e)
                logging.warning(f"Attempt {attempts + 1}/{self.max_retries} failed: {last_error}")
            
            attempts += 1
            time.sleep(self.retry_delay)
        
        logging.error(f"All {self.max_retries} attempts failed for {perspective}. Last error: {last_error}")
        return []

    def extract_themes(self, text: str, is_global: bool = False) -> List:
        """Extract themes from multiple perspectives and combine results"""
        all_themes = []
        perspectives = ["theological", "narrative", "emotional", "literary"]
        
        # Extract themes from each perspective
        for perspective in perspectives:
            themes = self.extract_themes_for_perspective(text, perspective)
            all_themes.extend(themes)
        
        # Combine and deduplicate themes
        theme_dict = {}
        for word, weight in all_themes:
            word = word.lower()
            if word in theme_dict:
                # Keep highest weight for duplicates
                theme_dict[word] = max(theme_dict[word], weight)
            else:
                theme_dict[word] = weight
        
        # Sort by weight and take top N
        sorted_themes = sorted(theme_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_themes[:50 if is_global else 30]

def count_words(text: str) -> int:
    """Count words in text, handling biblical text specifics."""
    return len([word for word in text.split() if word.strip()])

def main(csv_path: str):
    logging.info("Starting enhanced Bible theme extraction...")
    
    # Initialize theme extractor
    extractor = BibleThemeExtractor()
    
    # Read and validate data
    df = pd.read_csv(csv_path)
    df['word_count'] = df['content'].apply(count_words)
    
    # Initialize output dictionaries
    output_data = {}
    volume_data = {}
    
    # Calculate global metrics
    all_word_count = df['word_count'].sum()
    categories = df['category'].unique()
    
    # Process global themes with expanded perspective
    logging.info("Extracting global themes...")
    all_text = " ".join(df['content'].tolist())
    global_themes = extractor.extract_themes(all_text, is_global=True)
    
    # Store global results
    output_data['global'] = global_themes
    volume_data['global'] = {
        'volume': int(all_word_count),
        'silhouette_score': 0.0,
        'calinski_harabasz_score': 0.0,
        'davies_bouldin_score': 0.0,
        'cluster_sizes': [int(df[df['category'] == cat]['word_count'].sum()) for cat in categories],
        'cluster_density': 0.0
    }
    
    # Process each book
    for category in tqdm(categories, desc="Processing books"):
        try:
            book_text = " ".join(df[df['category'] == category]['content'].tolist())
            book_word_count = df[df['category'] == category]['word_count'].sum()
            
            book_themes = extractor.extract_themes(book_text)
            
            output_data[category] = book_themes
            volume_data[category] = {
                'volume': int(book_word_count),
                'silhouette_score': volume_data['global']['silhouette_score'],
                'calinski_harabasz_score': volume_data['global']['calinski_harabasz_score'],
                'davies_bouldin_score': volume_data['global']['davies_bouldin_score'],
                'cluster_sizes': volume_data['global']['cluster_sizes'],
                'cluster_density': volume_data['global']['cluster_density']
            }
            
        except Exception as e:
            logging.error(f"Error processing book {category}: {e}")
            continue
    
    # Save results
    output_dir = Path('')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save themes
    themes_path = output_dir / 'proc_themes.json'
    with open(themes_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save volumes
    volumes_path = output_dir / 'cluster_volumes.json'
    with open(volumes_path, 'w') as f:
        json.dump(volume_data, f, indent=2)
    
    logging.info("\nProcessing complete!")
    logging.info(f"Results saved to {output_dir}")
    logging.info(f"Themes file: {themes_path}")
    logging.info(f"Volumes file: {volumes_path}")

if __name__ == "__main__":
    main("data/bible_data.csv")
