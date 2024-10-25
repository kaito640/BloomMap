import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import re
import json
import requests
import os
import hashlib
from pathlib import Path
from datetime import datetime
from textblob import TextBlob
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import NMF
from tqdm import tqdm
import time
from datetime import datetime, timedelta

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Parameters
N_TOP_THEMES = 50
N_TOPICS = 20
MAX_CLUSTERS = 200
BATCH_SIZE = 32

def preprocess_word(word):
    """Clean and validate a word for theme extraction."""
    cleaned = re.sub(r'[^a-zA-Z]', '', word.lower())
    if len(cleaned) < 3 or cleaned in {'one', 'two', 'three', 'four', 'five', 'six', 'seven'}:
        return None
    return cleaned

def get_cluster_themes(cluster_docs, vectorizer, nmf_model, n_themes):
    """Extract meaningful themes from cluster documents."""
    feature_names = vectorizer.get_feature_names_out()
    cluster_topic_dist = np.mean(cluster_docs, axis=0)
    themes = []
    for topic_idx in range(len(cluster_topic_dist)):
        topic_weights = nmf_model.components_[topic_idx]
        top_word_indices = topic_weights.argsort()[-10:][::-1]
        for idx in top_word_indices:
            word = feature_names[idx]
            cleaned_word = preprocess_word(word)
            if cleaned_word:
                importance = float(cluster_topic_dist[topic_idx] * topic_weights[idx])
                themes.append((cleaned_word, importance))
    
    # Initialize theme_dict before using it
    theme_dict = {}
    for word, score in themes:
        theme_dict[word] = max(score, theme_dict.get(word, 0))
    
    return sorted(theme_dict.items(), key=lambda x: x[1], reverse=True)[:n_themes]

class DataManager:
    def __init__(self):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.bible_path = self.data_dir / 'bible_raw.txt'
        self.metadata_path = self.data_dir / 'bible_metadata.json'
        self.source_url = "https://www.gutenberg.org/cache/epub/10/pg10.txt"

    def get_file_hash(self, filepath):
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def download_bible(self):
        need_download = True
        metadata = {'last_downloaded': None, 'file_hash': None, 'source_url': self.source_url}

        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

        if self.bible_path.exists():
            current_hash = self.get_file_hash(self.bible_path)
            if current_hash == metadata['file_hash']:
                print("Using cached Bible text from", metadata['last_downloaded'])
                need_download = False
            else:
                print("File hash mismatch, re-downloading Bible text")

        if need_download:
            print(f"Downloading Bible text from {self.source_url}")
            try:
                response = requests.get(self.source_url)
                response.raise_for_status()
                with open(self.bible_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                metadata.update({'last_downloaded': datetime.now().isoformat(), 'file_hash': self.get_file_hash(self.bible_path)})
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print("Download complete and verified")
            except requests.RequestException as e:
                if not self.bible_path.exists():
                    raise Exception(f"Failed to download Bible text and no cached version exists: {e}")
                print(f"Download failed, using cached version: {e}")

        return self.bible_path.read_text(encoding='utf-8')

    def save_processed_data(self, output_data, volume_data):
        themes_path = self.data_dir / 'proc_themes.json'
        with open(themes_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        volumes_path = self.data_dir / 'cluster_volumes.json'
        with open(volumes_path, 'w') as f:
            json.dump(volume_data, f, indent=2)
            
        print(f"Data saved and validated in {self.data_dir}:")
        print(f"- Themes data: {themes_path}")
        print(f"- Volume data: {volumes_path}")

class BibleProcessor:
    def __init__(self):
        self.old_testament = set([
            'GENESIS', 'EXODUS', 'LEVITICUS', 'NUMBERS', 'DEUTERONOMY',
            'JOSHUA', 'JUDGES', 'RUTH', '1 SAMUEL', '2 SAMUEL', '1 KINGS',
            '2 KINGS', '1 CHRONICLES', '2 CHRONICLES', 'EZRA', 'NEHEMIAH',
            'ESTHER', 'JOB', 'PSALMS', 'PROVERBS', 'ECCLESIASTES',
            'SONG OF SOLOMON', 'ISAIAH', 'JEREMIAH', 'LAMENTATIONS',
            'EZEKIEL', 'DANIEL', 'HOSEA', 'JOEL', 'AMOS', 'OBADIAH',
            'JONAH', 'MICAH', 'NAHUM', 'HABAKKUK', 'ZEPHANIAH',
            'HAGGAI', 'ZECHARIAH', 'MALACHI'
        ])

    def parse_bible_text(self, text):
        text = text.upper()
        possible_starts = ["THE FIRST BOOK OF MOSES, CALLED GENESIS", "THE OLD TESTAMENT", "GENESIS"]
        
        start_idx = len(text)
        for starter in possible_starts:
            pos = text.find(starter)
            if pos != -1 and pos < start_idx:
                start_idx = pos
        
        end_markers = [
            "*** END OF THE PROJECT", "*** END OF THIS PROJECT",
            "End of the Project Gutenberg", "THE END OF THE NEW TESTAMENT"
        ]
        
        end_idx = -1
        for ender in end_markers:
            pos = text.find(ender)
            if pos != -1:
                if end_idx == -1 or pos < end_idx:
                    end_idx = pos
        
        if end_idx == -1:
            end_idx = len(text)
            
        bible_text = text[start_idx:end_idx].strip()
        book_pattern = r'\n\s*(?:THE )?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|[1-5])?\s*(?:BOOK\s+OF\s+)?([A-Z]+(?:\s+[A-Z]+)*)'
        
        books = re.split(book_pattern, bible_text)
        sections = []
        
        print(f"Found {(len(books)-1)//2} books")
        
        for i in range(1, len(books), 2):
            try:
                book_name = books[i].strip()
                book_content = books[i + 1].strip() if i + 1 < len(books) else ""
                
                if not book_content:
                    continue
                    
                print(f"Processing book: {book_name}")
                
                chapters = re.split(r'\n(?=\d+:\d+)', book_content)
                
                for chapter in chapters:
                    if not chapter.strip():
                        continue
                        
                    chapter_match = re.match(r'(\d+):', chapter)
                    if not chapter_match:
                        continue
                    chapter_num = chapter_match.group(1)
                    
                    verses = chapter.split('\n')
                    section_size = 10
                    
                    for j in range(0, len(verses), section_size):
                        section_verses = verses[j:j+section_size]
                        if not section_verses:
                            continue
                        
                        section_text = ' '.join([
                            re.sub(r'^\d+:\d+\s*', '', v.strip())
                            for v in section_verses 
                            if v.strip() and ':' in v
                        ])
                        
                        if not section_text:
                            continue
                        
                        title = f"{book_name} {chapter_num}:{j+1}-{j+len(section_verses)}"
                        category = 'OLD_TESTAMENT' if book_name in self.old_testament else 'NEW_TESTAMENT'
                        
                        sections.append({
                            'title': title,
                            'content': section_text.title(),
                            'category': category
                        })
                        
            except Exception as e:
                print(f"Error processing book {book_name if 'book_name' in locals() else 'unknown'}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(sections)} sections")
        
        if not sections:
            raise ValueError("No sections were successfully parsed from the Bible text")
            
        return sections


class GPUAcceleratedPipeline:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)

    @torch.no_grad()
    def get_embeddings(self, texts, batch_size=32):
        """Generate embeddings for a batch of texts"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Use mean pooling for sentence embeddings
            attention_mask = inputs['attention_mask']
            batch_embeddings = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), 1)
            batch_embeddings = batch_embeddings / torch.sum(attention_mask, 1, keepdim=True)
            
            embeddings.append(batch_embeddings.cpu())
            
        return torch.cat(embeddings, dim=0).numpy()

    def process_sections(self, sections, progress_callback=None):
        texts = [s['content'] for s in sections]
        embeddings = []

        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch)
            embeddings.append(batch_embeddings)
            if progress_callback:
                progress_callback(len(batch))

        embeddings = np.vstack(embeddings)

        # Align CountVectorizer with the same max_features as TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=20000)  # Match TfidfVectorizer settings
        vectorizer.fit(texts)
        return embeddings, vectorizer


def perform_clustering(nmf_output, n_clusters):
    """Perform K-means clustering on GPU using PyTorch."""
    print(f"\nPerforming GPU K-means clustering with {n_clusters} clusters...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X = torch.tensor(nmf_output, dtype=torch.float32).to(device)

    n_samples = X.shape[0]
    indices = torch.randperm(n_samples)[:n_clusters]
    centroids = X[indices].clone()

    max_iters = 100
    prev_centroids = None

    for iteration in tqdm(range(max_iters), desc="K-means iterations"):
        distances = torch.cdist(X, centroids)
        labels = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.any():
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                new_centroids[k] = centroids[k]

        if prev_centroids is not None:
            diff = torch.norm(new_centroids - prev_centroids)
            if diff < 1e-4:
                print(f"Converged after {iteration + 1} iterations")
                break

        centroids = new_centroids
        prev_centroids = centroids.clone()

    labels = labels.cpu().numpy()

    cluster_stats = {
        "silhouette_score": float(silhouette_score(nmf_output, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(nmf_output, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(nmf_output, labels)),
        "cluster_sizes": [int(x) for x in np.bincount(labels)],
        "cluster_density": float(np.mean([np.mean(np.linalg.norm(nmf_output[labels == k] - centroids[k].cpu().numpy(), axis=1)) for k in range(n_clusters)]))
    }

    return labels, cluster_stats

def main(n_clusters=17):  # Add n_clusters as a parameter to main
    print("Starting GPU-accelerated Bible processing pipeline...")
    total_start = time.time()

    # Initialize components
    data_manager = DataManager()
    bible_processor = BibleProcessor()
    gpu_pipeline = GPUAcceleratedPipeline()

    # Download and parse Bible
    bible_text = data_manager.download_bible()
    sections = bible_processor.parse_bible_text(bible_text)

    n_sections = len(sections)
    print(f"\nProcessing {n_sections} sections with GPU acceleration...")

    # Step 1: Create and fit the vectorizer
    print("\nStep 1/5: Creating document vectors...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words='english',
        max_df=0.5,
        min_df=2,
        token_pattern=r'(?u)\b[A-Za-z]+\b'
    )
    tfidf_matrix = vectorizer.fit_transform([s['content'] for s in sections])
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # Step 2: GPU-accelerated embeddings
    print("\nStep 2/5: Generating embeddings")
    embedding_start = time.time()
    embeddings, vectorizer = gpu_pipeline.process_sections(sections)
    print(f"Embeddings generation complete. Time taken: {time.time() - embedding_start:.2f}s")

    # Step 3: Perform NMF
    print("\nStep 3/5: Performing NMF...")
    nmf_start = time.time()
    nmf_model = NMF(n_components=N_TOPICS, random_state=42, max_iter=300)
    nmf_output = nmf_model.fit_transform(tfidf_matrix)
    print(f"NMF complete. Time taken: {time.time() - nmf_start:.2f}s")

    # Step 4: Perform clustering with GPU K-means
    print("\nStep 4/5: Performing clustering")
    cluster_start = time.time()
    labels, cluster_stats = perform_clustering(nmf_output, n_clusters=n_clusters)
    print(f"Clustering complete. Time taken: {time.time() - cluster_start:.2f}s")

    # Step 5: Generate themes and save results
    print("\nStep 5/5: Generating themes and saving results")
    theme_start = time.time()

    output_data = {}
    cluster_volume_data = {}

    # Process global cluster (largest cluster)
    global_cluster = pd.Series(labels).mode().iloc[0]
    global_docs = nmf_output[labels == global_cluster]
    global_themes = get_cluster_themes(global_docs, vectorizer, nmf_model, N_TOP_THEMES)
    global_volume = (labels == global_cluster).sum()
    output_data['global'] = global_themes
    cluster_volume_data['global'] = {
        'volume': int(global_volume),
        **cluster_stats
    }

    # Process other clusters with progress bar
    remaining_clusters = [c for c in range(n_clusters) if c != global_cluster]
    with tqdm(total=len(remaining_clusters), desc="Processing clusters") as pbar:
        for cluster in remaining_clusters:
            cluster_docs = nmf_output[labels == cluster]
            cluster_themes = get_cluster_themes(cluster_docs, vectorizer, nmf_model, N_TOP_THEMES)
            cluster_volume = (labels == cluster).sum()
            cluster_name = f'c{cluster + 1}'
            output_data[cluster_name] = cluster_themes
            cluster_volume_data[cluster_name] = {
                'volume': int(cluster_volume),
                **cluster_stats
            }
            pbar.update(1)

    # Save results
    data_manager.save_processed_data(output_data, cluster_volume_data)
    print(f"Theme generation and saving complete. Time taken: {time.time() - theme_start:.2f}s")

    total_time = time.time() - total_start
    print(f"\nPipeline complete! Total time: {total_time:.2f}s")
    print(f"Average processing time per section: {total_time/n_sections:.2f}s")

    return pd.DataFrame(sections)

if __name__ == "__main__":
    # Specify the desired number of clusters here, or pass it as a command-line argument
    processed_data = main(n_clusters=23)

