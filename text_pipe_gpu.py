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
    
    theme_dict = {}
    for word, score in themes:
        theme_dict[word] = max(score, theme_dict.get(word, 0))
    
    return sorted(theme_dict.items(), key=lambda x: x[1], reverse=True)[:n_themes]

class DataManager:
    def __init__(self, source_url=None, local_file=None, cache_dir='data'):
        self.data_dir = Path(cache_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.source_url = source_url
        self.local_file = local_file
        
        # Determine the base filename
        if local_file:
            self.base_filename = Path(local_file).stem
        elif source_url:
            self.base_filename = hashlib.md5(source_url.encode()).hexdigest()[:10]
        else:
            raise ValueError("Either source_url or local_file must be provided")
            
        self.raw_path = self.data_dir / f'{self.base_filename}_raw.txt'
        self.metadata_path = self.data_dir / f'{self.base_filename}_metadata.json'

    def get_file_hash(self, filepath):
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def download_or_copy_text(self):
        need_update = True
        metadata = {
            'last_updated': None, 
            'file_hash': None, 
            'source_url': self.source_url,
            'local_file': str(self.local_file) if self.local_file else None
        }

        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

        if self.raw_path.exists():
            current_hash = self.get_file_hash(self.raw_path)
            if current_hash == metadata['file_hash']:
                print(f"Using cached text from {metadata['last_updated']}")
                need_update = False
            else:
                print("File hash mismatch, updating text")

        if need_update:
            if self.local_file:
                print(f"Copying text from {self.local_file}")
                try:
                    with open(self.local_file, 'r', encoding='utf-8') as src, \
                         open(self.raw_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                except Exception as e:
                    if not self.raw_path.exists():
                        raise Exception(f"Failed to copy text and no cached version exists: {e}")
                    print(f"Copy failed, using cached version: {e}")
            else:
                print(f"Downloading text from {self.source_url}")
                try:
                    response = requests.get(self.source_url)
                    response.raise_for_status()
                    with open(self.raw_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                except requests.RequestException as e:
                    if not self.raw_path.exists():
                        raise Exception(f"Failed to download text and no cached version exists: {e}")
                    print(f"Download failed, using cached version: {e}")

            metadata.update({
                'last_updated': datetime.now().isoformat(),
                'file_hash': self.get_file_hash(self.raw_path)
            })
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print("Text update complete and verified")

        return self.raw_path.read_text(encoding='utf-8')

    def save_processed_data(self, output_data, volume_data):
        themes_path = self.data_dir / f'{self.base_filename}_themes.json'
        with open(themes_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        volumes_path = self.data_dir / f'{self.base_filename}_volumes.json'
        with open(volumes_path, 'w') as f:
            json.dump(volume_data, f, indent=2)
            
        print(f"Data saved and validated in {self.data_dir}:")
        print(f"- Themes data: {themes_path}")
        print(f"- Volume data: {volumes_path}")


class TextProcessor:
    def __init__(self, chunk_size=1000, overlap=100, min_chunk_size=100):
        """
        Initialize text processor with configurable chunking parameters.
        
        Args:
            chunk_size (int): Target size for text chunks in characters
            overlap (int): Number of characters to overlap between chunks
            min_chunk_size (int): Minimum size for a valid chunk
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def clean_text(self, text):
        """Basic text cleaning."""
        # Remove Project Gutenberg header and footer
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "The Project Gutenberg EBook of"
        ]
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "End of the Project Gutenberg"
        ]
        
        # Find start of actual content
        start_pos = 0
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                start_pos = text.find("\n", pos) + 1
                break
                
        # Find end of actual content
        end_pos = len(text)
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                end_pos = pos
                break
                
        text = text[start_pos:end_pos]
        
        # Remove multiple newlines and whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def create_chunks_from_text(self, text):
        """Split text into overlapping chunks."""
        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk_size, store current chunk and start new one
            if current_length + sentence_length > self.chunk_size and current_length >= self.min_chunk_size:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap - keep last few sentences
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) > self.overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_size += len(s)
                
                current_chunk = overlap_chunk
                current_length = overlap_size
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if it's long enough
        if current_chunk and current_length >= self.min_chunk_size:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def parse_text(self, text):
        """Parse text into sections using sentence-aware chunking."""
        text = self.clean_text(text)
        chunks = self.create_chunks_from_text(text)
        
        sections = []
        for i, chunk in enumerate(chunks, 1):
            if len(chunk) >= self.min_chunk_size:
                sections.append({
                    'title': f'Section {i}',
                    'content': chunk,
                    'position': i
                })
        
        print(f"Successfully processed {len(sections)} sections")
        
        if not sections:
            raise ValueError("No sections were successfully parsed from the text")
            
        return sections

class GPUAcceleratedPipeline:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
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

        vectorizer = TfidfVectorizer(max_features=20000)
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


def main(source_url=None, local_file=None, n_clusters=17, chunk_size=1000, overlap=100):
    print("Starting GPU-accelerated text processing pipeline...")
    total_start = time.time()

    # Initialize components
    data_manager = DataManager(source_url=source_url, local_file=local_file)
    text_processor = TextProcessor(chunk_size=chunk_size, overlap=overlap)
    gpu_pipeline = GPUAcceleratedPipeline()

    # Download/load and parse text
    raw_text = data_manager.download_or_copy_text()
    sections = text_processor.parse_text(raw_text)

    n_sections = len(sections)
    print(f"\nProcessing {n_sections} sections with GPU acceleration...")

    # Adjust n_clusters based on number of sections
    n_clusters = min(n_clusters, max(3, n_sections // 3))  # Ensure we don't have too many clusters
    print(f"Adjusted number of clusters to {n_clusters}")

    # Step 1: Create and fit the vectorizer with adjusted parameters
    print("\nStep 1/5: Creating document vectors...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words='english',
        max_df=0.95,  # More permissive max_df
        min_df=2,     # Require words to appear at least twice
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
    n_components = min(N_TOPICS, n_sections - 1)  # Adjust number of topics based on sections
    nmf_model = NMF(n_components=n_components, random_state=42, max_iter=300)
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
            if len(cluster_docs) > 0:  # Only process non-empty clusters
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

    return pd.DataFrame({
        'title': [s['title'] for s in sections],
        'content': [s['content'] for s in sections],
        'position': [s['position'] for s in sections],
        'cluster': labels
    })

if __name__ == "__main__":
    # Example usage with smaller chunks for better processing
    processed_data = main(
        source_url="https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        n_clusters=24,
        chunk_size=500,  # Smaller chunks
        overlap=50       # Smaller overlap
    )

## From URL
#processed_data = main(
#    source_url="https://www.gutenberg.org/files/1342/1342-0.txt",
#    chunk_size=500,
#    overlap=50
#)

## From local file
#processed_data = main(
#    local_file="path/to/your/file.txt",
#    chunk_size=500,
#    overlap=50
#)
