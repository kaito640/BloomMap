import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import string
import re
import os
import csv
import multiprocessing as mp

nltk.download('punkt', quiet=True)

def read_csv_robust(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = []
        for row in reader:
            if len(row) >= 4:  # Ensure we have at least the required columns
                data.append({
                    'document_id': row[1],  # Using filename as document_id
                    'category': row[0],
                    'title': row[2],
                    'content': ' '.join(row[3:])  # Join all remaining fields as content
                })
    return pd.DataFrame(data)

def preprocess_text(text):
    # Remove punctuation and junk, but keep all words
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def process_chunk(chunk):
    return chunk.apply(preprocess_text)

def identify_themes(vectorizer, X, cluster_labels, n_themes=50):
    feature_names = vectorizer.get_feature_names_out()
    themes = []
    
    # Global themes
    global_tfidf = X.mean(axis=0)
    global_themes = [(feature_names[i], score) for i, score in enumerate(global_tfidf.A1)]
    global_themes.sort(key=lambda x: x[1], reverse=True)
    themes.extend([('global', theme, score) for theme, score in global_themes[:n_themes]])
    
    # Local themes
    for cluster in np.unique(cluster_labels):
        cluster_docs = X[cluster_labels == cluster]
        cluster_tfidf = cluster_docs.mean(axis=0)
        cluster_themes = [(feature_names[i], score) for i, score in enumerate(cluster_tfidf.A1)]
        cluster_themes.sort(key=lambda x: x[1], reverse=True)
        themes.extend([(f'cluster_{cluster}', theme, score) for theme, score in cluster_themes[:n_themes]])
    
    return themes


def process_data(file_path, n_clusters=10, n_themes=50):
    df = read_csv_robust(file_path)

    # Parallelize preprocessing
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    chunks = np.array_split(df['content'], num_processes)
    processed_chunks = pool.map(process_chunk, chunks)
    pool.close()
    pool.join()

    df['processed_content'] = pd.concat(processed_chunks).reset_index(drop=True)

    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df['processed_content'])

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    themes = identify_themes(vectorizer, X, cluster_labels, n_themes)

    return themes, df['document_id'].tolist(), cluster_labels.tolist()

def save_outputs(themes, document_ids, cluster_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save themes
    with open(f"{output_dir}/themes.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['theme', 'cluster', 'importance'])
        for cluster, theme, score in themes:
            writer.writerow([theme, cluster, score])

    # Save document to cluster mapping
    with open(f"{output_dir}/document_cluster_map.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['document_id', 'cluster'])
        for doc_id, cluster in zip(document_ids, cluster_labels):
            writer.writerow([doc_id, f'cluster_{cluster}'])


if __name__ == "__main__":
    file_path = 'data/bbc-news-data.csv'
    output_dir = './'

    themes, document_ids, cluster_labels = process_data(file_path)
    save_outputs(themes, document_ids, cluster_labels, output_dir)

    print("Results saved to output directory")
