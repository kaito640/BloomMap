import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import string
import re
import os
import csv

nltk.download('punkt', quiet=True)

def read_csv_robust(file_path):
    # This function looks fine, no changes needed
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

def identify_global_words(df, threshold_percentage):
    # Identify words that appear in a high percentage of documents
    word_doc_freq = Counter()
    total_docs = len(df)
    
    for doc in df['processed_content']:
        word_doc_freq.update(set(doc.split()))
    
    return {word for word, count in word_doc_freq.items() if count / total_docs > threshold_percentage}

def process_data(file_path, n_clusters=10, top_n=50, global_word_threshold=0.5):
    df = read_csv_robust(file_path)
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    # Identify global words
    global_words = identify_global_words(df, global_word_threshold)
    
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df['processed_content'])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Ensure each cluster has at least one sample
    unique_clusters = np.unique(df['cluster'])
    if len(unique_clusters) < n_clusters:
        print(f"Warning: Only {len(unique_clusters)} clusters were formed. Adjusting n_clusters.")
        n_clusters = len(unique_clusters)
    
    # Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, df['cluster'])
    
    print(f"Number of samples in each cluster: {np.bincount(df['cluster'])}")
    print(f"Number of estimators in Random Forest: {len(clf.estimators_)}")
    print(f"Number of global words: {len(global_words)}")
    
    top_predictive_words = get_top_predictive_words(vectorizer, clf, n_clusters, top_n, global_words)
    
    return df, top_predictive_words, global_words

def save_outputs(df, top_predictive_words, global_words, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    df[['document_id', 'cluster']].to_csv(f"{output_dir}/document_to_cluster_map.csv", index=False)
    
    cluster_summary = []
    for i, cluster_words in enumerate(top_predictive_words):
        cluster_name = 'cluster_0' if i == 0 else f'cluster_{i}'
        for word, score in cluster_words:
            cluster_summary.append({
                'cluster': cluster_name,
                'word': word,
                'predictive_score': score
            })
    
    pd.DataFrame(cluster_summary).to_csv(f"{output_dir}/cluster_summary.csv", index=False)
    
    pd.DataFrame(list(global_words), columns=['word']).to_csv(f"{output_dir}/global_words.csv", index=False)

if __name__ == "__main__":
    file_path = 'data/bbc-news-data.csv'
    output_dir = './'
    
    df, top_predictive_words, global_words = process_data(file_path)
    save_outputs(df, top_predictive_words, global_words, output_dir)
    
    print("Results saved to output directory")

