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

def identify_global_words(df, threshold_percentage):
    # Identify words that appear in a high percentage of documents
    word_doc_freq = Counter()
    total_docs = len(df)
    
    for doc in df['processed_content']:
        word_doc_freq.update(set(doc.split()))
    
    return {word for word, count in word_doc_freq.items() if count / total_docs > threshold_percentage}

def get_cluster_word_scores(args):
    cluster_docs, feature_names, global_words, overall_importance = args
    cluster_word_freq = Counter(" ".join(cluster_docs).split())
    total_words = sum(cluster_word_freq.values())
    
    word_scores = {}
    for word, freq in cluster_word_freq.items():
        if word in feature_names and word not in global_words:
            overall_freq = sum(doc.split().count(word) for doc in cluster_docs)
            overall_importance_score = overall_importance[feature_names.tolist().index(word)]
            frequency_ratio = (freq / total_words) / (overall_freq / len(cluster_docs))
            word_scores[word] = frequency_ratio * overall_importance_score
    
    return word_scores

def get_top_predictive_words(vectorizer, clf, df, n_clusters, n_words, global_words):
    feature_names = vectorizer.get_feature_names_out()
    overall_importance = clf.feature_importances_
    
    # Global words (Cluster_0)
    global_word_scores = {word: overall_importance[feature_names.tolist().index(word)] 
                          for word in global_words if word in feature_names}
    top_global_words = sorted(global_word_scores.items(), key=lambda x: x[1], reverse=True)[:n_words]
    
    # Cluster-specific words
    pool = mp.Pool(processes=mp.cpu_count())
    cluster_args = [(df[df['cluster'] == i]['processed_content'], feature_names, global_words, overall_importance) 
                    for i in range(n_clusters)]
    cluster_word_scores = pool.map(get_cluster_word_scores, cluster_args)
    pool.close()
    pool.join()
    
    top_words = [top_global_words] + [sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_words] 
                                      for scores in cluster_word_scores]
    
    return top_words

def process_data(file_path, n_clusters=10, top_n=50, global_word_threshold=0.5):
    df = read_csv_robust(file_path)
    
    # Parallelize preprocessing
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    chunks = np.array_split(df['content'], num_processes)
    processed_chunks = pool.map(process_chunk, chunks)
    pool.close()
    pool.join()
    
    df['processed_content'] = pd.concat(processed_chunks).reset_index(drop=True)
    
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
    
    top_predictive_words = get_top_predictive_words(vectorizer, clf, df, n_clusters, top_n, global_words)
    
    return df, top_predictive_words, global_words

def save_outputs(df, top_predictive_words, global_words, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    df[['document_id', 'cluster']].to_csv(f"{output_dir}/document_to_cluster_map.csv", index=False)
    
    cluster_summary = []
    for i, cluster_words in enumerate(top_predictive_words):
        cluster_name = f'cluster_{i}'
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
