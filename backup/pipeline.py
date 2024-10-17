import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def read_documents(directory):
    documents = []
    for filename in glob.glob(os.path.join(directory, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

def get_top_words(vectorizer, classifier, n_words):
    feature_names = vectorizer.get_feature_names_out()
    top_words = []
    for idx, centroid in enumerate(classifier.cluster_centers_):
        top_word_indices = centroid.argsort()[-n_words:][::-1]
        top_words.append([feature_names[i] for i in top_word_indices])
    return top_words

def process_documents(directory, n_clusters=10, m_subclusters=3, top_n=5, bottom_x=10):
    documents = read_documents(directory)
    processed_docs = [preprocess_text(doc) for doc in documents]

    # Global analysis
    all_words = ' '.join(processed_docs).split()
    word_freq = Counter(all_words)
    most_common_words = [word for word, _ in word_freq.most_common(bottom_x)]

    # Clustering
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(processed_docs)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    top_cluster_words = get_top_words(vectorizer, kmeans, top_n)

    # Subclustering
    subcluster_words = []
    for i in range(n_clusters):
        cluster_docs = X[kmeans.labels_ == i]
        subkmeans = KMeans(n_clusters=m_subclusters, random_state=42)
        subkmeans.fit(cluster_docs)
        subcluster_words.append(get_top_words(vectorizer, subkmeans, top_n))

    # Prepare output data
    output_data = []

    # Add global cluster
    for word in most_common_words:
        output_data.append({
            'cluster': 'all',
            'subcluster': 'corpus',
            'word': word,
            'frequency': word_freq[word]
        })

    # Add cluster and subcluster data
    for i, cluster_words in enumerate(top_cluster_words):
        for word in cluster_words:
            output_data.append({
                'cluster': f'cluster_{i}',
                'subcluster': 'main',
                'word': word,
                'frequency': word_freq[word]
            })
        for j, subcluster_word_list in enumerate(subcluster_words[i]):
            for word in subcluster_word_list:
                output_data.append({
                    'cluster': f'cluster_{i}',
                    'subcluster': f'subcluster_{j}',
                    'word': word,
                    'frequency': word_freq[word]
                })

    return pd.DataFrame(output_data)

# Example usage
directory = 'path/to/your/documents'
result_df = process_documents(directory)
print(result_df)

# Optionally, save to CSV
result_df.to_csv('clustering_results.csv', index=False)
