import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import string
import csv
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def read_csv_robust(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = []
        for row in reader:
            if len(row) >= 4:  # Ensure we have at least the required columns
                data.append({
                    'category': row[0],
                    'filename': row[1],
                    'title': row[2],
                    'content': ' '.join(row[3:])  # Join all remaining fields as content
                })
    return pd.DataFrame(data)

def preprocess_text(text):
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove tokens that are purely punctuation or numbers
    tokens = [token for token in tokens if not all(char in string.punctuation or char.isdigit() for char in token)]
    
    # Remove any tokens that are just unicode characters or other non-word characters
    # This regex allows for apostrophes within words (e.g., "don't")
    tokens = [token for token in tokens if re.match(r"^[a-z]+'?[a-z]*$", token)]
    
    return ' '.join(tokens)

def get_top_words(vectorizer, classifier, n_words):
    feature_names = vectorizer.get_feature_names_out()
    top_words = []
    for idx, centroid in enumerate(classifier.cluster_centers_):
        top_word_indices = centroid.argsort()[-n_words:][::-1]
        top_words.append([feature_names[i] for i in top_word_indices])
    return top_words

def process_data(file_path, n_clusters=10, m_subclusters=3, top_n=5, bottom_x=10):
    # Read the CSV file using our robust method
    df = read_csv_robust(file_path)
    
    # Preprocess the content
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    # Global analysis
    all_words = ' '.join(df['processed_content']).split()
    word_freq = Counter(all_words)
    most_common_words = [word for word, _ in word_freq.most_common(bottom_x)]
    
    # Vectorization and clustering
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_content'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    top_cluster_words = get_top_words(vectorizer, kmeans, top_n)
    
    # Subclustering
    subcluster_words = []
    for i in range(n_clusters):
        cluster_docs = X[kmeans.labels_ == i]
        n_samples = cluster_docs.shape[0]
        
        # Adjust number of subclusters if necessary
        actual_subclusters = min(m_subclusters, n_samples)
        
        if actual_subclusters > 1:
            subkmeans = KMeans(n_clusters=actual_subclusters, random_state=42)
            subkmeans.fit(cluster_docs)
            subcluster_words.append(get_top_words(vectorizer, subkmeans, top_n))
        else:
            # If we can't subcluster, just use the main cluster's words
            subcluster_words.append([top_cluster_words[i]])
    
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
if __name__ == "__main__":
    file_path = 'data/bbc-news-data.csv'
    result_df = process_data(file_path)
    print(result_df)

    # Optionally, save to CSV
    result_df.to_csv('output/clustering_results.csv', index=False)
    print("Results saved to output/clustering_results.csv")

