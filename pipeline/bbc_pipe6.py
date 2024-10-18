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
import csv
import re
import os

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
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if not all(char in string.punctuation or char.isdigit() for char in token)]
    tokens = [token for token in tokens if re.match(r"^[a-z]+'?[a-z]*$", token)]
    return ' '.join(tokens)

def get_top_predictive_words(vectorizer, classifier, n_clusters, n_words):
    feature_names = vectorizer.get_feature_names_out()
    top_words = []
    for i in range(n_clusters):
        word_importance = classifier.feature_importances_
        top_word_indices = word_importance.argsort()[-n_words:][::-1]
        top_words.append([(feature_names[j], word_importance[j]) for j in top_word_indices])
    return top_words

def process_data(file_path, n_clusters=12, top_n=50, bottom_x=100):
    df = read_csv_robust(file_path)
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    all_words = ' '.join(df['processed_content']).split()
    word_freq = Counter(all_words)
    most_common_words = [word for word, _ in word_freq.most_common(bottom_x)]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_content'])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Random Forest for prediction and feature importance
    X_train, X_test, y_train, y_test = train_test_split(X, df['cluster'], test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get top predictive words
    top_predictive_words = get_top_predictive_words(vectorizer, clf, n_clusters, top_n)
    
    return df, word_freq, top_predictive_words, most_common_words

def save_outputs(df, word_freq, top_predictive_words, most_common_words, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    df[['document_id', 'cluster']].to_csv(f"{output_dir}/document_to_cluster_map.csv", index=False)
    
    cluster_summary = []
    for i, cluster_words in enumerate(top_predictive_words):
        for word, score in cluster_words:
            cluster_summary.append({
                'cluster': f'cluster_{i}',
                'word': word,
                'frequency': word_freq[word],
                'predictive_score': score
            })
    
    pd.DataFrame(cluster_summary).to_csv(f"{output_dir}/cluster_summary.csv", index=False)
    
    global_summary = [{'word': word, 'frequency': word_freq[word]} for word in most_common_words]
    pd.DataFrame(global_summary).to_csv(f"{output_dir}/global_summary.csv", index=False)

if __name__ == "__main__":
    file_path = 'data/bbc-news-data.csv'
    output_dir = './'
    
    df, word_freq, top_predictive_words, most_common_words = process_data(file_path)
    save_outputs(df, word_freq, top_predictive_words, most_common_words, output_dir)
    
    print("Results saved to output directory")
