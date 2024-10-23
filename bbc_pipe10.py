import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import json

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Parameters
N_TOP_THEMES = 20  # Number of top themes to display per cluster
N_TOPICS = 20
MAX_CLUSTERS = 100 # Maximum number of clusters to consider

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

# Function to find optimal number of clusters
def find_optimal_clusters(data, max_clusters):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores.index(max(silhouette_scores)) + 2

# Function to get top themes for a cluster
def get_top_themes(cluster_docs, n_themes):
    theme_importance = []
    for topic in range(N_TOPICS):
        importance = np.mean(cluster_docs[:, topic])
        theme_importance.append({
            'theme': topic_words[topic],  # This is now a list of words
            'importance': float(importance)  # Ensure importance is JSON serializable
        })
    return sorted(theme_importance, key=lambda x: x['importance'], reverse=True)[:n_themes]

def get_cluster_themes(cluster_docs, n_themes):
    themes = []
    for topic_idx in range(N_TOPICS):
        topic_importance = np.mean(cluster_docs[:, topic_idx])
        for word in topic_words[topic_idx]:
            themes.append((word, float(topic_importance)))
    return sorted(themes, key=lambda x: x[1], reverse=True)[:n_themes]


# Read the data
data = pd.read_csv('data/bbc-news-data.csv', sep='\t')

# Preprocess the content
data['processed_content'] = data['content'].apply(preprocess_text)

# Create TF-IDF representation
vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', max_df=0.5, min_df=2)
tfidf_matrix = vectorizer.fit_transform(data['processed_content'])

# Perform topic modeling using NMF
nmf_model = NMF(n_components=N_TOPICS, random_state=42, max_iter=200)
nmf_output = nmf_model.fit_transform(tfidf_matrix)

# Get the top words for each topic
feature_names = vectorizer.get_feature_names_out()
topic_words = []
for topic_idx, topic in enumerate(nmf_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]  # Get top 5 words
    topic_words.append(top_words)

# Assign the most probable topic to each document
data['theme'] = [' '.join(topic_words[doc.argmax()]) for doc in nmf_output]

# Find optimal number of clusters
n_clusters = find_optimal_clusters(nmf_output, MAX_CLUSTERS)
print(f"Optimal number of clusters: {n_clusters}")

# Perform Agglomerative Clustering
clusterer = AgglomerativeClustering(n_clusters=n_clusters)
data['cluster'] = clusterer.fit_predict(nmf_output)

# Identify global cluster (assumed to be the largest cluster)
global_cluster = data['cluster'].mode().iloc[0]


# Process clusters and prepare data for JSON output
output_data = {}
cluster_volume_data = {}

# Process global cluster
global_docs = nmf_output[data['cluster'] == global_cluster]
global_themes = get_cluster_themes(global_docs, N_TOP_THEMES)
global_volume = (data['cluster'] == global_cluster).sum()
output_data['global'] = global_themes
cluster_volume_data['global'] = int(global_volume)


# Process other clusters
for cluster in range(n_clusters):
    if cluster == global_cluster:
        continue
    cluster_docs = nmf_output[data['cluster'] == cluster]
    cluster_themes = get_cluster_themes(cluster_docs, N_TOP_THEMES)
    cluster_volume = (data['cluster'] == cluster).sum()
    cluster_name = f'c{cluster + 1}'
    output_data[cluster_name] = cluster_themes
    # Calculate relative volume
    cluster_volume_data[cluster_name] = int(cluster_volume)

# Save processed themes as JSON
with open('proc_themes.json', 'w') as f:
    json.dump(output_data, f)

# Save cluster volumes as JSON
with open('cluster_volumes.json', 'w') as f:
    json.dump(cluster_volume_data, f)

print("Analysis complete. Results written to proc_themes.json and cluster_volumes.json")

