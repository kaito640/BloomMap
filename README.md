# ClusterWheel
ClusterWheel is visual exploration of Central vs Peripheral themes.

## Thought Experiment

During a pandemic, all employees of a global organisation are asked an open question - how do they feel about working in a pandemic?  We organise the survey to collect information about who people are, like their jobs, and their demographics, and collect their responses. 

If 100k responses are collected. How can we study the responses?

Ideally we'd like to produce a "balanced" study that shows the global themes everyone is worried about.
And we need to weigh that against the important local themes being raised by specific interest groups, or demographic groups. 

When you have those dimensions that tag a user - it's easy to get a regional view, or a organisational view. But what happens when this is not collected in the capture phase?
Do we give up? No - we can use data science to try and rebuild this global / local view - and present it effectively.

This is how our visualisation exploring central vs local themes came about. We present it here with demo grade data, the BBC kaggle dataset containing global news, but can easily be repurposed for any study where a global general centre, needs comparison with a set of local / specific interest groups.


## Output
The code provided will generate this diagram.
<img src="https://github.com/kaito640/ClusterWheel/blob/main/assets/ClusterWheel.svg" width="560">


# Methodology

## Params you can set in the pipeline:
# Methodology: Radial Clustering Visualization for News Articles

## Overview

This document outlines the methodological approach used to create a hierarchical visualization of news articles. The method combines topic modeling, hierarchical clustering, and specialized visualization techniques to reveal thematic structures in news content.

## Data Processing Pipeline

### 1. Text Preprocessing
```python
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])
```
- Case normalization
- Punctuation removal
- Stop word removal using NLTK
- Tokenization for feature extraction

### 2. Feature Extraction

#### TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    max_features=10000, 
    stop_words='english', 
    max_df=0.5,  # Remove terms that appear in >50% of docs
    min_df=2     # Remove terms that appear in <2 docs
)
```

Key Parameters:
- `max_features=10000`: Limits vocabulary size
- `max_df=0.5`: Removes overly common terms
- `min_df=2`: Removes rare terms

### 3. Topic Modeling

We use Non-negative Matrix Factorization (NMF) for topic modeling:
```python
nmf_model = NMF(
    n_components=N_TOPICS,
    random_state=42
)
nmf_output = nmf_model.fit_transform(tfidf_matrix)
```

#### Topic Model Validation
- Coherence scores assessment
- Manual review of top terms per topic
- Stability analysis across different random seeds

### 4. Cluster Analysis

#### Optimal Cluster Determination
```python
def find_optimal_clusters(data, max_clusters):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores.index(max(silhouette_scores)) + 2
```

#### Clustering Method
We use Agglomerative Clustering for its:
- Hierarchical structure preservation
- Stability across runs
- No assumptions about cluster shape

### 5. Theme Importance Calculation

For each cluster, themes are ranked by importance:
```python
def get_cluster_themes(cluster_docs, n_themes):
    themes = []
    for topic_idx in range(N_TOPICS):
        topic_importance = np.mean(cluster_docs[:, topic_idx])
        themes.append({
            'theme': topic_words[topic_idx],
            'importance': float(topic_importance)
        })
    return sorted(themes, key=lambda x: x[1], reverse=True)[:n_themes]
```

## Visualization Design Decisions

### 1. Structural Layout

#### Central (Global) Cluster
- Contains general interest stories
- Larger themes with broader appeal
- Light background for contrast with specialized clusters

#### Radial Clusters
- Specialized interest stories
- Volume-weighted visualization
- Color-coded for theme differentiation

### 2. Treemap Implementation

#### Voronoi Treemap Advantages
- Efficient space utilization
- Natural organic shapes
- Better accommodation of varying text lengths

#### Size Encoding
```javascript
const volumeScale = d3.scaleLinear()
    .domain([0, d3.max(localVolumes)*1.2])
    .range([0, outerRadius - middleRadius]);
```
- Linear scaling of cluster volumes
- 20% padding for visual clarity
- Excludes global cluster from scale calculation

### 3. Visual Encoding Choices

#### Color Scheme
- Categorical color palette for cluster differentiation
- Dark backgrounds for specialized clusters
- White text for readability
- Consistent color mapping across visualization elements

#### Text Size
```javascript
.attr("font-size", d => Math.min(20, d.data[1] * 1000) + "px")
```
- Dynamic scaling based on theme importance
- Upper bound to maintain readability
- Lower bound to ensure visibility

## Parameter Optimization

### Key Parameters and Their Effects

1. Topic Modeling
```python
N_TOPICS = 100  # Number of topics to extract
```
- Tested range: 50-200
- Optimal value determined by:
  * Topic coherence scores
  * Silhouette scores of resulting clusters
  * Manual review of theme quality

2. Theme Selection
```python
N_TOP_THEMES = 40  # Themes per cluster
```
- Balance between:
  * Information density
  * Visual clarity
  * Processing performance

3. Cluster Optimization
```python
MAX_CLUSTERS = 500  # Upper bound for cluster search
```
- Actual optimal clusters determined by:
  * Silhouette score maximization
  * Theme coherence within clusters
  * Visual complexity management
