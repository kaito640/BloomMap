# BloomMap
BloomMap is visual exploration of Central vs Peripheral themes in large text collections.


## What it looks like
The code provided will generate this diagram from the bbc news dataset found on kaggle.

<img src="https://github.com/kaito640/ClusterWheel/blob/main/assets/ClusterWheel.svg" width="800">

## About BloomMap Visualisations
A sophisticated D3.js visualization that displays hierarchical topic clusters in a radial layout, combining treemaps, volume indicators, and hierarchical relationships in an intuitive flower-like pattern.

## Overview

This visualization consists of three main components:
1. A central topic network displayed as a hexagonal treemap
2. Radial volume indicators showing relative cluster sizes
3. Peripheral topic clusters arranged in a circular pattern, each containing their own treemap

## Features

- Interactive visualization of hierarchical topic clusters
- Volume-based indicators showing relative cluster sizes
- Configurable parameters for fine-tuning the visualization:
  - Font sizes for global and cluster text
  - Word counts for central and peripheral clusters
  - Layout settings (padding, volume scale)
  - Radius settings for different components
  - Rotation and positioning controls

## Dependencies

- D3.js v6
- D3 Weighted Voronoi
- D3 Voronoi Map
- D3 Voronoi Treemap

```html
<script src="https://d3js.org/d3.v6.min.js"></script>
<script src="https://rawcdn.githack.com/Kcnarf/d3-weighted-voronoi/v1.1.3/build/d3-weighted-voronoi.js"></script>
<script src="https://rawcdn.githack.com/Kcnarf/d3-voronoi-map/v2.1.1/build/d3-voronoi-map.js"></script>
<script src="https://rawcdn.githack.com/Kcnarf/d3-voronoi-treemap/v1.1.2/build/d3-voronoi-treemap.js"></script>
```

## Data Format

The visualization expects two JSON files:

### proc_themes.json
Contains the hierarchical topic data:
```json
{
  "global": [[topic, weight], ...],
  "cluster1": [[topic, weight], ...],
  "cluster2": [[topic, weight], ...],
  ...
}
```

### cluster_volumes.json
Contains volume data for each cluster:
```json
{
  "global": value,
  "cluster1": value,
  "cluster2": value,
  ...
}
```

## Usage

1. Include the required dependencies
2. Set up an SVG container:
```html
<svg id="visualization" width="1400" height="1400"></svg>
```
3. Include the visualization script
4. Initialize with your data:
```javascript
Promise.all([
  d3.json("proc_themes.json"),
  d3.json("cluster_volumes.json")
]).then(([themeData, volumeData]) => {
  // Visualization will automatically render
});
```

## Configuration

The visualization can be customized through the UI controls or by modifying the `VIZ_CONFIG` object:

```javascript
const VIZ_CONFIG = {
  // Dimensions
  width: 1400,
  height: 1400,
  
  // Font sizes
  fonts: {...},
  
  // Word counts
  wordCounts: {...},
  
  // Visualization settings
  visualization: {...},

  // Treemap settings
  treemap: {...}
};
```

## Controls

- **Font Sizes**: Adjust text size ranges for global and cluster labels
- **Word Counts**: Control the number of words shown in central and peripheral clusters
- **Layout Settings**: Modify padding and volume scaling
- **Radius Settings**: Fine-tune the radial layout
- **Treemap Settings**: Adjust rotation and positioning of peripheral clusters
- **Rotation**: Control the overall rotation of the visualization

## Features

- SVG Export: Download the visualization as an SVG file
- Interactive Updates: Real-time visualization updates when parameters change
- Responsive Layout: Automatically adjusts to container size
- Consistent Text Orientation: Maintains readable text alignment
- Color-coded Clusters: Visual distinction between different topic groups

## Limitations

- Designed for datasets with a single central topic and multiple peripheral clusters
- Best suited for 15-25 peripheral clusters
- Text size automatically scales but may become unreadable with very small clusters
- Requires modern browser support for D3.js and SVG rendering

# BloomMap Data Pipeline


## Clustering Pipeline Methodology
### Workflow

We provide you the data pipeline code, which will generate the exact files you need to drive the visualisation. You just need to format your responses into the input data format. From there, the data science is detailed below. Once you have rendered the D3.js image, you can download it using the button and post process and beautify the viz in Inkscape.

### Data Science Method

We create a hierarchical visualization of news articles using a method that combines topic modeling, hierarchical clustering, and specialized visualization techniques to reveal thematic structures in news content.

### Data Processing Pipeline

#### 1. Text Preprocessing
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

#### 2. Feature Extraction

##### TF-IDF Vectorization
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

#### 3. Topic Modeling

We use Non-negative Matrix Factorization (NMF) for topic modeling:
```python
nmf_model = NMF(
    n_components=N_TOPICS,
    random_state=42
)
nmf_output = nmf_model.fit_transform(tfidf_matrix)
```

##### Topic Model Validation
- Coherence scores assessment
- Manual review of top terms per topic
- Stability analysis across different random seeds

#### 4. Cluster Analysis

##### Optimal Cluster Determination
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

##### Clustering Method
We use Agglomerative Clustering for its:
- Hierarchical structure preservation
- Stability across runs
- No assumptions about cluster shape

#### 5. Theme Importance Calculation

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


``
