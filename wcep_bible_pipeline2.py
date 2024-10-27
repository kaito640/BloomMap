import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from tqdm import tqdm
import json
import logging
from pathlib import Path
import re
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_word(word):
    """Clean and validate a word for theme extraction."""
    cleaned = re.sub(r'[^a-zA-Z]', '', word.lower())
    if len(cleaned) < 3 or cleaned in {'one', 'two', 'three', 'four', 'five', 'six', 'seven'}:
        return None
    return cleaned

def count_words(text):
    """Count words in text, handling biblical text specifics."""
    # Remove verse numbers (e.g., "1:1")
    text = re.sub(r'\d+:\d+', '', text)
    # Split on whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)

def validate_and_prepare_data(csv_path):
    """Validate and prepare the input data with word count metrics."""
    try:
        df = pd.read_csv(csv_path)
        
        required_columns = ['category', 'headline', 'content']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Clean data and calculate word counts
        df['content'] = df['content'].fillna('')
        df['content'] = df['content'].astype(str)
        df['word_count'] = df['content'].apply(count_words)
        
        # Remove empty documents
        df = df[df['word_count'] > 0]
        
        # Calculate statistics
        total_words = df['word_count'].sum()
        words_by_category = df.groupby('category')['word_count'].sum()
        
        logging.info("\nDataset Statistics:")
        logging.info(f"Total documents: {len(df)}")
        logging.info(f"Total words: {total_words:,}")
        logging.info(f"Number of categories (books): {df['category'].nunique()}")
        logging.info(f"Average words per document: {df['word_count'].mean():.1f}")
        
        logging.info("\nWord count by book:")
        for book, word_count in words_by_category.sort_values(ascending=False).items():
            logging.info(f"{book}: {word_count:,} words ({(word_count/total_words)*100:.1f}%)")
        
        return df
    
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def create_vectorizer(df):
    """Create TF-IDF vectorizer with appropriate parameters based on dataset size."""
    n_documents = len(df)
    
    if n_documents < 100:
        min_df = 1
        max_df = 0.95
    elif n_documents < 1000:
        min_df = 2
        max_df = 0.9
    else:
        min_df = 3
        max_df = 0.85
    
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words='english',
        max_df=max_df,
        min_df=min_df,
        token_pattern=r'(?u)\b[A-Za-z]+\b'
    )
    
    logging.info(f"\nVectorizer parameters:")
    logging.info(f"min_df: {min_df}")
    logging.info(f"max_df: {max_df}")
    
    return vectorizer

def extract_themes(docs, vectorizer, n_topics=10, n_top_words=50):
    """Extract themes using NMF."""
    tfidf_matrix = vectorizer.fit_transform(docs)
    logging.info(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    nmf = NMF(n_components=n_topics, random_state=42)
    doc_topic_matrix = nmf.fit_transform(tfidf_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    return doc_topic_matrix, nmf, feature_names

def calculate_book_metrics(df, category):
    """Calculate detailed metrics for a book."""
    book_df = df[df['category'] == category]
    
    total_words = book_df['word_count'].sum()
    avg_words_per_verse = book_df['word_count'].mean()
    num_verses = len(book_df)
    
    # Get word frequency distribution
    all_words = ' '.join(book_df['content']).split()
    word_freq = Counter(all_words)
    
    return {
        'word_count': int(total_words),
        'verse_count': int(num_verses),
        'avg_words_per_verse': float(avg_words_per_verse),
        'unique_words': len(word_freq),
        'most_common_words': dict(word_freq.most_common(10))
    }

def main(csv_path):
    logging.info("Starting Bible text processing pipeline...")
    
    # Validate and prepare data
    df = validate_and_prepare_data(csv_path)
    
    # Create vectorizer
    vectorizer = create_vectorizer(df)
    
    logging.info("\nStep 1: Creating global document vectors...")
    texts = df['content'].tolist()
    try:
        doc_topic_matrix, nmf_model, feature_names = extract_themes(texts, vectorizer)
        logging.info("Global themes extracted successfully")
    except Exception as e:
        logging.error(f"Error extracting global themes: {e}")
        raise
    
    logging.info("\nStep 2: Processing individual books...")
    categories = df['category'].unique()
    
    # Initialize output dictionaries
    output_data = {}
    volume_data = {}
    
    # Calculate global metrics first
    all_word_count = df['word_count'].sum()
    
    # Process global themes first
    global_matrix = vectorizer.transform(texts)
    global_topic_dist = np.mean(global_matrix.toarray(), axis=0)
    global_themes = []
    
    for topic_idx in range(nmf_model.n_components_):
        topic = nmf_model.components_[topic_idx]
        top_words_idx = topic.argsort()[-50:][::-1]
        for idx in top_words_idx:
            word = feature_names[idx]
            importance = float(global_topic_dist[idx] * topic[idx])
            global_themes.append((word, importance))
    
    # Process global themes
    global_theme_dict = {}
    for word, score in global_themes:
        cleaned_word = preprocess_word(word)
        if cleaned_word:
            global_theme_dict[cleaned_word] = max(score, global_theme_dict.get(cleaned_word, 0))
    
    sorted_global_themes = sorted(global_theme_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    
    # Store global results
    output_data['global'] = sorted_global_themes
    volume_data['global'] = {
        'volume': int(all_word_count),
        'silhouette_score': 0.0,  # Placeholder metrics required by visualization
        'calinski_harabasz_score': 0.0,
        'davies_bouldin_score': 0.0,
        'cluster_sizes': [int(df[df['category'] == cat]['word_count'].sum()) for cat in categories],
        'cluster_density': 0.0
    }
    
    # Process each book
    for category in tqdm(categories, desc="Processing books"):
        try:
            category_docs = df[df['category'] == category]['content'].tolist()
            category_matrix = vectorizer.transform(category_docs)
            
            # Extract themes
            category_topic_dist = np.mean(category_matrix.toarray(), axis=0)
            themes = []
            
            for topic_idx in range(nmf_model.n_components_):
                topic = nmf_model.components_[topic_idx]
                top_words_idx = topic.argsort()[-50:][::-1]
                for idx in top_words_idx:
                    word = feature_names[idx]
                    importance = float(category_topic_dist[idx] * topic[idx])
                    themes.append((word, importance))
            
            theme_dict = {}
            for word, score in themes:
                cleaned_word = preprocess_word(word)
                if cleaned_word:
                    theme_dict[cleaned_word] = max(score, theme_dict.get(cleaned_word, 0))
            
            sorted_themes = sorted(theme_dict.items(), key=lambda x: x[1], reverse=True)[:50]
            
            # Calculate detailed book metrics
            book_metrics = calculate_book_metrics(df, category)
            
            # Store results
            output_data[category] = sorted_themes
            volume_data[category] = {
                'volume': book_metrics['word_count'],
                'silhouette_score': volume_data['global']['silhouette_score'],
                'calinski_harabasz_score': volume_data['global']['calinski_harabasz_score'],
                'davies_bouldin_score': volume_data['global']['davies_bouldin_score'],
                'cluster_sizes': volume_data['global']['cluster_sizes'],
                'cluster_density': volume_data['global']['cluster_density']
            }
            
        except Exception as e:
            logging.error(f"Error processing book {category}: {e}")
            continue
    
    # Save results
    output_dir = Path('')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'proc_themes.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    with open(output_dir / 'cluster_volumes.json', 'w') as f:
        json.dump(volume_data, f, indent=2)
    
    logging.info("\nProcessing complete!")
    logging.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main("data/bible_data.csv")
