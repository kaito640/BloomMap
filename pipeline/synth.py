import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define continents and cities
continents = {
    'North America': ['New York', 'Los Angeles', 'Chicago', 'Toronto', 'Mexico City'],
    'South America': ['São Paulo', 'Buenos Aires', 'Lima', 'Bogotá', 'Santiago'],
    'Europe': ['London', 'Paris', 'Berlin', 'Rome', 'Madrid'],
    'Africa': ['Cairo', 'Lagos', 'Johannesburg', 'Nairobi', 'Casablanca'],
    'Asia': ['Tokyo', 'Shanghai', 'Mumbai', 'Seoul', 'Bangkok'],
    'Oceania': ['Sydney', 'Melbourne', 'Auckland', 'Brisbane', 'Perth']
}

# Define global and local themes
global_themes = [
    'Climate Change', 'Global Economy', 'Technology Advancements', 'International Relations',
    'Health and Pandemics', 'Space Exploration', 'Cybersecurity', 'Renewable Energy',
    'Artificial Intelligence', 'Global Trade'
]

local_themes = {
    'North America': ['US Politics', 'Silicon Valley', 'Hollywood', 'Wall Street', 'NFL'],
    'South America': ['Amazon Rainforest', 'Soccer', 'Latin Music', 'Carnival', 'Andes Mountains'],
    'Europe': ['Brexit', 'European Union', 'Champions League', 'Art Exhibitions', 'Royal Families'],
    'Africa': ['Sahara Desert', 'Wildlife Conservation', 'Nile River', 'African Union', 'Safaris'],
    'Asia': ['K-pop', 'Bollywood', 'Great Wall', 'Silk Road', 'Sumo Wrestling'],
    'Oceania': ['Great Barrier Reef', 'Outback', 'Indigenous Rights', 'Rugby', 'Kangaroos']
}

# Generate synthetic news data
def generate_news_data(num_articles=1000):
    data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for _ in range(num_articles):
        continent = random.choice(list(continents.keys()))
        city = random.choice(continents[continent])
        
        if random.random() < 0.3:  # 30% chance of global theme
            theme = random.choice(global_themes)
            is_global = True
        else:
            theme = random.choice(local_themes[continent])
            is_global = False
        
        date = start_date + timedelta(days=random.randint(0, 30))
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'continent': continent,
            'city': city,
            'theme': theme,
            'is_global': is_global,
            'importance': round(random.uniform(0.1, 1.0), 2)
        })
    
    return pd.DataFrame(data)

# Generate the dataset
news_df = generate_news_data(5000)

# Save to CSV
news_df.to_csv('synthetic_news_data.csv', index=False)
print("Synthetic news dataset saved to 'synthetic_news_data.csv'")

# Display some statistics
print("\nDataset Statistics:")
print(f"Total articles: {len(news_df)}")
print(f"Global themes: {news_df['is_global'].sum()}")
print(f"Local themes: {len(news_df) - news_df['is_global'].sum()}")
print("\nArticles per continent:")
print(news_df['continent'].value_counts())
print("\nTop 10 themes:")
print(news_df['theme'].value_counts().head(10))
