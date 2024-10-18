import pandas as pd
import numpy as np
from collections import defaultdict

def process_synthetic_data(input_file, output_file):
    # Read the synthetic data
    df = pd.read_csv(input_file)
    
    # Initialize data structures
    global_themes = defaultdict(list)
    local_themes = defaultdict(lambda: defaultdict(list))
    
    # Process each row
    for _, row in df.iterrows():
        theme = row['theme']
        importance = row['importance']
        
        if row['is_global']:
            global_themes[theme].append(importance)
        else:
            local_themes[row['continent']][theme].append(importance)
    
    # Prepare output data
    output_data = []
    
    # Process global themes
    for theme, importances in global_themes.items():
        avg_importance = np.mean(importances)
        output_data.append({
            'theme': theme,
            'cluster': 'global',
            'importance': avg_importance
        })
    
    # Process local themes
    for continent, themes in local_themes.items():
        for theme, importances in themes.items():
            avg_importance = np.mean(importances)
            output_data.append({
                'theme': theme,
                'cluster': continent,
                'importance': avg_importance
            })
    
    # Create and save the output DataFrame
    output_df = pd.DataFrame(output_data)
    output_df.sort_values(['cluster', 'importance'], ascending=[True, False], inplace=True)
    output_df.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")
    
    # Print some statistics
    print("\nData Summary:")
    print(f"Total themes: {len(output_df)}")
    print(f"Global themes: {len(output_df[output_df['cluster'] == 'global'])}")
    print("Themes per continent:")
    print(output_df['cluster'].value_counts())
    print("\nTop 5 global themes:")
    print(output_df[output_df['cluster'] == 'global'].head())

if __name__ == "__main__":
    input_file = 'synthetic_news_data.csv'
    output_file = 'processed_themes.csv'
    process_synthetic_data(input_file, output_file)
