import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add root folder to path so we can import from src
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_loader import load_netflix_data
from src.data_cleaning import clean_netflix_data

# Create an images folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

def perform_eda(df, project_name="Netflix Analysis"):
    """
    Generates high-end EDA visualizations for a professional portfolio.
    """
    print(f"📊 Running EDA for project: {project_name}")

    # Ensure images directory exists
    os.makedirs('images', exist_ok=True)

    # 1. Correlation Heatmap (Only for numerical columns)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f'Feature Correlation Heatmap: {project_name}', fontsize=15)
        plt.savefig('images/correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📈 Saved Correlation Heatmap to images/correlation_heatmap.png")
    else:
        print("⚠️ Not enough numerical columns to build a correlation heatmap.")

    # 2. Distribution of a Key Metric (e.g., Release Year)
    plt.figure(figsize=(10, 6))
    if 'release_year' in df.columns:
        sns.histplot(df['release_year'], kde=True, color='purple', bins=30)
        plt.title('Distribution of Release Year', fontsize=15)
        plt.xlabel('Release Year')
        plt.ylabel('Count')
        plt.savefig('images/release_year_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📈 Saved Release Year Distribution to images/release_year_distribution.png")
    else:
        # Fallback to whatever first numerical column
        if not numeric_df.empty:
            target_col = numeric_df.columns[0]
            sns.histplot(df[target_col], kde=True, color='purple', bins=30)
            plt.title(f'Distribution of {target_col}', fontsize=15)
            plt.savefig(f'images/{target_col}_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📈 Saved {target_col} Distribution to images/{target_col}_distribution.png")

    print(f"✅ EDA complete! Visuals saved to the images/ folder.")

if __name__ == "__main__":
    # Load and clean dataset for testing
    try:
        df_raw = load_netflix_data()
        df_clean = clean_netflix_data(df_raw)
        perform_eda(df_clean, project_name="Netflix Analysis")
    except Exception as e:
        print(f"❌ Error during direct EDA execution: {e}")
