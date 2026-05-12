import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def perform_eda(df, project_name="Analysis"):
    """
    Generates high-end EDA visualizations for a professional portfolio.
    """
    # 1. Correlation Heatmap (Only for numerical columns)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Feature Correlation Heatmap: {project_name}', fontsize=15)
    plt.savefig('images/correlation_heatmap.png')
    plt.close()

    # 2. Distribution of a Key Metric (e.g., Release Year or Sales)
    plt.figure(figsize=(10, 6))
    # Automatically pick the first numerical column if not specified
    target_col = numeric_df.columns[0] 
    sns.histplot(df[target_col], kde=True, color='purple')
    plt.title(f'Distribution of {target_col}', fontsize=15)
    plt.savefig(f'images/{target_col}_distribution.png')
    plt.close()

    print(f"✅ EDA complete! Visuals saved to the images/ folder.")

if __name__ == "__main__":
    # Test with your dataset
    try:
        data = pd.read_csv('cleaned_netflix_data.csv') # or amazon_sales.csv
        perform_eda(data, project_name="Netflix Analysis")
    except FileNotFoundError:
        print("⚠️ File not found. Please check your path.")
