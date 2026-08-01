import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images and data folders if they don't exist
if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('data'):
    os.makedirs('data')

def clean_netflix_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the Netflix dataset.

    Handles:
    - Missing values in critical columns
    - Duplicates
    - Extracting 'primary_country'
    - Extracting 'duration_int'
    - Extracting 'year_added'
    """
    df_clean = df.copy()

    # 1. Drop duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['title', 'type', 'release_year']).copy()
    dropped_duplicates = initial_rows - len(df_clean)
    if dropped_duplicates > 0:
        print(f"🧹 Removed {dropped_duplicates} duplicate records.")

    # 2. Fill standard text missing values
    df_clean['director'] = df_clean['director'].fillna('Unknown')
    df_clean['cast'] = df_clean['cast'].fillna('Unknown')
    df_clean['country'] = df_clean['country'].fillna('Unknown')
    df_clean['rating'] = df_clean['rating'].fillna('Unknown')

    # 3. Extract primary country (first country listed)
    df_clean['primary_country'] = df_clean['country'].apply(
        lambda x: x.split(',')[0].strip() if x != 'Unknown' else 'Unknown'
    )

    # 4. Extract year_added from date_added
    df_clean['date_added_clean'] = pd.to_datetime(df_clean['date_added'].str.strip(), errors='coerce')
    df_clean['year_added'] = df_clean['date_added_clean'].dt.year

    # Fill year_added missing values with the median year added
    median_year = df_clean['year_added'].median()
    if pd.isna(median_year):
        median_year = 2019.0 # Sensible fallback
    df_clean['year_added'] = df_clean['year_added'].fillna(median_year).astype(int)

    # 5. Extract duration_int as integer
    def parse_duration(val):
        if pd.isna(val):
            return np.nan
        # Format is usually 'X min' or 'Y Season(s)'
        parts = str(val).strip().split(' ')
        try:
            return float(parts[0])
        except (ValueError, IndexError):
            return np.nan

    df_clean['duration_int'] = df_clean['duration'].apply(parse_duration)
    median_duration = df_clean['duration_int'].median()
    if pd.isna(median_duration):
        median_duration = 90.0 # Sensible fallback
    df_clean['duration_int'] = df_clean['duration_int'].fillna(median_duration).astype(int)

    print(f"✅ Data cleaning complete. Cleaned shape: {df_clean.shape}")
    return df_clean

def save_trend_chart(df):
    plt.figure(figsize=(12, 6))
    df_trend = df[df['release_year'] >= 2010]
    trend_data = df_trend.groupby(['release_year', 'type']).size().reset_index(name='Total')
    
    sns.lineplot(data=trend_data, x='release_year', y='Total', hue='type', palette=['#E50914', '#564d4d'])
    plt.title('Netflix Content Trends')
    plt.savefig('images/content_trends.png')
    plt.close()
    print("📈 Trend chart saved to images/content_trends.png")

if __name__ == "__main__":
    from data_loader import load_netflix_data
    df = load_netflix_data()
    df_clean = clean_netflix_data(df)
    # Save the cleaned dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base_dir, "data", "cleaned_netflix_data.csv")
    df_clean.to_csv(cleaned_path, index=False)
    print(f"💾 Saved cleaned data to: {cleaned_path}")
    save_trend_chart(df_clean)
