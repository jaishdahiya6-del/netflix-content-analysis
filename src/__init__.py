import pandas as pd

def clean_netflix_data(file_path):
    df = pd.read_csv(file_path)
    
    # Handling missing values
    df['country'] = df['country'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('No Data')
    df['director'] = df['director'].fillna('No Data')
    
    # Drop rows with missing date_added or rating (crucial for analysis)
    df.dropna(subset=['date_added', 'rating'], inplace=True)
    
    # Standardize date format
    df['date_added'] = pd.to_numeric(pd.to_datetime(df['date_added'].str.strip(), errors='coerce').dt.year)
    
    print("✅ Data Cleaning Complete!")
    return df

if __name__ == "__main__":
    # Change 'netflix_titles.csv' to your actual file name
    try:
        data = clean_netflix_data('netflix_titles.csv')
        data.to_csv('cleaned_netflix_data.csv', index=False)
    except FileNotFoundError:
        print("❌ Error: netflix_titles.csv not found.")
