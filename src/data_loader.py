# src/data_loader.py
# ─────────────────────────────────────────────────────────
# PURPOSE: Load the Netflix CSV dataset into a Pandas DataFrame
# Think of a DataFrame like an Excel spreadsheet in Python
# ─────────────────────────────────────────────────────────

import pandas as pd   # pandas helps us work with tabular data
import os             # os helps us work with file paths

def load_netflix_data(filepath: str = None) -> pd.DataFrame:
    """
    Load Netflix dataset from CSV file.
    
    Parameters:
        filepath (str): Path to the CSV file. 
                        If None, uses default path.
    
    Returns:
        pd.DataFrame: The loaded dataset as a DataFrame
    """
    
    # If no path given, use the default location
    if filepath is None:
        # Build path relative to this file's location
        # This means: go up one folder from src/, then into data/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "netflix_titles.csv")
    
    # Check if file actually exists before trying to load it
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please place netflix_titles.csv in the /data folder."
        )
    
    # Load the CSV into a DataFrame
    df = pd.read_csv(filepath)
    
    # Print basic info so we know it loaded correctly
    print("=" * 50)
    print("✅ Dataset Loaded Successfully!")
    print("=" * 50)
    print(f"📊 Total rows    : {df.shape[0]:,}")
    print(f"📋 Total columns : {df.shape[1]}")
    print(f"📁 File path     : {filepath}")
    print("=" * 50)
    print("\n📌 Column Names:")
    for col in df.columns:
        print(f"   • {col}")
    print("=" * 50)
    
    return df


# ─────────────────────────────────────────────────────────
# This block only runs when you run THIS file directly
# It won't run when imported by another file
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_netflix_data()
    print("\n📌 First 3 rows of data:")
    print(df.head(3))