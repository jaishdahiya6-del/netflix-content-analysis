# src/data_cleaning.py
# ─────────────────────────────────────────────────────────
# PURPOSE: Clean the raw Netflix data so it's ready for analysis
# Raw data always has problems - missing values, wrong types, etc.
# This file fixes all of that.
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

def clean_netflix_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all cleaning steps on the raw Netflix DataFrame.
    
    Steps:
        1. Show raw data summary (what problems exist)
        2. Handle missing values
        3. Fix data types
        4. Remove duplicates
        5. Create new useful columns
    
    Returns:
        pd.DataFrame: Clean, ready-to-analyze DataFrame
    """
    
    print("\n" + "=" * 55)
    print("🧹 STARTING DATA CLEANING")
    print("=" * 55)
    
    # Make a copy so we never modify the original data
    # Always work on a copy - good data science practice!
    df = df.copy()
    
    # ─────────────────────────────────────────────
    # STEP 7A: Show missing values BEFORE cleaning
    # ─────────────────────────────────────────────
    print("\n📌 Missing Values BEFORE Cleaning:")
    print("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    # Only show columns that actually have missing values
    print(missing_report[missing_report['Missing Count'] > 0])
    
    # ─────────────────────────────────────────────
    # STEP 7B: Handle missing values
    # ─────────────────────────────────────────────
    print("\n🔧 Handling Missing Values...")
    
    # 'director' - many movies don't have director listed
    # Fill with 'Unknown' so we can still count/group
    df['director'] = df['director'].fillna('Unknown')
    
    # 'cast' - some titles have no cast info
    df['cast'] = df['cast'].fillna('Unknown')
    
    # 'country' - some titles have no country listed
    df['country'] = df['country'].fillna('Unknown')
    
    # 'rating' - content rating like TV-MA, PG-13, etc.
    # Fill with the most common rating (mode)
    most_common_rating = df['rating'].mode()[0]
    df['rating'] = df['rating'].fillna(most_common_rating)
    
    # 'date_added' - when Netflix added the title
    # Fill with 'Unknown' - we'll handle this as a string first
    df['date_added'] = df['date_added'].fillna('Unknown')
    
    # ─────────────────────────────────────────────
    # STEP 7C: Fix data types
    # ─────────────────────────────────────────────
    print("🔧 Fixing Data Types...")
    
    # Convert 'date_added' to datetime type
    # errors='coerce' means: if it can't parse, put NaT (not a time) instead of crashing
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    
    # Extract useful time features from date_added
    # This creates new columns we'll use in analysis
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['month_name_added'] = df['date_added'].dt.strftime('%B')  # e.g. "January"
    
    # Convert 'release_year' to integer (sometimes loaded as float)
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').astype('Int64')
    
    # ─────────────────────────────────────────────
    # STEP 7D: Clean text columns
    # ─────────────────────────────────────────────
    print("🔧 Cleaning Text Columns...")
    
    # Strip extra whitespace from string columns
    string_cols = ['type', 'title', 'director', 'country', 'rating', 'listed_in', 'description']
    for col in string_cols:
        df[col] = df[col].str.strip()
    
    # ─────────────────────────────────────────────
    # STEP 7E: Create new useful columns
    # ─────────────────────────────────────────────
    print("🔧 Engineering New Features...")
    
    # For Movies: extract duration as integer (e.g., "90 min" → 90)
    # For TV Shows: extract number of seasons (e.g., "2 Seasons" → 2)
    df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(float)
    
    # Create a 'primary_genre' column — take just the first genre listed
    # 'listed_in' looks like: "Dramas, International Movies, Thrillers"
    df['primary_genre'] = df['listed_in'].str.split(',').str[0].str.strip()
    
    # Create 'primary_country' — take just the first country listed
    df['primary_country'] = df['country'].str.split(',').str[0].str.strip()
    
    # ─────────────────────────────────────────────
    # STEP 7F: Remove duplicates
    # ─────────────────────────────────────────────
    print("🔧 Removing Duplicates...")
    before_count = len(df)
    df = df.drop_duplicates(subset=['title', 'type', 'release_year'])
    after_count = len(df)
    removed = before_count - after_count
    print(f"   Removed {removed} duplicate rows")
    
    # ─────────────────────────────────────────────
    # STEP 7G: Reset index after all cleaning
    # ─────────────────────────────────────────────
    df = df.reset_index(drop=True)
    
    # ─────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────
    print("\n📌 Missing Values AFTER Cleaning:")
    print("-" * 40)
    missing_after = df.isnull().sum()
    remaining = missing_after[missing_after > 0]
    if len(remaining) == 0:
        print("   ✅ No missing values remaining (except date-derived columns)")
    else:
        print(remaining)
    
    print("\n" + "=" * 55)
    print("✅ DATA CLEANING COMPLETE!")
    print(f"   Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 55)
    
    return df


if __name__ == "__main__":
    from data_loader import load_netflix_data
    raw_df = load_netflix_data()
    clean_df = clean_netflix_data(raw_df)
    print("\n📌 New columns created:")
    new_cols = ['year_added', 'month_added', 'duration_int', 'primary_genre', 'primary_country']
    print(clean_df[new_cols].head())