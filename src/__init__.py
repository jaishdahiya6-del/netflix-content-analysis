import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def handle_missing_values(self):
        """Impute numerical with median and categorical with mode."""
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self.df

    def remove_outliers(self, column):
        """Detect and remove outliers using the IQR Method."""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self.df

    def get_clean_data(self):
        return self.df

if __name__ == "__main__":
    # Test with your Amazon or Netflix data
    df = pd.read_csv('amazon_sales.csv') # or your target file
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values()
    
    # Example: Removing outliers from 'Sales' or 'Price'
    if 'Sales' in df.columns:
        cleaner.remove_outliers('Sales')
        
    clean_df = cleaner.get_clean_data()
    clean_df.to_csv('final_cleaned_data.csv', index=False)
    print("✅ Professional cleaning and outlier removal complete!")
