import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

def run_stats_report(df, column_name):
    """
    Calculates the statistical distribution of a specific column.
    """
    data = df[column_name].dropna()

    # 1. Calculate Core Stats
    stats = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std Dev': np.std(data),
        'Skewness': skew(data),
        'Kurtosis': kurtosis(data)
    }

    # 2. Visualize Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(data, kde=True, color='teal')
    plt.axvline(stats['Mean'], color='red', linestyle='--', label='Mean')
    plt.axvline(stats['Median'], color='yellow', linestyle='-', label='Median')
    plt.title(f'Statistical Distribution of {column_name}')
    plt.legend()
    plt.savefig(f'images/{column_name}_stats.png')
    
    return stats

if __name__ == "__main__":
    # Test with Stock data or Sales data
    try:
        df = pd.read_csv('amazon_sales.csv') # or big_tech_stocks.csv
        report = run_stats_report(df, 'Sales')
        print("📊 Statistical Insights:")
        for key, value in report.items():
            print(f"{key}: {value:.2f}")
    except FileNotFoundError:
        print("⚠️ File not found. Ensure your dataset is in the directory.")
