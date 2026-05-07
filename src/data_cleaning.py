import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create an images folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

def save_trend_chart(df):
    plt.figure(figsize=(12, 6))
    df_trend = df[df['release_year'] >= 2010]
    trend_data = df_trend.groupby(['release_year', 'type']).size().reset_index(name='Total')
    
    sns.lineplot(data=trend_data, x='release_year', y='Total', hue='type', palette=['#E50914', '#564d4d'])
    plt.title('Netflix Content Trends')
    plt.savefig('images/content_trends.png')
    plt.close()
    print("📈 Trend chart saved to images/folder")

# Add more functions for other charts here...
