import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
df_netflix = pd.read_csv('netflix_titles.csv')

# 2. Check for missing values (Day 8 habit)
print(df_netflix.isnull().sum())

# 3. Handling Nulls
df_netflix['country'] = df_netflix['country'].fillna('Unknown')
df_netflix['cast'] = df_netflix['cast'].fillna('No Cast')
df_netflix.dropna(subset=['date_added', 'rating'], inplace=True)
plt.figure(figsize=(8, 6))
# Pie chart for distribution
colors = ['#b20710', '#221f1f'] # Netflix red and black theme
df_netflix['type'].value_counts().plot.pie(explode=[0.05, 0.05], autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribution of Netflix Content: Movies vs TV Shows', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.show()
plt.figure(figsize=(12, 6))
# Country column mein multiple names hote hain, isliye split karke pehla count karenge
top_countries = df_netflix['country'].str.split(', ').str[0].value_counts().head(10)

sns.barplot(x=top_countries.values, y=top_countries.index, palette='Reds_r')
plt.title('Top 10 Countries with most content on Netflix', fontsize=16, fontweight='bold')
plt.xlabel('Number of Titles')
plt.show()
# Year extract karna
df_netflix['year_added'] = pd.to_datetime(df_netflix['date_added']).dt.year

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_netflix.groupby('year_added')['show_id'].count(), marker='o', color='#b20710')
plt.title('Content Added Over Years', fontsize=15, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Count of Titles')
plt.grid(True, alpha=0.3)
plt.show()
