import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the theme
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Filtering data from 2010 to 2021 (the most relevant decade)
df_trend = df[df['release_year'] >= 2010]
trend_data = df_trend.groupby(['release_year', 'type']).size().reset_index(name='Total')

# Plotting
sns.lineplot(data=trend_data, x='release_year', y='Total', hue='type', 
             palette=['#E50914', '#564d4d'], linewidth=3, marker='o')

plt.title('Netflix Content Strategy Shift (2010 - 2021)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Category')
plt.show()
# Splitting genres
genres = df['listed_in'].str.split(', ').explode()
top_genres = genres.value_counts().head(10).reset_index()
top_genres.columns = ['Genre', 'Count']

plt.figure(figsize=(12, 8))
sns.barplot(data=top_genres, x='Count', y='Genre', palette='rocket')

plt.title('Top 10 Genres on Netflix', fontsize=16, fontweight='bold')
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.tight_layout()
plt.show()
import plotly.express as px

# Getting counts by country
country_counts = df['country'].str.split(', ').explode().value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

# Creating the interactive map
fig = px.choropleth(country_counts, 
                    locations="Country", 
                    locationmode='country names',
                    color="Count", 
                    hover_name="Country", 
                    title='Global Distribution of Netflix Content',
                    color_continuous_scale=px.colors.sequential.Reds)

fig.update_layout(template='plotly_dark')
fig.show()
