import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Ensure project root is in path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_loader import load_netflix_data
from src.data_cleaning import clean_netflix_data

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

# Styling
st.markdown("""
<style>
    .section-title {
        color: #E50914;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: bold;
        font-size: 2rem;
        margin-top: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">📊 Exploratory Data Analysis & Content Explorer</div>', unsafe_allow_html=True)

# Helper for caching loading and cleaning
@st.cache_data
def load_and_clean_dataset():
    df_raw = load_netflix_data()
    df_clean = clean_netflix_data(df_raw)
    return df_clean

df = load_and_clean_dataset()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filter Options")

# 1. Content Type Filter
content_types = ["All"] + list(df['type'].unique())
selected_type = st.sidebar.selectbox("Content Type", content_types)

# 2. Year Filter
min_year_val = int(df['release_year'].min())
max_year_val = int(df['release_year'].max())
selected_years = st.sidebar.slider(
    "Release Year Range",
    min_value=min_year_val,
    max_value=max_year_val,
    value=(2000, max_year_val)
)

# 3. Rating Filter
ratings = sorted(list(df['rating'].unique()))
selected_ratings = st.sidebar.multiselect("Content Ratings", ratings, default=[])

# 4. Genre Filter
# Extract unique genres from 'listed_in' column (comma-separated lists)
all_genres = set()
df['listed_in'].dropna().apply(lambda x: [all_genres.add(g.strip()) for g in x.split(',')])
unique_genres = sorted(list(all_genres))
selected_genres = st.sidebar.multiselect("Genres / Categories", unique_genres, default=[])

# 5. Country Filter
unique_countries = sorted([c for c in df['primary_country'].unique() if c != 'Unknown'])
selected_countries = st.sidebar.multiselect("Top Producing Countries", unique_countries, default=[])

# Apply filters
df_filtered = df.copy()

# Filter by type
if selected_type != "All":
    df_filtered = df_filtered[df_filtered['type'] == selected_type]

# Filter by year
df_filtered = df_filtered[
    (df_filtered['release_year'] >= selected_years[0]) &
    (df_filtered['release_year'] <= selected_years[1])
]

# Filter by rating
if selected_ratings:
    df_filtered = df_filtered[df_filtered['rating'].isin(selected_ratings)]

# Filter by genre
if selected_genres:
    # A row matches if any of selected_genres is in its listed_in list
    genre_mask = df_filtered['listed_in'].apply(
        lambda x: any(g.strip() in selected_genres for g in str(x).split(','))
    )
    df_filtered = df_filtered[genre_mask]

# Filter by country
if selected_countries:
    df_filtered = df_filtered[df_filtered['primary_country'].isin(selected_countries)]

# --- INTERACTIVE VISUALIZATIONS ---

# High-level info
st.write(f"Showing **{len(df_filtered):,}** matching titles based on your filters.")

col1, col2 = st.columns(2)

with col1:
    # 1. Top Genres Bar Chart
    st.markdown("### 🏆 Top Genres")
    # Count genre occurrences in filtered set
    genre_series = df_filtered['listed_in'].str.split(',').dropna().explode().str.strip()
    genre_counts = genre_series.value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    top_genres = genre_counts.head(15)

    if not top_genres.empty:
        fig_genre = px.bar(
            top_genres,
            x='Count',
            y='Genre',
            orientation='h',
            color='Count',
            color_continuous_scale='Reds',
            labels={'Genre': 'Genre', 'Count': 'Number of Titles'},
            title="Top 15 Genre Distributions"
        )
        fig_genre.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_genre, use_container_width=True)
    else:
        st.info("No genre data available for current filters.")

with col2:
    # 2. Release Year Growth
    st.markdown("### 📅 Release Year Distribution")
    year_type = df_filtered.groupby(['release_year', 'type']).size().reset_index(name='Count')

    if not year_type.empty:
        fig_year = px.area(
            year_type,
            x='release_year',
            y='Count',
            color='type',
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#F5A623'},
            labels={'release_year': 'Release Year', 'Count': 'Number of Titles'},
            title="Content Release Trends Over Decades"
        )
        fig_year.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info("No release year data available.")

col3, col4 = st.columns(2)

with col3:
    # 3. Ratings Distribution
    st.markdown("### 🏷️ Content Ratings")
    rating_counts = df_filtered['rating'].value_counts().reset_index()
    rating_counts.columns = ['Rating', 'Count']

    if not rating_counts.empty:
        fig_rating = px.bar(
            rating_counts.head(10),
            x='Rating',
            y='Count',
            color='Count',
            color_continuous_scale='Blues',
            title="Top 10 Ratings Breakdown"
        )
        fig_rating.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_rating, use_container_width=True)
    else:
        st.info("No rating data available.")

with col4:
    # 4. Country Breakdown (excluding Unknown)
    st.markdown("### 🌍 Top Content-Producing Countries")
    country_counts = df_filtered[df_filtered['primary_country'] != 'Unknown']['primary_country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    top_countries = country_counts.head(15)

    if not top_countries.empty:
        fig_country = px.bar(
            top_countries,
            x='Count',
            y='Country',
            orientation='h',
            color='Count',
            color_continuous_scale='Oranges',
            title="Top 15 Producing Countries (Primary Country Only)"
        )
        fig_country.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.info("No country data available.")

# 5. Duration Analysis
st.markdown("---")
st.markdown("### ⏱️ Duration / Runtime Analysis")

col5, col6 = st.columns([1, 2])

with col5:
    st.write("""
        Netflix duration formats differ significantly depending on the content type:
        - **Movies** are measured in **minutes** (run-time).
        - **TV Shows** are measured in **seasons** (number of seasons).

        Our analysis dynamically handles and parses these. Feel free to inspect how duration is distributed across both types below.
    """)

with col6:
    tab_movie, tab_show = st.tabs(["🎥 Movie Durations (Min)", "📺 TV Show Seasons"])

    with tab_movie:
        movie_df = df_filtered[df_filtered['type'] == 'Movie']
        if not movie_df.empty:
            fig_movie_dur = px.histogram(
                movie_df,
                x='duration_int',
                nbins=30,
                color_discrete_sequence=['#E50914'],
                labels={'duration_int': 'Duration (minutes)'},
                title="Movie Runtime Distribution"
            )
            fig_movie_dur.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_movie_dur, use_container_width=True)
        else:
            st.info("Please filter by Movie type or expand your year/country/genre selection to view movie durations.")

    with tab_show:
        show_df = df_filtered[df_filtered['type'] == 'TV Show']
        if not show_df.empty:
            show_durations = show_df['duration_int'].value_counts().reset_index()
            show_durations.columns = ['Seasons', 'Count']
            show_durations = show_durations.sort_values(by='Seasons')
            fig_show_dur = px.bar(
                show_durations,
                x='Seasons',
                y='Count',
                color='Count',
                color_continuous_scale='YlOrRd',
                labels={'Seasons': 'Number of Seasons'},
                title="TV Show Season Distribution"
            )
            fig_show_dur.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_show_dur, use_container_width=True)
        else:
            st.info("Please filter by TV Show type or expand your selection to view TV show season distributions.")
