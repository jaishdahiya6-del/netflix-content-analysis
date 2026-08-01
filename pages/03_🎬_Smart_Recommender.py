import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root is in path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_loader import load_netflix_data
from src.data_cleaning import clean_netflix_data

st.set_page_config(page_title="Smart Recommender", page_icon="🎬", layout="wide")

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
    .rec-box {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #E50914;
    }
    .rec-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: white;
    }
    .rec-meta {
        font-size: 0.85rem;
        color: #8c8c8c;
    }
    .rec-score {
        font-size: 0.95rem;
        font-weight: bold;
        color: #00D4AA;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">🎬 Smart Natural Language Recommender</div>', unsafe_allow_html=True)

# Helper for caching loading and cleaning
@st.cache_data
def load_and_clean_dataset():
    df_raw = load_netflix_data()
    df_clean = clean_netflix_data(df_raw)
    return df_clean

df = load_and_clean_dataset()

# Combine multiple text fields for matching
@st.cache_data
def get_tfidf_matrices(dataframe):
    # Standardizing features soup
    soup = (
        dataframe['title'].fillna('') + ' ' +
        dataframe['director'].fillna('') + ' ' +
        dataframe['listed_in'].fillna('') + ' ' +
        dataframe['description'].fillna('') + ' ' +
        dataframe['primary_country'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english', max_features=4000, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(soup)
    return matrix

tfidf_matrix = get_tfidf_matrices(df)

# Create mapping from title to index
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

st.write("""
    Netflix relies on state-of-the-art recommendation algorithms.
    This interactive tool uses a **TF-IDF Vectorizer** (Term Frequency - Inverse Document Frequency)
    and **Cosine Similarity** to compare word frequencies across descriptions, genres, directors, countries, and titles.

    Choose a title below to see the **Top 5** recommendations!
""")

# Selector
all_titles = sorted(list(df['title'].unique()))
selected_title = st.selectbox("Type or select a Netflix Title:", all_titles, index=all_titles.index("Stranger Things") if "Stranger Things" in all_titles else 0)

if selected_title:
    # Get details of the selected title
    selected_row = df[df['title'] == selected_title].iloc[0]

    # Showcase selected title details
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Selected Title Info")
        st.markdown(f"""
        - **Type**: {selected_row['type']}
        - **Release Year**: {selected_row['release_year']}
        - **Rating**: {selected_row['rating']}
        - **Duration**: {selected_row['duration']}
        - **Country**: {selected_row['primary_country']}
        - **Genres**: *{selected_row['listed_in']}*
        """)
        st.info(f"**Description**: {selected_row['description']}")

    with col2:
        st.subheader("🎯 Top 5 Recommendations")

        # Calculate cosine similarity on the fly for the selected title (extremely fast with spare matrix)
        idx = indices[selected_title.lower()]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        selected_vector = tfidf_matrix[idx]
        sim_scores = cosine_similarity(selected_vector, tfidf_matrix).flatten()

        # Sort and take top 5 (ignoring itself)
        sim_indices = np.argsort(sim_scores)[::-1]
        sim_indices = [i for i in sim_indices if i != idx][:5]

        # Build recommendations df
        recs_df = df.iloc[sim_indices].copy()
        recs_df['score'] = sim_scores[sim_indices]

        # Display recommendations
        for idx_rec, row_rec in recs_df.iterrows():
            score_pct = int(row_rec['score'] * 100)

            # Show progress bar with similarity match
            st.markdown(f"""
            <div class="rec-box">
                <div class="rec-title">{row_rec['title']} ({row_rec['type']})</div>
                <div class="rec-meta"><b>Release Year</b>: {row_rec['release_year']} | <b>Country</b>: {row_rec['primary_country']} | <b>Genres</b>: {row_rec['listed_in']}</div>
                <div class="rec-meta" style="margin-top: 5px;">{row_rec['description']}</div>
                <div class="rec-score">Match Score: {score_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(row_rec['score']))

    # Visualizing similarity heatmap
    st.markdown("---")
    st.subheader("🗺️ Similarity Heatmap")
    st.write("Understand the pairwise similarities between the selected title and its top recommendations.")

    # Build list of 6 titles (selected + 5 recommendations)
    heatmap_indices = [idx] + list(recs_df.index)
    heatmap_titles = [df.iloc[i]['title'] for i in heatmap_indices]

    # Slice similarity matrix
    subset_matrix = tfidf_matrix[heatmap_indices]
    pair_sims = cosine_similarity(subset_matrix, subset_matrix)

    # Shorten titles for plotting labels
    short_titles = [t[:18] + '...' if len(t) > 18 else t for t in heatmap_titles]

    fig_heat = px.imshow(
        pair_sims,
        labels=dict(x="Title", y="Title", color="Cosine Similarity"),
        x=short_titles,
        y=short_titles,
        color_continuous_scale='Reds',
        text_auto=".2f",
        title="Content Similarity Heatmap"
    )
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_heat, use_container_width=True)
