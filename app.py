import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Ensure project root is in path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_loader import load_netflix_data
from src.data_cleaning import clean_netflix_data

# Set page layout and config
st.set_page_config(
    page_title="Netflix Content Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme styling via markdown
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #E50914;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #cccccc;
        margin-bottom: 30px;
    }
    .kpi-box {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #E50914;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: white;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #8c8c8c;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Helper for caching loading and cleaning
@st.cache_data
def load_and_clean_dataset():
    df_raw = load_netflix_data()
    df_clean = clean_netflix_data(df_raw)
    return df_clean

# Load data
try:
    df = load_and_clean_dataset()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-header">🎬 NETFLIX CONTENT ANALYSIS & REC TRENDS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An Interactive Data Science Dashboard exploring over 8,800 Titles</div>', unsafe_allow_html=True)

# Main Banner / Description
col1, col2 = st.columns([2, 1])
with col1:
    st.image("images/content_trends.png", use_container_width=True)
with col2:
    st.subheader("Welcome to the Netflix Content Dashboard!")
    st.write("""
        This dashboard lets you explore Netflix's content strategy shifts, global footprints, and underlying machine learning features.

        Use the sidebar to navigate between:
        - **📊 Exploratory Data Analysis**: Deep-dive filters, interactive charts, and trend lines.
        - **🤖 Machine Learning**: Real-time content type classifiers, K-Means clustering groups, and statistical distribution curves.
        - **🎬 Smart Recommender**: A custom TF-IDF natural language recommendation engine built on show descriptions.

        *Built with Streamlit, Plotly, and Scikit-Learn.*
    """)
    st.info("💡 **Fun Fact**: Netflix shifted heavily towards TV Shows starting around 2015 to increase subscriber engagement and retention!")

st.markdown("---")

# Key KPIs
st.subheader("📈 Executive Overview")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_titles = len(df)
movies_count = len(df[df['type'] == 'Movie'])
shows_count = len(df[df['type'] == 'TV Show'])
unique_countries = df[df['primary_country'] != 'Unknown']['primary_country'].nunique()
min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())

with kpi1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-value">{total_titles:,}</div>
        <div class="kpi-label">Total Titles</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-box" style="border-left-color: #E50914;">
        <div class="kpi-value">{movies_count:,}</div>
        <div class="kpi-label">Movies</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-box" style="border-left-color: #F5A623;">
        <div class="kpi-value">{shows_count:,}</div>
        <div class="kpi-label">TV Shows</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-box" style="border-left-color: #00D4AA;">
        <div class="kpi-value">{unique_countries}</div>
        <div class="kpi-label">Countries</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="kpi-box" style="border-left-color: #9B59B6;">
        <div class="kpi-value">{min_year}-{max_year}</div>
        <div class="kpi-label">Year Range</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# High-Level Visual Summary
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.write("📊 **Content Type Breakdown**")
    type_counts = df['type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    fig_pie = px.pie(
        type_counts,
        values='Count',
        names='Type',
        color='Type',
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#F5A623'},
        hole=0.4
    )
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_chart2:
    st.write("📅 **Content Added to Library Over Time**")
    added_by_year = df.groupby(['year_added', 'type']).size().reset_index(name='Count')
    fig_line = px.line(
        added_by_year,
        x='year_added',
        y='Count',
        color='type',
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#F5A623'},
        markers=True
    )
    fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', xaxis_title="Year Added", yaxis_title="Number of Titles")
    st.plotly_chart(fig_line, use_container_width=True)

# Dataset Preview
st.markdown("---")
st.write("📋 **Dataset Sample (Cleaned)**")
if st.checkbox("Show raw sample dataframe"):
    st.dataframe(df.head(100), use_container_width=True)
else:
    st.write("Check the checkbox above to view the dataframe.")
