"""
===============================================================================
🎬 Netflix Content Analysis — Executive Landing Dashboard
===============================================================================

File Purpose:
-------------
This file (`app.py`) serves as the production-grade landing page and executive
overview dashboard for the multi-page Streamlit web application. It integrates
data ingestion, validation, exploratory metrics, interactive Plotly visualizations,
and dataset exploration tools.

Multi-Page Navigation Structure:
--------------------------------
Streamlit automatically detects scripts in the `pages/` directory to construct
the primary sidebar navigation:
  • app.py                             -> Executive Landing Page & Catalog Overview
  • pages/01_📊_Exploratory_Data_Analysis.py -> Deep-Dive Exploratory Analysis & Filters
  • pages/02_🤖_Machine_Learning.py        -> Classifier, KMeans Clustering, Stats
  • pages/03_🎬_Smart_Recommender.py       -> NLP TF-IDF Cosine Recommender Engine

Data Dependencies:
------------------
  • data/netflix_titles.csv         : Raw Netflix dataset (~8,800 records)
  • data/cleaned_netflix_data.csv   : Preprocessed dataset (cached fallback)
  • src/data_loader.py              : Safe dataset ingestion handler
  • src/data_cleaning.py            : Feature extraction and data hygiene

Architecture & Design Principles:
---------------------------------
  • Modern Dark Theme (Netflix Red #E50914, Charcoal, Gold, Teal accents)
  • Caching with @st.cache_data for instant load times
  • Defensive error handling & schema validation to prevent application crashes
  • Fully interactive Plotly charts with rich tooltips, legends, and dark styling
  • Modular design complying with PEP8 guidelines and clean software architecture

===============================================================================
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# System Path Setup
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from src.data_loader import load_netflix_data
    from src.data_cleaning import clean_netflix_data
except ImportError:
    # Graceful fallback if executing from sub-directory
    sys.path.append(os.path.join(BASE_DIR, "src"))
    from data_loader import load_netflix_data
    from data_cleaning import clean_netflix_data

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("netflix_dashboard")

# -----------------------------------------------------------------------------
# Configuration Management
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DashboardConfig:
    """Configuration settings for dashboard aesthetics, files, and validation."""

    PAGE_TITLE: str = "Netflix Content Strategy & Analytics"
    PAGE_ICON: str = "🎬"
    LAYOUT: str = "wide"

    # Palette definition
    COLOR_RED: str = "#E50914"
    COLOR_GOLD: str = "#F5A623"
    COLOR_TEAL: str = "#00D4AA"
    COLOR_PURPLE: str = "#9B59B6"
    COLOR_BLUE: str = "#3498DB"
    COLOR_DARK_BG: str = "#141414"
    COLOR_CARD_BG: str = "#1F1F1F"
    COLOR_TEXT_LIGHT: str = "#E5E5E5"
    COLOR_TEXT_MUTED: str = "#A0A0A0"

    REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: [
        "show_id", "type", "title", "director", "cast", "country",
        "date_added", "release_year", "rating", "duration", "listed_in",
        "description"
    ])

    DEFAULT_YEAR_START: int = 2000
    FALLBACK_MOVIE_DURATION: float = 90.0
    FALLBACK_YEAR: int = 2019


CONFIG = DashboardConfig()

# Set Streamlit Page Configuration early
st.set_page_config(
    page_title=CONFIG.PAGE_TITLE,
    page_icon=CONFIG.PAGE_ICON,
    layout=CONFIG.LAYOUT,
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# UI Custom Styling Injection
# -----------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject polished custom CSS for executive Netflix theme styling."""
    css = f"""
    <style>
        /* Base page background styling */
        .stApp {{
            background-color: {CONFIG.COLOR_DARK_BG};
            color: {CONFIG.COLOR_TEXT_LIGHT};
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }}

        /* Main Header Banner */
        .hero-banner {{
            background: linear-gradient(135deg, #111111 0%, #1a0003 50%, #2b0005 100%);
            border-bottom: 3px solid {CONFIG.COLOR_RED};
            padding: 2.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 24px rgba(229, 9, 20, 0.25);
            text-align: center;
        }}

        .hero-title {{
            color: {CONFIG.COLOR_RED};
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
            letter-spacing: -0.5px;
            text-transform: uppercase;
        }}

        .hero-subtitle {{
            color: {CONFIG.COLOR_TEXT_MUTED};
            font-size: 1.15rem;
            font-weight: 400;
            max-width: 800px;
            margin: 0 auto;
        }}

        /* Metric Cards */
        .kpi-card {{
            background-color: {CONFIG.COLOR_CARD_BG};
            border-radius: 10px;
            padding: 1.2rem;
            border-top: 4px solid {CONFIG.COLOR_RED};
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(229, 9, 20, 0.2);
        }}

        .kpi-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.2;
        }}

        .kpi-label {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {CONFIG.COLOR_TEXT_MUTED};
            margin-top: 0.4rem;
        }}

        /* Tab Navigation Styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 12px;
            border-bottom: 1px solid #333333;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: #1A1A1A;
            border-radius: 6px 6px 0 0;
            padding: 10px 20px;
            color: {CONFIG.COLOR_TEXT_MUTED};
            font-weight: 600;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {CONFIG.COLOR_RED} !important;
            color: #FFFFFF !important;
        }}

        /* Info box badge */
        .badge-info {{
            background-color: #1e293b;
            border-left: 4px solid {CONFIG.COLOR_TEAL};
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Validation Engine
# -----------------------------------------------------------------------------
def validate_netflix_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the structure and integrity of the loaded dataset.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        Tuple[bool, List[str]]: Validation status (True if valid) and error messages list.
    """
    errors = []

    if df is None or df.empty:
        return False, ["Dataset is empty or None."]

    missing_cols = [col for col in CONFIG.REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    if len(df) < 10:
        errors.append(f"Dataset has suspiciously few records ({len(df)} rows).")

    return (len(errors) == 0), errors


# -----------------------------------------------------------------------------
# Cached Data Ingestion Pipeline
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Ingesting and preprocessing Netflix dataset...")
def load_and_validate_data() -> pd.DataFrame:
    """
    Load, clean, and validate Netflix dataset with caching and error handling.

    Returns:
        pd.DataFrame: Validated cleaned pandas dataframe.
    """
    logger.info("Initiating dataset loading sequence...")

    df_raw = None
    # Method 1: Load via src.data_loader
    try:
        df_raw = load_netflix_data()
    except Exception as err_loader:
        logger.warning(f"Primary data loader failed: {err_loader}. Attempting fallback path...")
        fallback_path = os.path.join(BASE_DIR, "data", "cleaned_netflix_data.csv")
        if os.path.exists(fallback_path):
            df_raw = pd.read_csv(fallback_path)
        else:
            raise FileNotFoundError(
                "Neither raw nor cleaned data files were found in the data/ directory."
            ) from err_loader

    # Preprocess and clean data
    try:
        df_clean = clean_netflix_data(df_raw)
    except Exception as err_clean:
        logger.warning(f"Custom clean pipeline error: {err_clean}. Applying fallback hygiene.")
        df_clean = df_raw.copy()

    # Schema hygiene & derived feature checks
    if "primary_country" not in df_clean.columns:
        df_clean["primary_country"] = df_clean["country"].fillna("Unknown").apply(
            lambda x: str(x).split(",")[0].strip() if x != "Unknown" else "Unknown"
        )

    if "duration_int" not in df_clean.columns:
        def parse_dur(val):
            if pd.isna(val):
                return np.nan
            parts = str(val).strip().split(" ")
            try:
                return float(parts[0])
            except (ValueError, IndexError):
                return np.nan
        df_clean["duration_int"] = df_clean["duration"].apply(parse_dur)
        df_clean["duration_int"] = df_clean["duration_int"].fillna(CONFIG.FALLBACK_MOVIE_DURATION).astype(int)

    if "year_added" not in df_clean.columns:
        if "date_added" in df_clean.columns:
            df_clean["year_added"] = pd.to_datetime(
                df_clean["date_added"].astype(str).str.strip(), errors="coerce"
            ).dt.year
            df_clean["year_added"] = df_clean["year_added"].fillna(CONFIG.FALLBACK_YEAR).astype(int)
        else:
            df_clean["year_added"] = CONFIG.FALLBACK_YEAR

    # Validate dataset
    is_valid, validation_errors = validate_netflix_dataset(df_clean)
    if not is_valid:
        raise ValueError(f"Dataset failed schema validation: {'; '.join(validation_errors)}")

    logger.info(f"Dataset ready. Final shape: {df_clean.shape}")
    return df_clean


# -----------------------------------------------------------------------------
# Interactive Visualizations (Plotly Engine)
# -----------------------------------------------------------------------------
def plot_content_type_donut(df: pd.DataFrame) -> go.Figure:
    """Generate interactive donut chart for content type distribution."""
    counts = df["type"].value_counts().reset_index()
    counts.columns = ["Type", "Count"]

    fig = px.pie(
        counts,
        values="Count",
        names="Type",
        color="Type",
        color_discrete_map={"Movie": CONFIG.COLOR_RED, "TV Show": CONFIG.COLOR_GOLD},
        hole=0.45,
        title="<b>Content Catalog Distribution</b>"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent:.1%}<extra></extra>"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CONFIG.COLOR_TEXT_LIGHT),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_content_addition_trend(df: pd.DataFrame) -> go.Figure:
    """Generate line chart for cumulative and yearly content additions over time."""
    trend_df = df.groupby(["year_added", "type"]).size().reset_index(name="Count")

    fig = px.line(
        trend_df,
        x="year_added",
        y="Count",
        color="type",
        color_discrete_map={"Movie": CONFIG.COLOR_RED, "TV Show": CONFIG.COLOR_GOLD},
        markers=True,
        labels={"year_added": "Year Added to Netflix", "Count": "Titles Added"},
        title="<b>Content Additions Over Time</b>"
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CONFIG.COLOR_TEXT_LIGHT),
        xaxis=dict(showgrid=True, gridcolor="#333333"),
        yaxis=dict(showgrid=True, gridcolor="#333333"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_top_genres(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Generate horizontal bar chart for top genres."""
    genres_series = df["listed_in"].dropna().str.split(",").explode().str.strip()
    genre_counts = genres_series.value_counts().head(top_n).reset_index()
    genre_counts.columns = ["Genre", "Count"]

    fig = px.bar(
        genre_counts,
        x="Count",
        y="Genre",
        orientation="h",
        color="Count",
        color_continuous_scale="Reds",
        title=f"<b>Top {top_n} Genres / Content Categories</b>",
        text="Count"
    )

    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CONFIG.COLOR_TEXT_LIGHT),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_top_countries(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Generate horizontal bar chart for top producing countries."""
    valid_countries = df[df["primary_country"] != "Unknown"]
    country_counts = valid_countries["primary_country"].value_counts().head(top_n).reset_index()
    country_counts.columns = ["Country", "Count"]

    fig = px.bar(
        country_counts,
        x="Count",
        y="Country",
        orientation="h",
        color="Count",
        color_continuous_scale="Oranges",
        title=f"<b>Top {top_n} Producing Countries</b>",
        text="Count"
    )

    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CONFIG.COLOR_TEXT_LIGHT),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_duration_distributions(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Generate movie duration histogram and TV show season count bar chart."""
    movies_df = df[df["type"] == "Movie"]
    shows_df = df[df["type"] == "TV Show"]

    # Movie Duration Histogram
    fig_movie = px.histogram(
        movies_df,
        x="duration_int",
        nbins=35,
        color_discrete_sequence=[CONFIG.COLOR_RED],
        title="<b>Movie Duration Distribution (Minutes)</b>",
        labels={"duration_int": "Runtime (mins)", "count": "Movie Count"}
    )

    if not movies_df.empty:
        median_movie = movies_df["duration_int"].median()
        fig_movie.add_vline(
            x=median_movie,
            line_dash="dash",
            line_color=CONFIG.COLOR_TEAL,
            annotation_text=f"Median: {median_movie:.0f} mins",
            annotation_position="top right"
        )

    fig_movie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CONFIG.COLOR_TEXT_LIGHT),
        xaxis=dict(showgrid=True, gridcolor="#333333"),
        yaxis=dict(showgrid=True, gridcolor="#333333"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # TV Show Season Distribution
    season_counts = shows_df["duration_int"].value_counts().reset_index()
    season_counts.columns = ["Seasons", "Count"]
    season_counts = season_counts.sort_values(by="Seasons").head(10)

    fig_show = px.bar(
        season_counts,
        x="Seasons",
        y="Count",
        color="Count",
        color_continuous_scale="YlOrRd",
        title="<b>TV Show Lifespan (Number of Seasons)</b>",
        labels={"Seasons": "Seasons", "Count": "Number of Shows"},
        text="Count"
    )
    fig_show.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_show.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CONFIG.COLOR_TEXT_LIGHT),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig_movie, fig_show


def plot_top_talent(df: pd.DataFrame, col_name: str, title: str, top_n: int = 10) -> go.Figure:
    """Generate bar chart for top directors or cast members."""
    series = df[df[col_name] != "Unknown"][col_name].dropna().str.split(",").explode().str.strip()
    counts = series.value_counts().head(top_n).reset_index()
    counts.columns = [col_name.capitalize(), "Count"]

    fig = px.bar(
        counts,
        x="Count",
        y=col_name.capitalize(),
        orientation="h",
        color="Count",
        color_continuous_scale="Tealgrn",
        title=f"<b>{title}</b>",
        text="Count"
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CONFIG.COLOR_TEXT_LIGHT),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# -----------------------------------------------------------------------------
# Dynamic Sidebar Filters
# -----------------------------------------------------------------------------
def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render sidebar widgets for real-time filtering and return filtered dataframe.

    Parameters:
        df (pd.DataFrame): Source dataset.

    Returns:
        pd.DataFrame: Filtered dataset subset.
    """
    st.sidebar.markdown("## 🔍 **Catalog Filters**")
    st.sidebar.markdown("---")

    # 1. Content Type
    type_options = ["All Types"] + list(df["type"].unique())
    selected_type = st.sidebar.selectbox("Content Format", type_options)

    # 2. Release Year Range
    min_yr = int(df["release_year"].min())
    max_yr = int(df["release_year"].max())
    selected_years = st.sidebar.slider(
        "Release Year Window",
        min_value=min_yr,
        max_value=max_yr,
        value=(CONFIG.DEFAULT_YEAR_START, max_yr)
    )

    # 3. Ratings Filter
    all_ratings = sorted([str(r) for r in df["rating"].unique() if pd.notna(r) and r != "Unknown"])
    selected_ratings = st.sidebar.multiselect("Audience Ratings", all_ratings)

    # 4. Genres Filter
    all_genres = sorted(list(set(df["listed_in"].dropna().str.split(",").explode().str.strip())))
    selected_genres = st.sidebar.multiselect("Genres / Categories", all_genres)

    # 5. Country Filter
    all_countries = sorted([c for c in df["primary_country"].unique() if c != "Unknown"])
    selected_countries = st.sidebar.multiselect("Producing Country", all_countries)

    # 6. Global Search Query
    st.sidebar.markdown("---")
    search_query = st.sidebar.text_input("🔎 Search Title, Director, or Cast", "")

    # Filter processing logic
    df_filtered = df.copy()

    if selected_type != "All Types":
        df_filtered = df_filtered[df_filtered["type"] == selected_type]

    df_filtered = df_filtered[
        (df_filtered["release_year"] >= selected_years[0]) &
        (df_filtered["release_year"] <= selected_years[1])
    ]

    if selected_ratings:
        df_filtered = df_filtered[df_filtered["rating"].isin(selected_ratings)]

    if selected_genres:
        genre_mask = df_filtered["listed_in"].apply(
            lambda g: any(selected in str(g) for selected in selected_genres)
        )
        df_filtered = df_filtered[genre_mask]

    if selected_countries:
        df_filtered = df_filtered[df_filtered["primary_country"].isin(selected_countries)]

    if search_query.strip():
        q = search_query.lower().strip()
        search_mask = (
            df_filtered["title"].str.lower().str.contains(q, na=False) |
            df_filtered["director"].str.lower().str.contains(q, na=False) |
            df_filtered["cast"].str.lower().str.contains(q, na=False) |
            df_filtered["description"].str.lower().str.contains(q, na=False)
        )
        df_filtered = df_filtered[search_mask]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing **{len(df_filtered):,}** of **{len(df):,}** titles")

    return df_filtered


# -----------------------------------------------------------------------------
# Main Application Render Function
# -----------------------------------------------------------------------------
def main() -> None:
    """Main application layout and controller routine."""
    inject_custom_css()

    # Header Hero Banner
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">🎬 NETFLIX CONTENT STRATEGY DASHBOARD</div>
            <div class="hero-subtitle">
                An Executive-Grade Data Science & Machine Learning Platform analyzing over 8,800 Titles
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load dataset with caching
    try:
        df_clean = load_and_validate_data()
    except Exception as e:
        st.error(f"🚨 **Dataset Loading Error**: {e}")
        st.info("Please verify that `data/netflix_titles.csv` or `data/cleaned_netflix_data.csv` exists.")
        st.stop()

    # Apply Sidebar Filters
    df_filtered = render_sidebar_filters(df_clean)

    if df_filtered.empty:
        st.warning("⚠️ No titles match your selected filter criteria. Please broaden your selection in the sidebar.")
        st.stop()

    # KPI Executive Cards
    st.markdown("### 📊 **Executive Key Performance Indicators**")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

    total_titles = len(df_filtered)
    movie_count = len(df_filtered[df_filtered["type"] == "Movie"])
    show_count = len(df_filtered[df_filtered["type"] == "TV Show"])
    countries_count = df_filtered[df_filtered["primary_country"] != "Unknown"]["primary_country"].nunique()
    movie_share = (movie_count / total_titles * 100) if total_titles > 0 else 0

    with col_kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_titles:,}</div>
            <div class="kpi-label">Filtered Titles</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{movie_count:,}</div>
            <div class="kpi-label">Movies ({movie_share:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{show_count:,}</div>
            <div class="kpi-label">TV Shows ({100 - movie_share:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{countries_count}</div>
            <div class="kpi-label">Active Countries</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi5:
        min_y = int(df_filtered["release_year"].min()) if not df_filtered.empty else 0
        max_y = int(df_filtered["release_year"].max()) if not df_filtered.empty else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{min_y}-{max_y}</div>
            <div class="kpi-label">Release Span</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # App Information Banner
    st.markdown(
        f"""
        <div class="badge-info">
            💡 <b>Executive Insight</b>: Netflix shifted strategic investments heavily toward multi-season TV Shows starting in
            2015 to maximize viewer retention and build long-term subscription loyalty. Use the multi-page sidebar on the left to explore deeper
            <b>EDA</b>, <b>Machine Learning Models</b>, and our <b>Smart NLP Recommender</b>!
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main Dashboard Tabs
    tab_overview, tab_genres, tab_duration, tab_explorer = st.tabs([
        "📈 Library Growth & Trends",
        "📊 Genre & Demographics",
        "⏱️ Duration & Talent Insights",
        "🔍 Interactive Data Explorer"
    ])

    # Tab 1: Library Growth & Trends
    with tab_overview:
        col_t1_left, col_t1_right = st.columns([1, 1])
        with col_t1_left:
            fig_donut = plot_content_type_donut(df_filtered)
            st.plotly_chart(fig_donut, use_container_width=True)
        with col_t1_right:
            fig_addition = plot_content_addition_trend(df_filtered)
            st.plotly_chart(fig_addition, use_container_width=True)

    # Tab 2: Genre & Geographic Demographics
    with tab_genres:
        col_t2_left, col_t2_right = st.columns([1, 1])
        with col_t2_left:
            fig_genres = plot_top_genres(df_filtered, top_n=12)
            st.plotly_chart(fig_genres, use_container_width=True)
        with col_t2_right:
            fig_countries = plot_top_countries(df_filtered, top_n=12)
            st.plotly_chart(fig_countries, use_container_width=True)

    # Tab 3: Duration & Talent Insights
    with tab_duration:
        col_t3_1, col_t3_2 = st.columns([1, 1])
        fig_movie_dur, fig_show_dur = plot_duration_distributions(df_filtered)
        with col_t3_1:
            st.plotly_chart(fig_movie_dur, use_container_width=True)
        with col_t3_2:
            st.plotly_chart(fig_show_dur, use_container_width=True)

        st.markdown("---")
        col_talent_1, col_talent_2 = st.columns([1, 1])
        with col_talent_1:
            fig_directors = plot_top_talent(df_filtered, "director", "Top Directors by Title Count", top_n=10)
            st.plotly_chart(fig_directors, use_container_width=True)
        with col_talent_2:
            fig_actors = plot_top_talent(df_filtered, "cast", "Top Actors / Cast Members", top_n=10)
            st.plotly_chart(fig_actors, use_container_width=True)

    # Tab 4: Data Explorer
    with tab_explorer:
        st.subheader("📋 **Interactive Dataset Explorer**")
        st.write("Examine, filter, search, and export the underlying Netflix dataset.")

        # Display summary stats
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown(f"• **Records Displayed**: `{len(df_filtered):,}`")
            st.markdown(f"• **Columns Included**: `{len(df_filtered.columns)}`")
        with col_exp2:
            csv_data = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name="filtered_netflix_dataset.csv",
                mime="text/csv",
                type="primary"
            )

        st.dataframe(
            df_filtered[["title", "type", "director", "cast", "primary_country", "release_year", "rating", "duration", "listed_in"]],
            use_container_width=True,
            height=400
        )


if __name__ == "__main__":
    main()
