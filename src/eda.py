# src/eda.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: Exploratory Data Analysis — create 8 professional charts
# Each chart tells a different story about Netflix's content strategy
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── Global style: makes all charts look professional ──────────────
plt.rcParams['figure.facecolor'] = '#0F0F0F'   # Netflix dark background
plt.rcParams['axes.facecolor']   = '#141414'
plt.rcParams['text.color']       = 'white'
plt.rcParams['axes.labelcolor']  = 'white'
plt.rcParams['xtick.color']      = 'white'
plt.rcParams['ytick.color']      = 'white'
plt.rcParams['axes.edgecolor']   = '#333333'
plt.rcParams['grid.color']       = '#222222'
plt.rcParams['font.family']      = 'DejaVu Sans'

NETFLIX_RED  = '#E50914'
NETFLIX_DARK = '#141414'
COLOR_MOVIE  = '#E50914'   # red for movies
COLOR_SHOW   = '#F5A623'   # orange for TV shows
ACCENT       = '#00D4AA'   # teal accent

# Output folder for saving charts
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(filename):
    """Save figure to visualizations folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=plt.rcParams['figure.facecolor'])
    print(f"   ✅ Saved: visualizations/{filename}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# CHART 1: Movies vs TV Shows — Pie + Bar (side by side)
# Business question: What type of content does Netflix focus on?
# ═══════════════════════════════════════════════════════════════
def plot_content_distribution(df):
    print("\n📊 Chart 1: Content Distribution (Movies vs TV Shows)...")

    counts = df['type'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Netflix Content: Movies vs TV Shows',
                 fontsize=18, color='white', fontweight='bold', y=1.02)

    # Pie chart
    colors = [COLOR_MOVIE, COLOR_SHOW]
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': '#0F0F0F', 'linewidth': 2},
        textprops={'color': 'white', 'fontsize': 13}
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_color('white')
        at.set_fontweight('bold')
    ax1.set_title('Content Type Split', color='white', fontsize=13, pad=10)

    # Bar chart
    bars = ax2.bar(counts.index, counts.values, color=colors,
                   edgecolor='#0F0F0F', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 50,
                 f'{val:,}', ha='center', va='bottom',
                 color='white', fontsize=12, fontweight='bold')
    ax2.set_title('Total Count by Type', color='white', fontsize=13, pad=10)
    ax2.set_ylabel('Number of Titles', color='white')
    ax2.set_ylim(0, counts.max() * 1.15)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig('01_content_distribution.png')


# ═══════════════════════════════════════════════════════════════
# CHART 2: Year-wise Content Growth (Line Chart)
# Business question: How fast is Netflix growing its library?
# ═══════════════════════════════════════════════════════════════
def plot_yearly_growth(df):
    print("📊 Chart 2: Year-wise Content Growth...")

    # Filter valid years and count by year + type
    df_yr = df[df['year_added'].notna() & (df['year_added'] >= 2008)]
    yearly = df_yr.groupby(['year_added', 'type']).size().unstack(fill_value=0)
    yearly.index = yearly.index.astype(int)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Netflix Library Growth Over the Years',
                 fontsize=18, color='white', fontweight='bold')

    if 'Movie' in yearly.columns:
        ax.fill_between(yearly.index, yearly['Movie'],
                        alpha=0.3, color=COLOR_MOVIE)
        ax.plot(yearly.index, yearly['Movie'], color=COLOR_MOVIE,
                linewidth=2.5, marker='o', markersize=5, label='Movies')

    if 'TV Show' in yearly.columns:
        ax.fill_between(yearly.index, yearly['TV Show'],
                        alpha=0.3, color=COLOR_SHOW)
        ax.plot(yearly.index, yearly['TV Show'], color=COLOR_SHOW,
                linewidth=2.5, marker='s', markersize=5, label='TV Shows')

    ax.set_xlabel('Year Added to Netflix', fontsize=12)
    ax.set_ylabel('Number of Titles Added', fontsize=12)
    ax.legend(fontsize=11, facecolor='#1a1a1a', labelcolor='white')
    ax.grid(alpha=0.3)

    # Annotate peak year
    if 'Movie' in yearly.columns:
        peak_yr = yearly['Movie'].idxmax()
        peak_val = yearly['Movie'].max()
        ax.annotate(f'Peak: {peak_yr}\n{peak_val} movies',
                    xy=(peak_yr, peak_val),
                    xytext=(peak_yr - 2, peak_val + 30),
                    color=COLOR_MOVIE, fontsize=10,
                    arrowprops=dict(arrowstyle='->', color=COLOR_MOVIE))

    plt.tight_layout()
    save_fig('02_yearly_growth.png')


# ═══════════════════════════════════════════════════════════════
# CHART 3: Top 10 Genres — Horizontal Bar Chart
# Business question: What genres dominate Netflix?
# ═══════════════════════════════════════════════════════════════
def plot_top_genres(df):
    print("📊 Chart 3: Top 10 Genres...")

    # Split genres (each title can have multiple genres)
    all_genres = df['listed_in'].dropna().str.split(',').explode().str.strip()
    top_genres = all_genres.value_counts().head(10)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Top 10 Content Genres on Netflix',
                 fontsize=18, color='white', fontweight='bold')

    # Color gradient from red to lighter
    colors_list = [NETFLIX_RED] + ['#C0392B'] * 3 + ['#E74C3C'] * 3 + ['#F1948A'] * 3
    bars = ax.barh(top_genres.index[::-1], top_genres.values[::-1],
                   color=colors_list[::-1], edgecolor='#0F0F0F', linewidth=1)

    for bar, val in zip(bars, top_genres.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', color='white', fontsize=10)

    ax.set_xlabel('Number of Titles', fontsize=12)
    ax.set_xlim(0, top_genres.max() * 1.15)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig('03_top_genres.png')


# ═══════════════════════════════════════════════════════════════
# CHART 4: Top 15 Countries — World-style Bar Chart
# Business question: Which countries produce Netflix content?
# ═══════════════════════════════════════════════════════════════
def plot_country_distribution(df):
    print("📊 Chart 4: Country Distribution...")

    countries = df['primary_country'].replace('Unknown', np.nan).dropna()
    top_countries = countries.value_counts().head(15)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('Top 15 Content-Producing Countries on Netflix',
                 fontsize=18, color='white', fontweight='bold')

    color_map = [NETFLIX_RED if i == 0 else ACCENT if i < 3 else '#555555'
                 for i in range(len(top_countries))]
    bars = ax.bar(range(len(top_countries)), top_countries.values,
                  color=color_map, edgecolor='#0F0F0F', linewidth=1)

    ax.set_xticks(range(len(top_countries)))
    ax.set_xticklabels(top_countries.index, rotation=40, ha='right', fontsize=9)

    for bar, val in zip(bars, top_countries.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 15,
                f'{val:,}', ha='center', va='bottom',
                color='white', fontsize=8)

    ax.set_ylabel('Number of Titles', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=NETFLIX_RED, label='#1 Producer'),
               mpatches.Patch(color=ACCENT, label='Top 3'),
               mpatches.Patch(color='#555555', label='Others')]
    ax.legend(handles=patches, facecolor='#1a1a1a', labelcolor='white')
    plt.tight_layout()
    save_fig('04_country_distribution.png')


# ═══════════════════════════════════════════════════════════════
# CHART 5: Ratings Distribution — Grouped Bar
# Business question: Who is Netflix's target audience?
# ═══════════════════════════════════════════════════════════════
def plot_ratings_analysis(df):
    print("📊 Chart 5: Ratings Analysis...")

    # Standard Netflix ratings in order
    rating_order = ['G', 'PG', 'PG-13', 'TV-G', 'TV-Y', 'TV-Y7',
                    'TV-PG', 'TV-14', 'TV-MA', 'R', 'NC-17', 'NR', 'UR']

    rating_data = df[df['rating'].isin(rating_order)]
    rating_counts = (rating_data.groupby(['rating', 'type'])
                                .size().unstack(fill_value=0))

    # Keep only ratings with data
    rating_counts = rating_counts[
        rating_counts.index.isin(rating_order)
    ].reindex([r for r in rating_order if r in rating_counts.index])

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Content Ratings: Who Does Netflix Target?',
                 fontsize=18, color='white', fontweight='bold')

    x = np.arange(len(rating_counts))
    width = 0.35

    if 'Movie' in rating_counts.columns:
        bars1 = ax.bar(x - width/2, rating_counts['Movie'],
                       width, label='Movies', color=COLOR_MOVIE,
                       edgecolor='#0F0F0F', linewidth=1)
    if 'TV Show' in rating_counts.columns:
        bars2 = ax.bar(x + width/2, rating_counts['TV Show'],
                       width, label='TV Shows', color=COLOR_SHOW,
                       edgecolor='#0F0F0F', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(rating_counts.index, fontsize=10)
    ax.set_ylabel('Number of Titles', fontsize=12)
    ax.legend(fontsize=11, facecolor='#1a1a1a', labelcolor='white')
    ax.grid(axis='y', alpha=0.3)

    # Highlight adult content
    ax.axvspan(len(rating_counts) - 3.5, len(rating_counts),
               alpha=0.08, color=NETFLIX_RED, label='Adult ratings')

    plt.tight_layout()
    save_fig('05_ratings_analysis.png')


# ═══════════════════════════════════════════════════════════════
# CHART 6: Movie Duration Distribution — Histogram
# Business question: How long are Netflix movies?
# ═══════════════════════════════════════════════════════════════
def plot_duration_distribution(df):
    print("📊 Chart 6: Duration Distribution...")

    movies = df[(df['type'] == 'Movie') &
                (df['duration_int'].notna()) &
                (df['duration_int'] > 0) &
                (df['duration_int'] < 300)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Netflix Content Duration Analysis',
                 fontsize=18, color='white', fontweight='bold')

    # Movie duration histogram
    axes[0].hist(movies['duration_int'], bins=30,
                 color=COLOR_MOVIE, edgecolor='#0F0F0F', linewidth=0.5, alpha=0.85)
    axes[0].axvline(movies['duration_int'].mean(), color=ACCENT,
                    linewidth=2, linestyle='--',
                    label=f"Mean: {movies['duration_int'].mean():.0f} min")
    axes[0].axvline(movies['duration_int'].median(), color=COLOR_SHOW,
                    linewidth=2, linestyle=':',
                    label=f"Median: {movies['duration_int'].median():.0f} min")
    axes[0].set_xlabel('Duration (minutes)', fontsize=12)
    axes[0].set_ylabel('Number of Movies', fontsize=12)
    axes[0].set_title('Movie Length Distribution', color='white', fontsize=13)
    axes[0].legend(facecolor='#1a1a1a', labelcolor='white')
    axes[0].grid(alpha=0.3)

    # TV Show seasons
    shows = df[(df['type'] == 'TV Show') & (df['duration_int'].notna())]
    season_counts = shows['duration_int'].value_counts().head(8).sort_index()
    axes[1].bar(season_counts.index.astype(int), season_counts.values,
                color=COLOR_SHOW, edgecolor='#0F0F0F', linewidth=1)
    axes[1].set_xlabel('Number of Seasons', fontsize=12)
    axes[1].set_ylabel('Number of TV Shows', fontsize=12)
    axes[1].set_title('TV Show Season Count', color='white', fontsize=13)
    axes[1].set_xticks(season_counts.index.astype(int))
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig('06_duration_distribution.png')


# ═══════════════════════════════════════════════════════════════
# CHART 7: Monthly Content Addition — Heatmap
# Business question: When does Netflix add the most content?
# ═══════════════════════════════════════════════════════════════
def plot_monthly_heatmap(df):
    print("📊 Chart 7: Monthly Addition Heatmap...")

    df_heat = df[df['year_added'].notna() & df['month_added'].notna()].copy()
    df_heat = df_heat[df_heat['year_added'] >= 2015]
    df_heat['year_added']  = df_heat['year_added'].astype(int)
    df_heat['month_added'] = df_heat['month_added'].astype(int)

    pivot = df_heat.pivot_table(index='month_added', columns='year_added',
                                 values='show_id', aggfunc='count', fill_value=0)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.index = [month_names[m-1] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('When Does Netflix Add Content? (Monthly Heatmap)',
                 fontsize=18, color='white', fontweight='bold')

    sns.heatmap(pivot, annot=True, fmt='d', cmap='Reds',
                linewidths=0.5, linecolor='#0F0F0F',
                ax=ax, cbar_kws={'label': 'Titles Added'})
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)
    ax.tick_params(colors='white')

    plt.tight_layout()
    save_fig('07_monthly_heatmap.png')


# ═══════════════════════════════════════════════════════════════
# CHART 8: Top Directors — Horizontal Bar
# Business question: Which directors work most with Netflix?
# ═══════════════════════════════════════════════════════════════
def plot_top_directors(df):
    print("📊 Chart 8: Top Directors...")

    top_dir = (df[df['director'] != 'Unknown']['director']
               .value_counts().head(12))

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Most Prolific Directors on Netflix',
                 fontsize=18, color='white', fontweight='bold')

    colors_list = [NETFLIX_RED] + [ACCENT] * 2 + ['#555555'] * 9
    bars = ax.barh(top_dir.index[::-1], top_dir.values[::-1],
                   color=colors_list[::-1], edgecolor='#0F0F0F', linewidth=1)

    for bar, val in zip(bars, top_dir.values[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color='white', fontsize=10)

    ax.set_xlabel('Number of Titles', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig('08_top_directors.png')


# ═══════════════════════════════════════════════════════════════
# MASTER FUNCTION — run all charts
# ═══════════════════════════════════════════════════════════════
def run_all_eda(df):
    print("\n" + "="*55)
    print("📊 STARTING EXPLORATORY DATA ANALYSIS")
    print("="*55)

    plot_content_distribution(df)
    plot_yearly_growth(df)
    plot_top_genres(df)
    plot_country_distribution(df)
    plot_ratings_analysis(df)
    plot_duration_distribution(df)
    plot_monthly_heatmap(df)
    plot_top_directors(df)

    print("\n" + "="*55)
    print(f"✅ ALL 8 CHARTS SAVED to /visualizations/")
    print("="*55)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_netflix_data
    from data_cleaning import clean_netflix_data

    df_raw   = load_netflix_data()
    df_clean = clean_netflix_data(df_raw)
    run_all_eda(df_clean)