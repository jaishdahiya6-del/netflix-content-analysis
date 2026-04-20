# src/ml_models.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: Machine Learning on Netflix data
#
# We build THREE ML models:
#   1. Classification  → predict if content is Movie or TV Show
#   2. Clustering      → group similar content together (KMeans)
#   3. Content-based   → TF-IDF similarity for recommendations
#
# These make your project 10x more impressive on GitHub!
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
def load_netflix_data():
    # CHANGE THIS LINE:
    # From: file_path = "data/netflix_titles.csv"
    # To:
    file_path = "netflix_titles.csv" 
    
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {os.path.abspath(file_path)}")
    
    # ... rest of your loading code (pd.read_csv, etc)
from sklearn.model_selection   import train_test_split, cross_val_score # type: ignore
import sklearn.preprocessing # type: ignore
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier # type: ignore
from sklearn.linear_model      import LogisticRegression # type: ignore
from sklearn.metrics           import (classification_report, confusion_matrix, # type: ignore
                                       accuracy_score, ConfusionMatrixDisplay)
from sklearn.cluster           import KMeans # type: ignore
from sklearn.decomposition     import TruncatedSVD # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise  import cosine_similarity # type: ignore

# Reuse the dark style from eda.py
plt.rcParams['figure.facecolor'] = '#0F0F0F'
plt.rcParams['axes.facecolor']   = '#141414'
plt.rcParams['text.color']       = 'white'
plt.rcParams['axes.labelcolor']  = 'white'
plt.rcParams['xtick.color']      = 'white'
plt.rcParams['ytick.color']      = 'white'
plt.rcParams['axes.edgecolor']   = '#333333'
plt.rcParams['grid.color']       = '#222222'

NETFLIX_RED = '#E50914'
COLOR_SHOW  = '#F5A623'
ACCENT      = '#00D4AA'

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=plt.rcParams['figure.facecolor'])
    print(f"   ✅ Saved: visualizations/{filename}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# ML MODEL 1: CLASSIFICATION
# Goal: Predict whether a title is a Movie or TV Show
#       using features like release_year, rating, duration, country
#
# Why this is useful in real life:
#   Netflix can automatically classify new uploaded content
# ═══════════════════════════════════════════════════════════════
def build_classifier(df):
    print("\n" + "="*55)
    print("🤖 ML MODEL 1: Content Type Classifier")
    print("="*55)

    # ── Feature Engineering ──────────────────────────────────
    # We need to turn raw columns into numbers (ML only understands numbers)
    df_ml = df.copy()

    # Encode rating (e.g., TV-MA → 8, PG → 3, etc.)
    le_rating  = sklearn.preprocessing.LabelEncoder()
    le_country = sklearn.preprocessing.LabelEncoder()

    df_ml['rating_enc']  = le_rating.fit_transform(df_ml['rating'].fillna('Unknown'))
    df_ml['country_enc'] = le_country.fit_transform(df_ml['primary_country'].fillna('Unknown'))
    df_ml['duration_int'] = df_ml['duration_int'].fillna(df_ml['duration_int'].median())
    df_ml['release_year'] = df_ml['release_year'].fillna(df_ml['release_year'].median()).astype(float)
    df_ml['year_added']   = df_ml['year_added'].fillna(df_ml['year_added'].median())

    # Our features (X) and target (y)
    features = ['release_year', 'rating_enc', 'duration_int',
                 'country_enc', 'year_added']
    target   = 'type'

    X = df_ml[features]
    y = df_ml[target]

    # ── Train / Test Split (80% train, 20% test) ─────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Training samples : {len(X_train):,}")
    print(f"📊 Testing samples  : {len(X_test):,}")

    # ── Train 3 Models and Compare ────────────────────────────
    models = {
        'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cv_acc = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        results[name] = {'model': model, 'accuracy': acc, 'cv_accuracy': cv_acc,
                         'y_pred': y_pred}
        print(f"\n✅ {name}")
        print(f"   Test Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
        print(f"   CV Accuracy (5k) : {cv_acc:.4f} ({cv_acc*100:.2f}%)")

    # ── Best model detailed report ────────────────────────────
    best_name  = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    best_pred  = results[best_name]['y_pred']

    print(f"\n🏆 Best Model: {best_name}")
    print("\n📋 Full Classification Report:")
    print(classification_report(y_test, best_pred))

    # ── VISUALIZATION 1: Model Comparison Bar Chart ───────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ML Model 1: Content Type Classifier Results',
                 fontsize=16, color='white', fontweight='bold')

    names    = list(results.keys())
    accs     = [results[n]['accuracy']    for n in names]
    cv_accs  = [results[n]['cv_accuracy'] for n in names]

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, accs,    w, label='Test Accuracy',  color=NETFLIX_RED, alpha=0.9)
    axes[0].bar(x + w/2, cv_accs, w, label='CV Accuracy',    color=ACCENT,      alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Random\nForest', 'Logistic\nReg.', 'Gradient\nBoosting'],
                             fontsize=10)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(facecolor='#1a1a1a', labelcolor='white')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_title('Model Accuracy Comparison', color='white', fontsize=13)
    for i, (a, cv) in enumerate(zip(accs, cv_accs)):
        axes[0].text(i - w/2, a + 0.01, f'{a:.3f}', ha='center', color='white', fontsize=9)
        axes[0].text(i + w/2, cv + 0.01, f'{cv:.3f}', ha='center', color='white', fontsize=9)

    # Confusion Matrix for best model
    cm = confusion_matrix(y_test, best_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=best_model.classes_)
    disp.plot(ax=axes[1], colorbar=False, cmap='Reds')
    axes[1].set_title(f'Confusion Matrix\n({best_name})',
                       color='white', fontsize=13)
    axes[1].set_facecolor('#141414')

    plt.tight_layout()
    save_fig('09_classifier_results.png')

    # ── VISUALIZATION 2: Feature Importance (Random Forest) ──
    rf_model = results['Random Forest']['model']
    importances = pd.Series(rf_model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Feature Importance — What Predicts Movie vs TV Show?',
                 fontsize=15, color='white', fontweight='bold')

    colors_fi = [NETFLIX_RED if i == len(importances)-1 else ACCENT
                 for i in range(len(importances))]
    ax.barh(importances.index, importances.values,
            color=colors_fi, edgecolor='#0F0F0F')
    ax.set_xlabel('Feature Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig('10_feature_importance.png')

    return best_model, results


# ═══════════════════════════════════════════════════════════════
# ML MODEL 2: CLUSTERING (KMeans)
# Goal: Group Netflix content into natural clusters
#       Discover hidden content categories Netflix uses
#
# Why this matters:
#   Netflix uses clustering internally to group similar shows
#   for their recommendation engine
# ═══════════════════════════════════════════════════════════════
def build_clustering(df):
    print("\n" + "="*55)
    print("🤖 ML MODEL 2: KMeans Content Clustering")
    print("="*55)

    # Prepare features for clustering
    df_cl = df.copy()
    le = sklearn.preprocessing.LabelEncoder()
    df_cl['rating_enc']  = le.fit_transform(df_cl['rating'].fillna('Unknown'))
    df_cl['type_enc']    = (df_cl['type'] == 'Movie').astype(int)
    df_cl['duration_int'] = df_cl['duration_int'].fillna(df_cl['duration_int'].median())
    df_cl['release_year'] = df_cl['release_year'].fillna(2015).astype(float)
    df_cl['year_added']   = df_cl['year_added'].fillna(2019)

    features = ['type_enc', 'rating_enc', 'duration_int',
                 'release_year', 'year_added']
    X = df_cl[features].fillna(0)

    # Scale features (KMeans is distance-based, scale matters!)
    scaler = sklearn.preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Find optimal K using Elbow Method ────────────────────
    print("\n🔍 Finding optimal number of clusters (Elbow Method)...")
    inertias = []
    K_range  = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"   K={k} → Inertia: {km.inertia_:.0f}")

    # Use K=5 (good balance for Netflix content types)
    OPTIMAL_K = 5
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    df_cl['cluster'] = kmeans.fit_predict(X_scaled)

    # Describe each cluster
    print(f"\n📊 Cluster Analysis (K={OPTIMAL_K}):")
    cluster_summary = df_cl.groupby('cluster').agg({
        'type'        : lambda x: x.value_counts().index[0],
        'rating'      : lambda x: x.value_counts().index[0],
        'release_year': 'mean',
        'duration_int': 'mean',
        'title'       : 'count'
    }).rename(columns={'title': 'count', 'type': 'dominant_type'})
    print(cluster_summary.to_string())

    # ── VISUALIZATION: Elbow Curve + Cluster Distribution ─────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ML Model 2: KMeans Content Clustering',
                 fontsize=16, color='white', fontweight='bold')

    # Elbow curve
    axes[0].plot(list(K_range), inertias, color=NETFLIX_RED,
                 linewidth=2.5, marker='o', markersize=7)
    axes[0].axvline(OPTIMAL_K, color=ACCENT, linestyle='--',
                    linewidth=1.5, label=f'Optimal K={OPTIMAL_K}')
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia (lower = tighter clusters)', fontsize=12)
    axes[0].set_title('Elbow Method', color='white', fontsize=13)
    axes[0].legend(facecolor='#1a1a1a', labelcolor='white')
    axes[0].grid(alpha=0.3)

    # Cluster size distribution
    cluster_sizes = df_cl['cluster'].value_counts().sort_index()
    cluster_labels = [f'Cluster {i}' for i in cluster_sizes.index]
    colors_cl = [NETFLIX_RED, COLOR_SHOW, ACCENT, '#9B59B6', '#3498DB']
    axes[1].bar(cluster_labels, cluster_sizes.values,
                color=colors_cl, edgecolor='#0F0F0F', linewidth=1)
    for i, val in enumerate(cluster_sizes.values):
        axes[1].text(i, val + 20, str(val), ha='center',
                     color='white', fontsize=10)
    axes[1].set_ylabel('Number of Titles', fontsize=12)
    axes[1].set_title('Titles per Cluster', color='white', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig('11_clustering_results.png')

    return kmeans, df_cl


# ═══════════════════════════════════════════════════════════════
# ML MODEL 3: TF-IDF CONTENT RECOMMENDER
# Goal: Given a title, find the 5 most similar Netflix titles
#       using their descriptions (NLP technique)
#
# TF-IDF = Term Frequency - Inverse Document Frequency
# It turns text descriptions into numbers we can compare
# ═══════════════════════════════════════════════════════════════
def build_recommender(df):
    print("\n" + "="*55)
    print("🤖 ML MODEL 3: TF-IDF Content Recommender")
    print("="*55)

    # Combine multiple text fields for richer matching
    df_rec = df.copy()
    df_rec['content_soup'] = (
        df_rec['title'].fillna('') + ' ' +
        df_rec['director'].fillna('') + ' ' +
        df_rec['listed_in'].fillna('') + ' ' +
        df_rec['description'].fillna('') + ' ' +
        df_rec['primary_country'].fillna('')
    )

    print("🔨 Building TF-IDF matrix...")
    # TfidfVectorizer converts text to a numerical matrix
    # stop_words='english' removes common words like "the", "is", "and"
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000,
                             ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df_rec['content_soup'])
    print(f"   Matrix shape: {tfidf_matrix.shape}")
    print(f"   (rows=titles, cols=unique words/phrases)")

    # Compute cosine similarity between all pairs of titles
    # Cosine similarity: 1.0 = identical, 0.0 = completely different
    print("🔨 Computing cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"   Similarity matrix: {cosine_sim.shape}")

    # Create a mapping from title name → index
    indices = pd.Series(df_rec.index, index=df_rec['title']).drop_duplicates()

    def get_recommendations(title, n=5):
        """
        Given a Netflix title, return the N most similar titles.
        
        How it works:
        1. Find the title's index
        2. Get its similarity scores with all other titles
        3. Sort by similarity (highest first)
        4. Return top N titles
        """
        if title not in indices:
            return f"Title '{title}' not found in dataset."

        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]  # skip index 0 (the title itself)

        title_indices = [i[0] for i in sim_scores]
        scores        = [round(i[1], 4) for i in sim_scores]

        result = df_rec.iloc[title_indices][['title', 'type', 'listed_in',
                                              'release_year', 'primary_country']].copy()
        result['similarity_score'] = scores
        return result

    # Test with 3 example titles
    test_titles = ['Stranger Things', 'The Dark Knight', 'Breaking Bad']

    print("\n🎬 Testing Recommender System:\n")
    for title in test_titles:
        print(f"  📽️  Similar to '{title}':")
        recs = get_recommendations(title, n=5)
        if isinstance(recs, pd.DataFrame):
            for _, row in recs.iterrows():
                print(f"      → {row['title']} ({row['type']}, {int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'}) "
                      f"| Score: {row['similarity_score']:.4f}")
        else:
            print(f"      {recs}")
        print()

    # ── VISUALIZATION: Similarity heatmap for sample titles ──
    sample_titles_for_plot = []
    for t in test_titles:
        if t in indices:
            sample_titles_for_plot.append(t)
            recs = get_recommendations(t, n=3)
            if isinstance(recs, pd.DataFrame):
                sample_titles_for_plot.extend(recs['title'].tolist())

    # Remove duplicates, keep first 12
    seen = set()
    unique_sample = []
    for t in sample_titles_for_plot:
        if t not in seen:
            seen.add(t)
            unique_sample.append(t)
    unique_sample = unique_sample[:12]

    if len(unique_sample) >= 4:
        sample_idx = [indices[t] for t in unique_sample if t in indices]
        sim_subset = cosine_sim[np.ix_(sample_idx, sample_idx)]
        labels_short = [t[:20] for t in unique_sample if t in indices]

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('TF-IDF Similarity Heatmap — Content Clusters',
                     fontsize=15, color='white', fontweight='bold')
        sns.heatmap(sim_subset, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=labels_short, yticklabels=labels_short,
                    ax=ax, linewidths=0.5, linecolor='#0F0F0F',
                    annot_kws={'size': 8})
        ax.tick_params(colors='white', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_fig('12_similarity_heatmap.png')

    return get_recommendations, cosine_sim, indices


# ═══════════════════════════════════════════════════════════════
# MASTER FUNCTION — run all ML models
# ═══════════════════════════════════════════════════════════════
def run_all_ml(df):
    print("\n" + "="*55)
    print("🤖 STARTING MACHINE LEARNING PIPELINE")
    print("="*55)

    classifier, results  = build_classifier(df)
    kmeans_model, df_cl  = build_clustering(df)
    recommender, sim_mat, idx = build_recommender(df)

    print("\n" + "="*55)
    print("✅ ALL ML MODELS COMPLETE!")
    print("   Artifacts saved to /visualizations/")
    print("="*55)

    return classifier, kmeans_model, recommender


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader  import load_netflix_data
    from data_cleaning import clean_netflix_data

    df_raw   = load_netflix_data()
    df_clean = clean_netflix_data(df_raw)
    run_all_ml(df_clean)