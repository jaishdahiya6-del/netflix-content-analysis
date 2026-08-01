import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kurtosis, skew

# Ensure project root is in path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_loader import load_netflix_data
from src.data_cleaning import clean_netflix_data

st.set_page_config(page_title="Machine Learning Pipeline", page_icon="🤖", layout="wide")

# Helper for caching loading and cleaning
@st.cache_data
def load_and_clean_dataset():
    df_raw = load_netflix_data()
    df_clean = clean_netflix_data(df_raw)
    return df_clean

df = load_and_clean_dataset()

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

st.markdown('<div class="section-title">🤖 Machine Learning Pipeline & Statistical Insights</div>', unsafe_allow_html=True)

# Tabs
tab_cls, tab_clu, tab_stats = st.tabs(["🎯 Content Type Classifier", "🧩 KMeans Clustering", "📈 Statistical Audits"])

# -------------------------------------------------------------
# TAB 1: CLASSIFICATION
# -------------------------------------------------------------
with tab_cls:
    st.markdown("### 🎯 Content Type Classification Model")
    st.write("""
        This machine learning model predicts whether a Netflix title is a **Movie** or a **TV Show**
        based on five key features: release year, rating, duration, country, and the year it was added to the platform.
        We compare three algorithms: **Random Forest**, **Logistic Regression**, and **Gradient Boosting**.
    """)

    col_cls1, col_cls2 = st.columns([1, 1])

    with col_cls1:
        st.subheader("📊 Classifier Performance")
        # Display saved classifier results image if exists, or show static results
        if os.path.exists("visualizations/09_classifier_results.png"):
            st.image("visualizations/09_classifier_results.png", caption="Model Comparison & Confusion Matrix", use_container_width=True)
        else:
            st.info("Performance stats: All models achieve near 100% test accuracy because duration format uniquely correlates with the content type (minutes vs seasons).")

        if os.path.exists("visualizations/10_feature_importance.png"):
            st.image("visualizations/10_feature_importance.png", caption="Feature Importance Analysis", use_container_width=True)

    with col_cls2:
        st.subheader("🔮 Interactive Predictor")
        st.write("Train a Random Forest classifier in real-time and predict any title's content type!")

        # Train simple RF on the fly for interactive prediction
        df_ml = df.copy()

        # Fit Label Encoders
        le_rating = LabelEncoder()
        df_ml['rating_enc'] = le_rating.fit_transform(df_ml['rating'].fillna('Unknown'))

        le_country = LabelEncoder()
        df_ml['country_enc'] = le_country.fit_transform(df_ml['primary_country'].fillna('Unknown'))

        features = ['release_year', 'rating_enc', 'duration_int', 'country_enc', 'year_added']
        X = df_ml[features]
        y = df_ml['type']

        # Train RF
        @st.cache_resource
        def train_rf_model(_X, _y):
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(_X, _y)
            return rf

        rf_model = train_rf_model(X, y)

        # Interactive Inputs
        input_release_year = st.slider("Release Year", 1925, 2021, 2018)
        input_rating = st.selectbox("Rating Category", list(le_rating.classes_))
        input_duration = st.number_input("Duration Metric (mins or seasons)", min_value=1, max_value=400, value=95)
        input_country = st.selectbox("Primary Country", list(le_country.classes_))
        input_year_added = st.slider("Year Added to Netflix", 2008, 2021, 2019)

        if st.button("🚀 Predict Content Type", type="primary"):
            # Prepare inputs
            rat_enc = int(np.where(le_rating.classes_ == input_rating)[0][0])
            cnt_enc = int(np.where(le_country.classes_ == input_country)[0][0])

            sample = np.array([[input_release_year, rat_enc, input_duration, cnt_enc, input_year_added]])
            pred = rf_model.predict(sample)[0]
            probs = rf_model.predict_proba(sample)[0]

            # Show output
            st.markdown(f"### Predicted: **{pred}**")
            # Probabilities
            prob_df = pd.DataFrame({
                'Class': rf_model.classes_,
                'Probability': probs
            })
            fig_prob = px.bar(prob_df, x='Probability', y='Class', orientation='h', color='Class',
                              color_discrete_map={'Movie': '#E50914', 'TV Show': '#F5A623'},
                              range_x=[0, 1], title="Prediction Probabilities")
            fig_prob.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_prob, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: CLUSTERING
# -------------------------------------------------------------
with tab_clu:
    st.markdown("### 🧩 KMeans Content Clustering")
    st.write("""
        Clustering groups titles into 5 natural clusters using their core attributes.
        This reveals high-level structures in Netflix's library content strategy.
    """)

    col_clu1, col_clu2 = st.columns([1, 1])

    with col_clu1:
        if os.path.exists("visualizations/11_clustering_results.png"):
            st.image("visualizations/11_clustering_results.png", caption="KMeans Elbow Curve & Cluster Distributions", use_container_width=True)
        else:
            st.info("KMeans clusters the Netflix catalog into 5 groups representing distinct content libraries (e.g., modern adult movies, multi-season TV shows, retro titles, etc.).")

    with col_clu2:
        st.subheader("📋 Cluster Descriptions & Interpretation")
        st.markdown("""
        Based on our K=5 analysis:
        - **Cluster 0**: *Mainstream Adult Movies* — dominated by Movies with `TV-MA` ratings, modern release dates, and standard movie durations (average ~90 mins).
        - **Cluster 1**: *The Binge-watch Catalog* — dominated by `TV-MA` rated multi-season TV Shows (average ~1.7 seasons) added after 2016.
        - **Cluster 2**: *Retro Classics & Epics* — dominated by classic movies and TV-14 ratings, released predominantly between 1970 and 2000, with longer runtimes (average ~113 mins).
        - **Cluster 3**: *Modern Teen & Family Movies* — dominated by Movies with `TV-14` or `PG-13` ratings, with longer durations (average ~108 mins).
        - **Cluster 4**: *Highly Curated Modern Originals* — modern titles with premium production value, added to Netflix recently, dominated by `TV-MA` Movies.
        """)

        st.write("Below is a sample of titles from a selected cluster:")
        cluster_id = st.selectbox("Select Cluster ID", [0, 1, 2, 3, 4], index=1)

        # Simple clustering assignment on the fly for viewing
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        @st.cache_data
        def get_clustered_df():
            df_cl = df.copy()
            le = LabelEncoder()
            df_cl['rating_enc'] = le.fit_transform(df_cl['rating'].fillna('Unknown'))
            df_cl['type_enc'] = (df_cl['type'] == 'Movie').astype(int)

            features_clu = ['type_enc', 'rating_enc', 'duration_int', 'release_year', 'year_added']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cl[features_clu].fillna(0))

            km = KMeans(n_clusters=5, random_state=42, n_init=10)
            df_cl['cluster'] = km.fit_predict(X_scaled)
            return df_cl

        df_cl = get_clustered_df()
        cluster_sample = df_cl[df_cl['cluster'] == cluster_id][['title', 'type', 'rating', 'release_year', 'duration']].head(10)
        st.dataframe(cluster_sample, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: STATISTICAL AUDIT
# -------------------------------------------------------------
with tab_stats:
    st.markdown("### 📈 Descriptive Statistical Analysis")
    st.write("""
        We perform robust descriptive statistical analysis on numeric features like the **release year** or **parsed duration**.
        This is critical to establish a solid statistical foundation before building machine learning models.
    """)

    col_stat1, col_stat2 = st.columns([1, 2])

    with col_stat1:
        stat_column = st.selectbox("Choose Numeric Column", ["release_year", "duration_int", "year_added"])
        data_col = df[stat_column].dropna()

        # Calculate descriptive stats
        mean_v = data_col.mean()
        med_v = data_col.median()
        std_v = data_col.std()
        sk_v = skew(data_col)
        kt_v = kurtosis(data_col)

        st.markdown(f"""
        ### 📋 Summary Metrics for `{stat_column}`:
        - **Count**: {len(data_col):,}
        - **Mean**: {mean_v:.2f}
        - **Median**: {med_v:.2f}
        - **Standard Deviation**: {std_v:.2f}
        - **Min**: {data_col.min():.2f}
        - **Max**: {data_col.max():.2f}
        - **Skewness**: {sk_v:.4f} *(indicates {"right/positive" if sk_v > 0 else "left/negative"} skew)*
        - **Kurtosis**: {kt_v:.4f}
        """)

        st.write("""
            **Interpretation**:
            - A negative skewness in `release_year` shows that Netflix content is heavily concentrated in very recent years.
            - A high kurtosis indicates that the distribution has a heavy peak, indicating massive acquisitions in specific time windows.
        """)

    with col_stat2:
        st.subheader("📊 Distribution Curve with Central Tendency Indicators")
        # Plotly Histogram with mean/median lines
        fig_dist = px.histogram(df, x=stat_column, nbins=30, color_discrete_sequence=['#9B59B6'], title=f"Statistical Distribution of '{stat_column}'", marginal="box")

        # Add Mean and Median as lines
        fig_dist.add_vline(x=mean_v, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_v:.2f}", annotation_position="top left")
        fig_dist.add_vline(x=med_v, line_color="green", annotation_text=f"Median: {med_v:.2f}", annotation_position="top right")

        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_dist, use_container_width=True)
