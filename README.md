# 🎬 Netflix Content Analysis & Smart Recommendation Dashboard

An interactive, production-ready full web app/dashboard built with **Streamlit**, **Plotly**, and **Scikit-Learn** to perform deep-dive exploratory data analysis, descriptive statistical audits, and machine learning insights on over **8,800 Netflix titles**.

---

## 🚀 Key Features

### 1. 📊 Interactive Exploratory Data Analysis (EDA)
- **Deep-Dive Filters**: Filter the Netflix dataset in real-time by Content Type (Movie vs. TV Show), Release Year, Ratings, Genres/Categories, and Primary Country of production.
- **Top Genres Distribution**: Interactive bar charts highlighting dominant content categories.
- **Library Growth Trends**: Dynamic line/area charts visualizing library expansions over time.
- **Ratings & Geographic Insights**: Explore content strategy and audience targeting.
- **Parsed Duration Analysis**: Automatically parses and models Movie runtimes (in minutes) and TV Show lifespans (in seasons).

### 2. 🤖 Interactive Machine Learning Pipeline
- **Content Type Classifier**: Train an on-the-fly **Random Forest** model using attributes like release year, rating, country, and year added. Input custom features to predict in real-time if a title will be a Movie or a TV Show.
- **K-Means Content Clustering**: Groups the catalog into 5 distinct cluster profiles (e.g., retro classics, mainstream adult movies, teen family movies, etc.) with detailed breakdowns.
- **Descriptive Statistical Audits**: Computes and plots skewness, kurtosis, standard deviations, and central tendencies (mean/median indicators) for any selected numeric metric.

### 3. 🎬 Natural Language Content Recommender
- **TF-IDF Vectorization**: Parses word soups combining titles, descriptions, directors, and genre tags to build a sparse NLP feature matrix.
- **Cosine Similarity Engine**: Quantifies likeness between all titles in the catalog.
- **Interactive Matching**: Type or select any Netflix title from a dropdown of over 8,800 to instantly see the Top 5 recommendations with percentage match scores.
- **Similarity Heatmap**: Pairwise similarity heatmap of recommended titles to visualize semantic clusters.

---

## 📂 Project Structure

```
netflix-content-analysis/
├── data/
│   ├── netflix_titles.csv         # Raw Netflix dataset (8,807 titles)
│   └── cleaned_netflix_data.csv   # Preprocessed/Cleaned dataset with engineered features
├── images/                        # Exported static analysis plots
├── pages/                         # Multi-page Streamlit views
│   ├── 01_📊_Exploratory_Data_Analysis.py
│   ├── 02_🤖_Machine_Learning.py
│   └── 03_🎬_Smart_Recommender.py
├── src/                           # Analytical core modules
│   ├── data_loader.py             # Robust data ingestion module
│   ├── data_cleaning.py           # Handles missing values, duplicates, feature engineering
│   ├── eda.py                     # Portfolio-grade EDA pipeline
│   ├── insights.py                # Descriptive statistics and distribution reports
│   ├── ml_models.py               # Classification, clustering, and TF-IDF engines
│   └── recommender.py             # Standalone recommendation logic
├── visualizations/                # Automated ML pipeline exports
├── app.py                         # Streamlit landing page & executive dashboard
├── requirements.txt               # Maintained dependency definitions
├── run_project.py                 # Core business logic unit tests
└── README.md                      # Setup and usage guide
```

---

## 🛠️ Local Installation & Setup

Follow these simple steps to run the dashboard locally in your environment:

### 1. Clone the repository:
```bash
git clone https://github.com/jaishdahiya6-del/netflix-content-analysis.git
cd netflix-content-analysis
```

### 2. Install dependencies:
Make sure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run unit tests to verify:
```bash
python run_project.py
```

### 4. Run the core pipelines to generate static visualizations:
```bash
python src/data_cleaning.py
python src/eda.py
python src/insights.py
python src/ml_models.py
```

### 5. Start the Interactive Web Dashboard:
```bash
streamlit run app.py
```
This will spin up a local development server and open the web application automatically in your browser at `http://localhost:8501`.

---

## 🤝 Key Insights & Recommendations
- **TV Show Transition**: Analysis shows Netflix began focusing heavily on multi-season TV Shows starting around 2015 to foster binge-watching habits and reduce user churn.
- **Global Strategy**: While the United States remains the largest producer, regional originals (especially in India, South Korea, and Spain) represent the fastest-growing categories.
- **Demographics Focus**: Adults (TV-MA, R) represent approximately 46% of the content library, serving premium paying demographics. Expanding family categories presents a key opportunity to reduce household subscription churn.
