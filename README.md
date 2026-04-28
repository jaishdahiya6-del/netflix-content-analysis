=======
# netflix-content-analysis
=======
# 🎬 Netflix Content Analysis

> End-to-end Data Science project analyzing 8,800+ Netflix titles using Python, Machine Learning, and Business Intelligence.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

This project performs a complete analysis of Netflix's content library to uncover strategic business insights about content strategy, audience targeting, global expansion, and growth trends.

**Dataset:** [Netflix Movies and TV Shows — Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
**Records:** 8,807 titles | **Features:** 12 columns | **Period:** 2008–2021

---

## 🎯 Key Findings

| Insight | Finding |
|---|---|
| Content split | ~70% Movies, ~30% TV Shows |
| Top market | United States dominates production |
| Peak growth | 2019–2020 highest content additions |
| Target audience | 40%+ content is adult-rated (TV-MA, R) |
| Global bet | International content is fastest-growing category |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| Pandas + NumPy | Data manipulation |
| Matplotlib + Seaborn | Visualizations |
| scikit-learn | ML models |
| TF-IDF (NLP) | Content recommender |
| KMeans | Content clustering |
| Jupyter Notebook | Interactive analysis |

---

## 📊 Visualizations (12 Charts)

| # | Chart | Insight |
|---|---|---|
| 1 | Content Distribution | Movie vs TV Show split |
| 2 | Yearly Growth | Library growth 2008–2021 |
| 3 | Top Genres | Most popular categories |
| 4 | Country Distribution | Top content-producing nations |
| 5 | Ratings Analysis | Audience targeting by rating |
| 6 | Duration Distribution | Movie length & show seasons |
| 7 | Monthly Heatmap | When Netflix adds content |
| 8 | Top Directors | Most prolific directors |
| 9 | ML Classifier Results | Model accuracy comparison |
| 10 | Feature Importance | What predicts content type |
| 11 | Clustering Results | KMeans elbow + cluster sizes |
| 12 | Similarity Heatmap | TF-IDF content similarity |

---

## 🤖 Machine Learning Models

### 1. Content Type Classifier
- **Goal:** Predict Movie vs TV Show from metadata
- **Models:** Random Forest, Logistic Regression, Gradient Boosting
- **Best accuracy:** ~85%+

### 2. KMeans Clustering
- **Goal:** Discover hidden content groupings
- **Method:** KMeans with elbow method (K=5)
- **Use case:** Content tagging automation

### 3. TF-IDF Content Recommender
- **Goal:** "If you liked X, you'll like Y"
- **Method:** TF-IDF vectorization + cosine similarity
- **Input:** Any Netflix title → Output: 5 similar titles

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/netflix-content-analysis.git
cd netflix-content-analysis

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset
# Download netflix_titles.csv from Kaggle and place in /data/

# 5. Run the full pipeline
python run_project.py
```

---

## 📁 Project Structure
---

## 💼 Business Insights

1. **Retention:** Invest in multi-season TV Shows to drive binge-watching
2. **Global:** Regional originals (India, Korea) have disproportionate global appeal
3. **Family:** Expanding kids content reduces household churn

---

## 👤 Author

**Jaish Dahiya**  
Data Science Portfolio Project  
[GitHub](https://github.com/YOUR_USERNAME) | [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

MIT License — feel free to use this project for learning and portfolio purposes.
>>>>>>> 987026b (Initial commit: Netflix Content Analysis with EDA, ML models, and Business Insights)
