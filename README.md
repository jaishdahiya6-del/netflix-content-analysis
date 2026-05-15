# 🎬 Netflix Content Analysis & Recommendation Trends

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## 📌 Project Overview
This project performs an in-depth Exploratory Data Analysis (EDA) on a dataset containing over **8,800 Netflix titles**. The goal is to uncover shifts in Netflix's content strategy, identify top-performing genres, and visualize the global distribution of content.

### 🔍 Key Insights
- **Content Shift:** Analysis shows a massive pivot toward TV Shows over Movies starting around 2015.
- **Global Leader:** The United States remains the top producer, followed closely by India for Movie content.
- **Genre Dominance:** International Movies and Dramas represent the largest share of the library.

---
### 📉 Statistical Foundation
To ensure our machine learning models are accurate, I performed a statistical audit:
- **Skewness Analysis:** Identified if the data is right-skewed (common in sales data) to decide on proper normalization techniques.
- **Outlier Impact:** Measured the gap between Mean and Median to understand the influence of high-value outliers.
- **Central Tendency:** Visualized the data distribution to verify if it meets the assumptions of Linear Regression.



[Image of Normal distribution vs skewed distribution curves]
## 📊 Visualizations

### 1. Strategy Shift (Movies vs TV Shows)
This chart tracks how Netflix has changed its library composition over the last decade.
![Content Trends](images/content_trends.png)

### 2. Global Content Distribution
An interactive map showing which countries are the biggest contributors to the Netflix library.
![Global Map](images/global_map.png)
### 📈 Exploratory Data Analysis (EDA)
Beyond basic counts, this phase focused on finding hidden relationships:
- **Feature Correlation:** Utilized Heatmaps to identify strong linear relationships between variables (e.g., Sales vs. Profit).
- **Data Distribution:** Analyzed skewness and kurtosis of key metrics to prepare for future predictive modeling.
- **Segment Analysis:** Grouped data by categories to identify high-value clusters.

![Correlation Heatmap](images/correlation_heatmap.png)
---

## 🛠️ Installation & Usage

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/jaishdahiya6-del/netflix-content-analysis.git](https://github.com/jaishdahiya6-del/netflix-content-analysis.git)
