import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(df, title):
    # 1. Combine features into a single 'soup' of words
    df['content_features'] = (df['description'] + " " + 
                             df['listed_in'] + " " + 
                             df['cast']).fillna('')

    # 2. Vectorize the text data
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_features'])

    # 3. Compute the similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 4. Find the movie index and return top 5 matches
    try:
        idx = df[df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get indices of top 5 similar movies (excluding itself)
        movie_indices = [i[0] for i in sim_scores[1:6]]
        return df['title'].iloc[movie_indices].tolist()
    except IndexError:
        return "Title not found in dataset."

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Ensure you have your cleaned data file ready
    data = pd.read_csv('cleaned_netflix_data.csv')
    print(f"Recommendations for 'Peaky Blinders':")
    print(get_recommendations(data, 'Peaky Blinders'))
