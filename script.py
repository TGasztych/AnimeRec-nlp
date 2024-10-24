import json
import xml.etree.ElementTree as ET
import pandas as pd
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Variables
source_list = 'animelist_Nomori'
minimal_score = 7

# Load the anime dataset
anime_df = pd.read_csv('data/anime_maldump.csv')
anime_df['synopsis'] = anime_df['synopsis'].fillna('')

# Synopsis cleanup
def clean_synopsis(text):
    text = re.sub(r'\[Written by MAL Rewrite\]', '', text)
    text = re.sub(r'\(Source: [^\)]+\)', '', text)
    if text.count('.') <= 1:
        return ''
    return text

anime_df['synopsis'] = anime_df['synopsis'].apply(clean_synopsis)

# Filtering
anime_df = anime_df[~anime_df['genres'].str.contains('Hentai', na=False)]
anime_df = anime_df[~anime_df['type'].str.contains('music', na=False)]
anime_df = anime_df[~anime_df['type'].str.contains('special', na=False)]
anime_df = anime_df[anime_df['score'] >= minimal_score]
anime_df = anime_df[anime_df['synopsis'].str.strip().ne('')]

# Initialize RAKE with NLTK stopwords
rake = Rake(stopwords=stopwords.words('english'))

# Extract keywords using RAKE
def extract_keywords(text):
    rake.extract_keywords_from_text(text)
    return ' '.join(rake.get_ranked_phrases())

anime_df['keywords'] = anime_df['synopsis'].apply(extract_keywords)

# Create TF-IDF and BoW Vectorizers for keywords
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
bow_vectorizer = CountVectorizer(stop_words='english')

# Apply TF-IDF and BoW to the combined text
tfidf_matrix = tfidf_vectorizer.fit_transform(anime_df['keywords'])
bow_matrix = bow_vectorizer.fit_transform(anime_df['keywords'])

# Load and process the watched anime XML data
tree = ET.parse('data/'+ source_list + '.xml')
root = tree.getroot()

watched_series = []
for anime in root.findall('anime'):
    series_id = anime.find('series_animedb_id').text
    series_title = anime.find('series_title').text
    my_status = anime.find('my_status').text
    my_score = anime.find('my_score').text

    if my_status == 'Completed':
        anime_row = anime_df.loc[anime_df['anime_id'] == int(series_id)]
        if not anime_row.empty:
            synopsis = anime_row['synopsis'].values[0]
            if pd.notnull(synopsis):
                watched_series.append({
                    'series_id': series_id,
                    'series_title': series_title,
                    'my_status': my_status,
                    'my_score': my_score,
                    'synopsis': synopsis
                })

# Save watched anime data to JSON
with open('data/anime_data.json', 'w', encoding='utf-8') as f:
    json.dump(watched_series, f, ensure_ascii=False, indent=4)

# Extract the list of watched series data
watched_ids = {int(series['series_id']) for series in watched_series}
watched_synopses = {}
watched_scores = {}
for series in watched_series:
    series_id = int(series['series_id'])
    watched_synopses[series_id] = series['synopsis']
    watched_scores[series_id] = float(series['my_score'])

# Customizable weight for each score
score_weights = {
    10: 100,
    9: 20,
    8: 10,
    7: 5,
    6: 2,
    5: -1,
    4: -5,
    3: -20,
    2: -50,
    1: -100,
    0: 1
}

# Weighted TF-IDF and BoW vectors for watched anime
watched_tfidf_vectors = []
watched_bow_vectors = []
for series_id, synopsis in watched_synopses.items():
    tfidf_matrix_watched = tfidf_vectorizer.transform([synopsis])
    bow_matrix_watched = bow_vectorizer.transform([synopsis])
    weight = score_weights.get(int(watched_scores[series_id]), 1.0)
    tfidf_matrix_watched = tfidf_matrix_watched.multiply(weight)
    bow_matrix_watched = bow_matrix_watched.multiply(weight)
    watched_tfidf_vectors.append(tfidf_matrix_watched)
    watched_bow_vectors.append(bow_matrix_watched)

# Aggregate the weighted TF-IDF and BoW vectors
watched_tfidf = sum(watched_tfidf_vectors) if watched_tfidf_vectors else None
watched_bow = sum(watched_bow_vectors) if watched_bow_vectors else None

# Combine TF-IDF and BoW vectors
def combine_vectors(vectors):
    return np.hstack(vectors)

# Combine vectors for all anime
combined_vectors = combine_vectors([tfidf_matrix.toarray(), bow_matrix.toarray()])
# Combine vectors for watched anime
watched_combined = combine_vectors([watched_tfidf.toarray(), watched_bow.toarray()]) if watched_tfidf is not None and watched_bow is not None else None

# Compute cosine similarity between watched synopses and all other anime
watched_cosine_sim = cosine_similarity(watched_combined, combined_vectors).flatten() if watched_combined is not None else np.zeros(combined_vectors.shape[0])

# Function to get recommendations excluding watched series by ID
def get_recommendations(watched_cosine_sim=watched_cosine_sim):
    sim_scores = list(enumerate(watched_cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [i for i in sim_scores if anime_df['anime_id'].iloc[i[0]] not in watched_ids]
    recommendations = [
        (rank + 1, anime_df['anime_id'].iloc[i[0]], anime_df['title'].iloc[i[0]], i[1])
        for rank, i in enumerate(sim_scores)
    ]
    return recommendations

# Get and save recommendations
recommendations = get_recommendations()
recommendations_df = pd.DataFrame(recommendations, columns=['Rank', 'ID', 'Title', 'Similarity'])

# Save the full recommendations to a text file
with open('data/recommendations.txt', 'w', encoding='utf-8') as f:
    f.write(recommendations_df.to_string(index=False))

print(recommendations_df.head(10))
