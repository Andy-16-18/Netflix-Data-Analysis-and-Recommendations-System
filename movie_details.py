# movie_details.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import urllib.parse

# Function to preprocess data
def prepare_data(x):
    return str.lower(x.replace(" ", ""))

# Function to create soup for recommendation
def create_soup(x):
    return x['Genre'] + ' ' + x['Tags'] + ' ' +x['Actors']+' '+ x['ViewerRating']

# Function to get recommendations
def get_recommendations(title, cosine_sim, netflix_data, selected_data, indices):
    title=title.replace(' ','').lower()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    movie_indices = [i[0] for i in sim_scores]
    result =  netflix_data.iloc[movie_indices]
    result.reset_index(inplace=True)
    return result

# Read Netflix dataset
netflix_data = pd.read_csv('Netflix_Dataset_Final.csv',index_col='Title')
netflix_data.index = netflix_data.index.str.title()
netflix_data = netflix_data[~netflix_data.index.duplicated()]
netflix_data.rename(columns={'View Rating':'ViewerRating'}, inplace=True)

# Preprocess features
Language = netflix_data.Languages.str.get_dummies(',')
Lang = set(Language.columns.str.strip().values.tolist())
Titles = set(netflix_data.index.to_list())

netflix_data['Genre'] = netflix_data['Genre'].astype('str')
netflix_data['Tags'] = netflix_data['Tags'].astype('str')
netflix_data['IMDb Score'] = netflix_data['IMDb Score'].apply(lambda x: 6.6 if math.isnan(x) else x)
netflix_data['Actors'] = netflix_data['Actors'].astype('str')
netflix_data['ViewerRating'] = netflix_data['ViewerRating'].astype('str')

new_features = ['Genre', 'Tags', 'Actors', 'ViewerRating']
selected_data = netflix_data[new_features]
for new_feature in new_features:
    selected_data.loc[:, new_feature] = selected_data.loc[:, new_feature].apply(prepare_data)
selected_data.index = selected_data.index.str.lower()
selected_data.index = selected_data.index.str.replace(" ",'')
selected_data['soup'] = selected_data.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(selected_data['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
selected_data.reset_index(inplace=True)
indices = pd.Series(selected_data.index, index=selected_data['Title'])
result = 0
df = pd.DataFrame()

# Function to display movie details and recommendations
def display_movie_details(title, genre, summary, netflix_link, image, release_date, viewer_rating, runtime, actors):
    st.title(title)
    st.markdown(
        f'<img src="{image}" alt="Movie Poster" style="width: 250px; height: auto;">',
        unsafe_allow_html=True
    )
    details = []
    if release_date:
        details.append(f"Release Date: {release_date}")
    if viewer_rating:
        details.append(f"Viewer Rating: {viewer_rating}")
    if runtime:
        details.append(f"Runtime: {runtime}")
    if genre:
        details.append(f"Genre: {genre}")
    st.write(" | ".join(details))
    if actors:
        st.write(f"**Starring:** {actors}")
    if summary:
        st.write(summary)
    if netflix_link:
        st.write(f"**Netflix Link:** [{title}]({netflix_link})")

    # Display recommendations
    st.subheader('Recommendations:')
    recommendations = get_recommendations(title, cosine_sim2, netflix_data, selected_data, indices)
    if not recommendations.empty:
        num_cols = 3  # Number of movies per row
        num_rows = (len(recommendations) - 1) // num_cols + 1

        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < len(recommendations):
                    rec_movie = recommendations.iloc[idx]
                    # Create URL link to movie details page
                    params = {
                        "page": "movie_details",
                        "title": rec_movie['Title'],
                        "genre": rec_movie['Genre'],
                        "summary": rec_movie['Summary'],
                        "netflix_link": rec_movie['Netflix Link'],
                        "image": rec_movie['Image'],
                        "release_date": rec_movie['Release Date'],
                        "viewer_rating": rec_movie['ViewerRating'],
                        "runtime": rec_movie['Runtime'],
                        "actors": rec_movie['Actors']
                    }
                    encoded_params = urllib.parse.urlencode(params)
                    movie_details_url = f"?{encoded_params}"

                    # Display clickable image
                    cols[j].markdown(f'<a href="{movie_details_url}"><img src="{rec_movie["Image"]}" width="200"></a>', unsafe_allow_html=True)
    else:
        st.write("No recommendations found for this movie.")
