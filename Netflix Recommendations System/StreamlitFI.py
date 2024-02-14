import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Netflix dataset
netflix_data = pd.read_csv('Netflix_Dataset_Final.csv')

# Function to filter movies based on language
def filter_movies(language):
    filtered_data = netflix_data.copy()
    # Apply language filter
    if language != 'All':
        filtered_data = filtered_data[filtered_data['Languages'] == language]
    return filtered_data

# Function to recommend similar movies or series based on selected movie
def recommend_movies(movie_title, filtered_options):
    selected_movie = filtered_options[filtered_options['Title'] == movie_title]
    if selected_movie.empty:
        return "Movie with title '{}' not found in the filtered options.".format(movie_title)
    # Combine genre tags, summary, and tags columns for feature extraction
    filtered_options['Features'] = filtered_options['Genre'] + ' ' + filtered_options['Tags'] + ' ' + filtered_options['Summary']
    # Extract features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(filtered_options['Features'])
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = selected_movie.index[0]
    if idx >= len(cosine_sim):
        return "No recommendations found for the selected movie."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = filtered_options.iloc[movie_indices]
    if recommended_movies.empty:
        return "No recommendations found for the selected movie."
    return recommended_movies

# Streamlit app
def main():
    # App title
    st.title('Netflix Recommendation')

    # Sidebar for user input
    with st.sidebar:
        st.subheader('Filter Options:')
        language = st.selectbox('Language', ['All'] + list(netflix_data['Languages'].unique()))

    # Filtered options
    filtered_options = filter_movies(language)

    # If no movies match the filters, display a warning message
    if filtered_options.empty:
        st.warning("No movies match the selected filters.")
    else:
        # Display filtered movie titles
        with st.sidebar:
            st.subheader('Select Movie:')
            movie_titles = filtered_options['Title'].unique()
            movie_title = st.selectbox('Movie Title', movie_titles)

        # Recommend movies
        recommendation_result = recommend_movies(movie_title, filtered_options)

        # Display recommended movies or message
        if isinstance(recommendation_result, pd.DataFrame):
            st.subheader('Recommended Movies:')
            num_cols = 3  # Number of movies per row
            num_rows = (len(recommendation_result) - 1) // num_cols + 1

            for i in range(num_rows):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i * num_cols + j
                    if idx < len(recommendation_result):
                        movie = recommendation_result.iloc[idx]
                        cols[j].markdown(
                            f'<a href="{movie["Netflix Link"]}" target="_blank"><img src="{movie["Image"]}" width="200"></a>',
                            unsafe_allow_html=True
                        )
        else:
            st.warning(recommendation_result)

if __name__ == '__main__':
    main()
