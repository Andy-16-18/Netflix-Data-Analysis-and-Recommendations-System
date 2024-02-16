import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import urllib.parse
from movie_details import display_movie_details  # Importing the function from movie_details.py

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

# Streamlit app
def main():
    
    st.sidebar.image('logo.svg', width=120)

    st.title('Netflix Recommendation System')
    
    # Check if movie details page is requested
    page = st.query_params.get("page", "")

    if page == "movie_details":
        title = st.query_params.get("title", "")
        genre = st.query_params.get("genre", "")
        summary = st.query_params.get("summary", "")
        netflix_link = st.query_params.get("netflix_link", "")
        image = st.query_params.get("image", "")
        release_date = st.query_params.get("release_date", "")  
        viewer_rating = st.query_params.get("viewer_rating", "")  
        runtime = st.query_params.get("runtime", "") 
        actors = st.query_params.get("actors", "")  
        
        # Display movie details using the imported function
        display_movie_details(title, genre, summary, netflix_link, image, release_date, viewer_rating, runtime, actors)
        
    else:
        # Selecting movies
        movienames = st.sidebar.multiselect('Select Movie:', Titles)
        languages = st.sidebar.multiselect('Select Language:', Lang)

        if movienames:
            st.write("Selected Movies:")
            for moviename in movienames:
                st.write(moviename)
                # Fetch details of the selected movie and display them here
                movie_details = netflix_data.loc[moviename]
                st.image(movie_details['Image'], caption=f"{moviename} Poster")
                st.write(f"Release Date: {movie_details['Release Date']} | Viewer Rating: {movie_details['ViewerRating']} | Runtime: {movie_details['Runtime']} | Genre: {movie_details['Genre']}")
                st.write(f"Starring: {movie_details['Actors']}")
                st.write(movie_details['Summary'])
                st.write(f"[{'Watch Here'}]({movie_details['Netflix Link']})")
                # st.write("Netflix Link:", movie_details['Netflix Link'])
                st.write("---")

        if not languages:  # Check if languages list is empty
            # If no languages selected, recommend movies in any language
            df = pd.DataFrame()
            if st.sidebar.button('Get Recommendations'):
                for moviename in movienames:
                    result = get_recommendations(moviename, cosine_sim2, netflix_data, selected_data, indices)
                    # Exclude the selected movie from the recommendations
                    result = result[~result['Title'].isin(movienames)]
                    df = pd.concat([result, df], ignore_index=True)
                    
        else:
            # If languages selected, filter movies by selected languages
            df = pd.DataFrame()
            if st.sidebar.button('Get Recommendations'):
                for moviename in movienames:
                    result = get_recommendations(moviename, cosine_sim2, netflix_data, selected_data, indices)
                    for language in languages:
                        # Exclude the selected movie from the recommendations
                        result = result[~result['Title'].isin(movienames)]
                        df = pd.concat([result[result['Languages'].str.count(language) > 0], df], ignore_index=True)
                    
        df.drop_duplicates(keep='first', inplace=True)

        if df.empty:
            st.write("No recommendations found.")
        else:
            df.sort_values(by='IMDb Score', ascending=False, inplace=True)
            images = df['Image'].tolist()

            st.subheader('Recommendations:')
            num_cols = 3  # Number of movies per row
            num_rows = (len(images) - 1) // num_cols + 1

            for i in range(num_rows):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i * num_cols + j
                    if idx < len(images):
                        # Create URL link to movie details page
                        params = {
                            "page": "movie_details",
                            "title": df.iloc[idx]['Title'],
                            "genre": df.iloc[idx]['Genre'],
                            "summary": df.iloc[idx]['Summary'],
                            "netflix_link": df.iloc[idx]['Netflix Link'],
                            "image": df.iloc[idx]['Image'],
                            "release_date": df.iloc[idx]['Release Date'],
                            "viewer_rating": df.iloc[idx]['ViewerRating'],
                            "runtime": df.iloc[idx]['Runtime'],
                            "actors": df.iloc[idx]['Actors']
                        }
                        encoded_params = urllib.parse.urlencode(params)
                        movie_details_url = f"?{encoded_params}"
                        cols[j].markdown(f'<a href="{movie_details_url}"><img src="{images[idx]}" width="200"></a>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
