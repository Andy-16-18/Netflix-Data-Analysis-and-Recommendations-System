
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Read the Netflix dataset
netflix_data = pd.read_csv('Netflix_Dataset_Final.csv')
netflix_data.reset_index(inplace=True)

# Combine 'Genre' and 'Tags' for TF-IDF vectorization
netflix_data['Combined_Text'] = netflix_data['Genre'] + ' ' + netflix_data['Tags']

# Extract numerical features
numerical_features = netflix_data[['IMDb Score', 'IMDb Votes']]

# Standardize numerical features
scaler = StandardScaler()
numerical_features_standardized = scaler.fit_transform(numerical_features)

# TF-IDF Vectorization for textual features
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(netflix_data['Combined_Text'].values.astype('U'))

# Cosine Similarity with TF-IDF
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# k-Nearest Neighbors (KNN) with Numerical and Textual Features
combined_matrix = cosine_similarity(numerical_features_standardized) + cosine_similarity(tfidf_matrix, tfidf_matrix)
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(combined_matrix)

# Add dummy user ids
netflix_data['User'] = range(len(netflix_data))

# Surprise - SVD (Collaborative Filtering)
reader = Reader(rating_scale=(1, 10))

# Load the dataset
data = Dataset.load_from_df(netflix_data[['User', 'Title', 'IMDb Score']], reader)

# Rest of your code remains unchanged
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Improving SVD performance
svd_model = SVD(n_factors=100, reg_all=0.02, lr_all=0.005, n_epochs=30)
svd_model.fit(trainset)
svd_predictions = svd_model.test(testset)

# Generate predictions for k-Nearest Neighbors (KNN) method
distances_knn, indices_knn = knn.kneighbors(combined_matrix)
predictions_knn = distances_knn.mean(axis=1)

# Evaluate accuracy for Surprise - SVD (Collaborative Filtering)
predictions_svd = [pred.est for pred in svd_predictions]
actual_ratings = [rating for (_, _, rating) in testset]

# Define a function to compute accuracy metrics
def evaluate_accuracy(predictions, actual_ratings):
    mae = mean_absolute_error(actual_ratings, predictions)
    rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
    return mae, rmse

# Recommendation Function
def recommend(movie_title):
    # Cosine Similarity with TF-IDF
    movie_index = netflix_data[netflix_data['Title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    sorted_movies_cosine = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # k-Nearest Neighbors (KNN) with Numerical and Textual Features
    movie_index_knn = netflix_data[netflix_data['Title'] == movie_title].index[0]
    distances_knn, indices_knn = knn.kneighbors([combined_matrix[movie_index_knn]])
    distances_knn = distances_knn.flatten()
    indices_knn = indices_knn.flatten()
    sorted_movies_knn = sorted(list(zip(indices_knn, distances_knn)), key=lambda x: x[1])

    # Surprise - SVD (Collaborative Filtering)
    movie_id_svd = trainset.to_inner_iid(movie_title)
    svd_recommendations = []
    for i in range(trainset.n_items):
        if i != movie_id_svd:
            prediction = svd_model.predict(trainset.to_raw_uid(0), trainset.to_raw_iid(i))
            svd_recommendations.append((i, prediction.est))
    svd_recommendations.sort(key=lambda x: x[1], reverse=True)
    svd_recommendations = svd_recommendations[:5]

    # Print recommendations
    print(f"\nTop 5 recommended movies for '{movie_title}' using Cosine Similarity with TF-IDF:")
    for i in range(1, 6):
        print(netflix_data.iloc[sorted_movies_cosine[i][0]].Title)

    print(f"\nTop 5 recommended movies for '{movie_title}' using k-Nearest Neighbors (KNN) with Numerical and Textual Features:")
    for i in range(1, min(6, len(sorted_movies_knn))):  # Ensure we don't exceed the length of sorted_movies_knn
        print(netflix_data.iloc[sorted_movies_knn[i][0]].Title)

    print(f"\nTop 5 recommended movies for '{movie_title}' using Surprise - SVD (Collaborative Filtering):")
    for movie_inner_id_svd, _ in svd_recommendations:
        movie_title_svd = trainset.to_raw_iid(movie_inner_id_svd)
        print(netflix_data[netflix_data['Title'] == movie_title_svd].Title.values[0])

# Example usage
def recommend_and_evaluate(movie_title, actual_ratings):
    # Perform recommendations
    recommend(movie_title)

    # Evaluate accuracy
    mae_knn, rmse_knn = evaluate_accuracy(predictions_knn[:len(actual_ratings)], actual_ratings)
    mae_svd, rmse_svd = evaluate_accuracy(predictions_svd, actual_ratings)

    # Print results
    print("\nKNN - Mean Absolute Error:", mae_knn)
    print("KNN - Root Mean Squared Error:", rmse_knn)
    print("SVD - Mean Absolute Error:", mae_svd)
    print("SVD - Root Mean Squared Error:", rmse_svd)

# Example usage
recommend_and_evaluate("Iron Man", actual_ratings)
