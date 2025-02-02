import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Step 1: Load the Dataset
# Replace 'path_to_dataset' with the actual path to the downloaded dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Display the first few rows of the datasets
print(movies.head())
print(ratings.head())

# Step 2: Preprocess the Data
# Merge movies and ratings data
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
user_item_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')
user_item_matrix.fillna(0, inplace=True)

# Convert the matrix to a sparse format
user_item_sparse = csr_matrix(user_item_matrix.values)

# Step 3: Build the Recommendation System
# Use K-Nearest Neighbors (KNN) for collaborative filtering
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(user_item_sparse)

# Step 4: Generate Recommendations
def recommend_movies(user_id, n_recommendations=5):
    # Get the user's ratings
    user_ratings = user_item_matrix.iloc[user_id - 1].values.reshape(1, -1)

    # Find the nearest neighbors
    distances, indices = knn.kneighbors(user_ratings, n_neighbors=n_recommendations + 1)

    # Get the movie titles
    recommendations = []
    for i in range(1, len(distances.flatten())):
        movie_title = user_item_matrix.columns[indices.flatten()[i]]
        recommendations.append(movie_title)

    return recommendations

# Example: Recommend movies for user with userId = 1
user_id = 1
recommendations = recommend_movies(user_id)
print(f"Recommended movies for user {user_id}:")
for movie in recommendations:
    print(movie)

# Step 5: Evaluate the Recommendation System (Optional)
# Split the data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Create a training user-item matrix
train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating')
train_matrix.fillna(0, inplace=True)

# Train the KNN model on the training data
knn_train = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn_train.fit(csr_matrix(train_matrix.values))

# Evaluate on the test data
def predict_rating(user_id, movie_id):
    # Get the user's ratings
    user_ratings = train_matrix.iloc[user_id - 1].values.reshape(1, -1)

    # Find the nearest neighbors
    distances, indices = knn_train.kneighbors(user_ratings, n_neighbors=20)

    # Calculate the predicted rating as the average of neighbors' ratings
    neighbor_ratings = train_matrix.iloc[indices.flatten()][movie_id]
    predicted_rating = neighbor_ratings.mean()

    return predicted_rating

# Example: Predict the rating for user 1 and movie 1
user_id = 1
movie_id = 1
predicted_rating = predict_rating(user_id, movie_id)
print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating}")

# Step 6: Calculate RMSE (Optional)
# Evaluate the model on the test data
test_data['predicted_rating'] = test_data.apply(lambda row: predict_rating(row['userId'], row['movieId']), axis=1)
rmse = np.sqrt(mean_squared_error(test_data['rating'], test_data['predicted_rating']))
print(f"RMSE: {rmse}")
