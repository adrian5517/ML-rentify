import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def train_kmeans(data, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    return model

def get_kmeans_labels(model, data):
    return model.predict(data)

def knn_recommend(data, input_point, k=5):
    model = NearestNeighbors(n_neighbors=k)
    model.fit(data)
    distances, indices = model.kneighbors([input_point])
    return indices[0].tolist()
