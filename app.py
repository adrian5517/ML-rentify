from flask import Flask, request, jsonify
from pymongo import MongoClient, errors
from preprocess import preprocess_data
from ml_model import train_kmeans, knn_recommend, summarize_budget_groups
import pandas as pd
import os
from dotenv import load_dotenv
from flask_cors import CORS
import logging
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache for scaler to maintain consistency
scaler_cache = None

def get_mongo_collection():
    try:
        mongo_uri = os.getenv("MONGO_URI")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        client.server_info()  # Force connection
        logging.info("✅ Successfully connected to MongoDB")
        db = client['test']
        return db['properties']
    except errors.ServerSelectionTimeoutError as err:
        logging.error("❌ MongoDB connection failed: %s", err)
        return None

collection = get_mongo_collection()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'service': 'Rentify ML API',
        'version': '2.0',
        'status': 'Production Ready',
        'endpoints': {
            '/': 'GET - API documentation',
            '/check-db': 'GET - Check database connection status',
            '/ml': 'POST - Machine learning predictions (kmeans clustering or knn recommendations)'
        },
        'examples': {
            'kmeans_clustering': {
                'url': '/ml',
                'method': 'POST',
                'body': {
                    'mode': 'kmeans',
                    'n_clusters': 3
                },
                'description': 'Cluster properties into budget categories (Low/Mid/High-End)'
            },
            'knn_recommendations': {
                'url': '/ml',
                'method': 'POST',
                'body': {
                    'mode': 'knn',
                    'price': 3500,
                    'latitude': 13.6260,
                    'longitude': 123.1848,
                    'k': 5
                },
                'description': 'Get personalized property recommendations'
            }
        },
        'features': {
            'clustering': 'KMeans algorithm with budget categorization',
            'recommendations': 'KNN algorithm with confidence scores',
            'preprocessing': 'Outlier removal and robust scaling',
            'budget_categories': 'Low Budget (33%), Mid-Range (33%), High-End (33%)',
            'currency': 'Philippine Peso (₱)'
        },
        'metrics': {
            'silhouette_score': 'Clustering quality (range: -1 to 1, higher is better)',
            'davies_bouldin_score': 'Cluster separation (range: 0 to ∞, lower is better)',
            'calinski_harabasz_score': 'Cluster density (range: 0 to ∞, higher is better)',
            'confidence_score': 'Recommendation reliability (range: 0 to 1, higher is better)'
        }
    })

@app.route('/check-db', methods=['GET'])
def check_db():
    if collection is None:
        return jsonify({"status": "disconnected", "message": "MongoDB connection failed"}), 500
    try:
        count = collection.estimated_document_count()
        return jsonify({"status": "connected", "documents": count}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/ml', methods=['POST'])
def ml_api():
    global scaler_cache
    
    if collection is None:
        return jsonify({"error": "Database not connected"}), 500

    req = request.get_json()
    mode = req.get('mode')
    price = req.get('price')
    latitude = req.get('latitude')
    longitude = req.get('longitude')
    
    # Optional parameters
    k_neighbors = req.get('k', 5)  # Number of neighbors for KNN
    n_clusters = req.get('n_clusters', None)  # Number of clusters for KMeans (auto if None)

    raw_data = list(collection.find())
    if not raw_data:
        return jsonify({"error": "No data found"}), 404

    df = pd.DataFrame(raw_data)
    
    # Use improved preprocessing with scaling
    try:
        data_array, scaler, df_clean = preprocess_data(df, scaler=scaler_cache, fit=True)
        scaler_cache = scaler  # Cache the scaler for consistency
        
        logging.info(f"Preprocessed data shape: {data_array.shape}")
        logging.info(f"Original data: {len(df)} records, After cleaning: {len(df_clean)} records")
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 400

    if data_array.shape[0] == 0:
        return jsonify({"error": "No valid data after preprocessing"}), 400

    if mode == 'kmeans':
        try:
            # Use tertiles for small datasets (20 samples) for stable Low/Mid/High grouping
            prefer_k = n_clusters if n_clusters is not None else 3
            kres = train_kmeans(data_array, df_clean.iloc[:, 0].to_numpy(), enforce_tertiles=True, prefer_k=prefer_k)
            labels = kres.get('labels', [])

            # Attach budget categories to cleaned dataframe
            df_clean = df_clean.reset_index(drop=True)
            df_clean['budget_category'] = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

            # Summarize budget groups
            prices = df_clean.iloc[:, 0].to_numpy()
            summary = summarize_budget_groups(prices, labels)

            output = df_clean[['price', 'latitude', 'longitude', 'budget_category']].fillna("").to_dict(orient='records')

            return jsonify({
                'properties': output,
                'method': kres.get('method'),
                'metrics': kres.get('metrics', {}),
                'budget_summary': summary,
                'n_properties': int(len(prices))
            }), 200

        except Exception as e:
            logging.error(f"KMeans budget categorization error: {e}")
            return jsonify({"error": f"KMeans processing failed: {str(e)}"}), 500

    elif mode == 'knn':
        if None in (price, latitude, longitude):
            return jsonify({"error": "Missing required fields for knn: price, latitude, longitude"}), 400
        
        try:
            # Scale the input point using the same scaler
            point = np.array([[float(price), float(latitude), float(longitude)]])
            point_scaled = scaler.transform(point)[0]

            # Validate k_neighbors
            if k_neighbors > len(data_array):
                k_neighbors = len(data_array)
                logging.warning(f"Requested k={k_neighbors} exceeds data size, adjusted to {len(data_array)}")

            # Get recommendations with improved KNN (new signature: features, original_df, query_scaled)
            recs = knn_recommend(data_array, df_clean.reset_index(drop=True), point_scaled, k=k_neighbors)

            return jsonify({
                'recommendations': recs.get('recommendations', []),
                'query': {
                    'price': float(price),
                    'latitude': float(latitude),
                    'longitude': float(longitude)
                },
                'k': k_neighbors,
                'avg_confidence': recs.get('avg_confidence')
            }), 200

        except ValueError as e:
            logging.error(f"Input validation error: {e}")
            return jsonify({"error": "Invalid input values"}), 400
        except Exception as e:
            logging.error(f"KNN error: {e}")
            return jsonify({"error": f"KNN recommendation failed: {str(e)}"}), 500

    return jsonify({"error": "Invalid mode. Use 'kmeans' or 'knn'"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
