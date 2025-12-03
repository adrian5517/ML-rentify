from flask import Flask, request, jsonify
from pymongo import MongoClient, errors
from preprocess import preprocess_data
from ml_model import train_kmeans, knn_recommend, summarize_budget_groups
import pandas as pd
import os
from dotenv import load_dotenv
from flask_cors import CORS
import traceback
import logging
import numpy as np
from typing import Optional

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

# Do not establish MongoDB connection at import time (can block startup if network unavailable).
# We'll open a short-lived/lazy connection inside endpoints when needed.
collection = None

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


@app.route('/', methods=['POST'])
def root_post_help():
    """Helpful JSON response for clients that mistakenly POST to '/'.
    This prevents HTML 405/404 bodies from confusing mobile clients.
    """
    return jsonify({
        "error": "POST not supported at this endpoint.",
        "usage": "POST to /ml with JSON body: {\"mode\": \"kmeans\"} or {\"mode\": \"knn\", \"price\":..., \"latitude\":..., \"longitude\":...}",
        "note": "See / for full API documentation (GET)"
    }), 405


def _json_error(message: str, code: int = 400, details: Optional[dict] = None):
    payload = {"error": message}
    if details:
        payload["details"] = details
    return jsonify(payload), code


@app.errorhandler(404)
def handle_404(e):
    return _json_error("Not found. Check the requested path.", 404, {"path": request.path})


@app.errorhandler(405)
def handle_405(e):
    return _json_error("Method not allowed for this endpoint.", 405, {"method": request.method, "path": request.path})


@app.route('/health', methods=['GET'])
def health():
    """Health endpoint returning service and optional DB status."""
    resp = {"service": "Rentify ML API", "status": "ok"}
    try:
        col = get_mongo_collection()
        if col is None:
            resp["db"] = {"status": "disconnected"}
        else:
            # quick cheap check
            try:
                resp_count = col.estimated_document_count()
                resp["db"] = {"status": "connected", "documents": int(resp_count)}
            except Exception as ex:
                resp["db"] = {"status": "error", "message": str(ex)}
    except Exception:
        resp["db"] = {"status": "unknown"}
    return jsonify(resp), 200

@app.route('/check-db', methods=['GET'])
def check_db():
    # Try to open a short-lived connection to report status
    col = get_mongo_collection()
    if col is None:
        return jsonify({"status": "disconnected", "message": "MongoDB connection failed"}), 500
    try:
        count = col.estimated_document_count()
        return jsonify({"status": "connected", "documents": count}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/ready', methods=['GET'])
def readiness():
    """Readiness probe: returns 200 when DB is reachable, 503 otherwise.

    This is intended for orchestration systems (load balancers / k8s) to
    determine whether the app is ready to receive traffic.
    """
    col = get_mongo_collection()
    if col is None:
        return _json_error("Service not ready: database not available", 503, {"db": "disconnected"})
    try:
        cnt = col.estimated_document_count()
        return jsonify({"ready": True, "db": {"status": "connected", "documents": int(cnt)}}), 200
    except Exception as e:
        return _json_error("Service not ready: database error", 503, {"db": "error", "message": str(e)})

@app.route('/ml', methods=['POST'])
def ml_api():
    global scaler_cache
    # Development flag: when true the API will include tracebacks in error JSON (only enable for local debugging)
    DEBUG_API = os.getenv('DEBUG_API', 'false').lower() in ('1', 'true', 'yes')

    # Ensure we have a usable collection reference for this request (lazy connect)
    col = get_mongo_collection()
    if col is None:
        return jsonify({"error": "Database not connected"}), 500

    req = request.get_json()
    if req is None:
        return _json_error("Invalid or missing JSON body", 400, {"hint": "Ensure Content-Type: application/json and a valid JSON payload is sent"})
    mode = req.get('mode')
    if not mode:
        return _json_error("Missing required field 'mode' in JSON body", 400, {"hint": "mode must be 'kmeans' or 'knn'"})

    # Wrap the rest of /ml handling in a top-level try/except so we can capture unexpected errors
    try:
        price = req.get('price')
        latitude = req.get('latitude')
        longitude = req.get('longitude')

        # Optional parameters
        k_neighbors = req.get('k', 5)  # Number of neighbors for KNN
        n_clusters = req.get('n_clusters', None)  # Number of clusters for KMeans (auto if None)

        raw_data = list(col.find())
        if not raw_data:
            # Return an empty, structured result (200) so mobile clients receive JSON instead of 404 HTML
            logging.info("/ml request: properties collection empty — returning empty result")
            return jsonify({'properties': [], 'n_properties': 0}), 200

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
    except Exception as e:
        # Log full traceback server-side
        logging.exception("Unhandled exception in /ml: %s", e)
        tb = traceback.format_exc()
        if DEBUG_API:
            # Include the traceback in the JSON response for local debugging
            return _json_error("Internal server error", 500, {"exception": str(e), "trace": tb})
        else:
            return _json_error("Internal server error", 500)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
