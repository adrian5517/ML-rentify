from flask import Flask, request, jsonify
from pymongo import MongoClient, errors
from preprocess import preprocess_data
from ml_model import train_kmeans, get_kmeans_labels, knn_recommend
import pandas as pd
import os
from dotenv import load_dotenv
from flask_cors import CORS
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Setup logger
logging.basicConfig(level=logging.INFO)

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
    if collection is None:
        return jsonify({"error": "Database not connected"}), 500

    req = request.get_json()
    mode = req.get('mode')
    price = req.get('price')
    latitude = req.get('latitude')
    longitude = req.get('longitude')

    raw_data = list(collection.find())
    if not raw_data:
        return jsonify({"error": "No data found"}), 404

    df = pd.DataFrame(raw_data)
    df_clean = preprocess_data(df)
    if df_clean.empty:
        return jsonify({"error": "No valid data after preprocessing"}), 400

    data_array = df_clean.to_numpy()

    if mode == 'kmeans':
        model = train_kmeans(data_array)
        labels = get_kmeans_labels(model, data_array)
        df['cluster'] = labels.tolist()
        output = df[['name', 'price', 'location', 'cluster']].fillna("").to_dict(orient='records')
        return jsonify(output)

    elif mode == 'knn':
        if None in (price, latitude, longitude):
            return jsonify({"error": "Missing required fields for knn"}), 400
        try:
            point = [float(price), float(latitude), float(longitude)]
        except ValueError:
            return jsonify({"error": "Invalid input values"}), 400
        indices = knn_recommend(data_array, point)
        result = df.iloc[indices][['name', 'price', 'location']].fillna("").to_dict(orient='records')
        return jsonify(result)

    return jsonify({"error": "Invalid mode"}), 400

if __name__ == '__main__':
    app.run(debug=True)
