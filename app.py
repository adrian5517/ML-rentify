from flask import Flask, request, jsonify
from pymongo import MongoClient, errors
from preprocess import preprocess_data
from ml_model import train_kmeans, get_kmeans_labels, knn_recommend
import pandas as pd
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app)

# MongoDB connection check
try:
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
    client.server_info()  # Force connection on a request
    db = client['test']
    collection = db['properties']
    print("✅ Successfully connected to MongoDB!")
except errors.ServerSelectionTimeoutError as err:
    print("❌ MongoDB connection failed:", err)
    client = None
    collection = None

# Diagnostic route
@app.route('/check-db', methods=['GET'])
def check_db():
    if client is None or collection is None:
        return jsonify({"status": "disconnected", "message": "MongoDB connection failed"}), 500
    try:
        test_doc = collection.find_one()
        if test_doc:
            return jsonify({"status": "connected", "sample": test_doc}), 200
        else:
            return jsonify({"status": "connected", "message": "No documents found"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/ml', methods=['POST'])
def ml_api():
    if collection is None:
        return jsonify({"error": "Database not connected"}), 500

    req = request.get_json()
    mode = req.get('mode')  # 'kmeans' or 'knn'
    price = req.get('price')
    latitude = req.get('latitude')
    longitude = req.get('longitude')

    raw_data = list(collection.find())
    if not raw_data:
        return jsonify({"error": "No data found"}), 404

    df = pd.DataFrame(raw_data)
    df_clean = preprocess_data(df)
    data_array = df_clean.to_numpy()

    if mode == 'kmeans':
        model = train_kmeans(data_array)
        labels = get_kmeans_labels(model, data_array)
        df['cluster'] = labels.tolist()
        return jsonify(df[['name', 'price', 'location', 'cluster']].to_dict(orient='records'))

    elif mode == 'knn':
        if price is None or latitude is None or longitude is None:
            return jsonify({"error": "Missing fields"}), 400
        point = [price, latitude, longitude]
        indices = knn_recommend(data_array, point)
        result = df.iloc[indices][['name', 'price', 'location']].to_dict(orient='records')
        return jsonify(result)

    return jsonify({"error": "Invalid mode"}), 400


if __name__ == '__main__':
    app.run(debug=True)
