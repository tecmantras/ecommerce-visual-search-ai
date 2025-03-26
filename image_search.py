import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pymysql
import boto3
import io
from config import *
from flask import Flask, request, jsonify
from flask_cors import CORS

# 🔹 Load FAISS Index & Product IDs
index = faiss.read_index("product_index.faiss")
product_ids = np.load("product_ids.npy")

# ✅ Normalize Feature Vectors Before FAISS Search
def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 🔹 Connect to MySQL
db = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB)
cursor = db.cursor()

# 🔹 Load AI Model (ResNet-50)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
model.eval()

# 🔹 Extract Features from an Image
def extract_features(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(img).numpy().flatten()

    return features / np.linalg.norm(features)  # ✅ Normalize for FAISS

# 🔹 Search for Similar Products
def search_similar_products(image_path, k=20, threshold=0.80):
    image = Image.open(image_path)
    query_vector = extract_features(image).reshape(1, -1)
    query_vector = normalize_vectors(query_vector)  # Normalize before FAISS search

    print(f"\n🔍 Query vector shape: {query_vector.shape}")
    print(f"📦 FAISS index dimension: {index.d}")

    if query_vector.shape[1] != index.d:
        raise ValueError(f"❌ Dimension mismatch: Query vector ({query_vector.shape[1]}) != FAISS index ({index.d})")

    # Perform FAISS search
    distances, indices = index.search(query_vector, k)

    # ✅ Filter results based on threshold
    valid_results = [(product_ids[idx], distances[0][i]) 
                     for i, idx in enumerate(indices[0]) 
                     if idx != -1 and distances[0][i] > threshold]  # Ensure high confidence

    if not valid_results:
        return {"message": "No similar products found"}  # ✅ Return this if no valid results

    # ✅ Sort by best similarity score
    valid_results.sort(key=lambda x: -x[1])

    # ✅ Fetch Product Details
    result_ids = [item[0] for item in valid_results]
    format_strings = ','.join(['%s'] * len(result_ids))
    query = f"SELECT id, name, image FROM product WHERE id IN ({format_strings})"
    cursor.execute(query, tuple(result_ids))
    products = cursor.fetchall()

    return [{"id": p[0], "name": p[1], "image_url": f"https://{S3_BUCKET}.s3.amazonaws.com/ecom-image-ai/{p[2]}"} for p in products]

# 🔹 Flask API
app = Flask(__name__)
CORS(app)

@app.route('/search', methods=['POST'])
def search():
    file = request.files['image']
    file.save("query.jpg")
    results = search_similar_products("query.jpg")

    if not results:
        return jsonify({"message": "No similar products found"}), 200

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
