import pymysql
import boto3
import faiss
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from config import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 🔹 Connect to AWS S3
s3_client = boto3.client(
    's3',
    region_name=S3_REGION,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

# 🔹 Connect to MySQL
db = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB)
cursor = db.cursor()

# 🔹 Load AI Model (ResNet-50)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# 🔹 Load Image from S3
def load_s3_image(image_name):
    try:
        s3_image_path = f"ecom-image-ai/{image_name}" 
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_image_path)
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data))

        if image.mode == "RGBA":
            image = image.convert("RGB")

        return image
    except Exception as e:
        print(f"❌ Error loading image {image_name}: {e}")
        return None

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

# 🔹 Initialize FAISS Index
d = 2048
index = faiss.IndexFlatIP(d)  # ✅ Use Inner Product for better matching
product_ids = []

# 🔹 Fetch Product Images from MySQL
cursor.execute("SELECT id, image FROM product")
products = cursor.fetchall()

for product in products:
    product_id, image_name = product
    image = load_s3_image(image_name)

    if image:
        vector = extract_features(image)
        vector = vector / np.linalg.norm(vector)  # ✅ Normalize before adding to FAISS
        index.add(np.array([vector]))  
        product_ids.append(product_id)

# 🔹 Save FAISS Index & Product ID Mapping
faiss.write_index(index, "product_index.faiss")
np.save("product_ids.npy", np.array(product_ids))

print(f"✅ FAISS Index Created Successfully with ResNet-50!")
