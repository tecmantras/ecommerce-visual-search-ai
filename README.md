# AI-Powered Image Search System

This project enables **image-based product searches** using **FAISS** and **ResNet-50**. It indexes product images from **AWS S3** and allows users to search for similar products via a **Flask API**.

### 🎥 Demo Video
<video width="600" 
   src="./assets/AI Image Search.mp4">
</video>



## 📌 Project Overview

This system is designed for **e-commerce platforms** that want to enable **image-based product search**. Users can upload an image, and the system finds **visually similar products** using **deep learning and FAISS-based indexing**.

---

## 📌 Prerequisites

Before installing, make sure you have:
- **Python 3.11.11**
- **MySQL for database**
- **AWS S3 Bucket** (to store images)
- **Conda (Miniconda/Anaconda)** for environment management

---

## 📌 Install & Set Up Conda Environment
To install Conda on your Ubuntu system, you can use the Miniforge installer, which comes pre-configured with Conda and the conda-forge channel.

1. Download the Miniforge Installer:
For a 64-bit x86 system (most common), use the following command:
    ```sh
    wget https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh
    ```


2. Verify the Installer's Integrity (Optional but Recommended):
    ```sh
    sha256sum Miniforge3-24.11.3-2-Linux-x86_64.sh
    ```

3. Run the Installer:
    ```
    chmod +x Miniforge3-24.11.3-2-Linux-x86_64.sh
    ./Miniforge3-24.11.3-2-Linux-x86_64.sh
    ```
4. Initialize Conda:
    ```
    ~/miniforge3/bin/conda init
    ```
5. Verify the Installation:
    ```
    conda --version
    ```

6. Create and activate a **Conda environment** for this project.
    ```sh
    conda create --name faiss_env python=3.11.11 -y
    conda activate faiss_env
    ```

---

## 📌  Install Required Dependencies
Run the following command to install FAISS and other required libraries:

```sh
conda install -c conda-forge faiss-cpu -y
pip install --upgrade pip
pip install flask flask-cors numpy pymysql boto3 torch torchvision pillow
```

## 📌 Verify FAISS Installation:

```sh
python -c "import faiss; print(f'✅ FAISS Installed: Version {faiss.__version__}')"
```

## 📌 MySQL Database Setup
Create a MySQL database and table for storing product information:
```sh
CREATE DATABASE image_search_db;

CREATE TABLE product (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    image VARCHAR(255) -- Image filename stored in S3
);
```

## 📌 Configure Project Settings
Create a config.py file in the project root and add the following details:

```sh
# MySQL Configuration
MYSQL_HOST = "your-mysql-host"
MYSQL_PORT = 3306
MYSQL_USER = "your-username"
MYSQL_PASSWORD = "your-password"
MYSQL_DB = "image_search_db"

# AWS S3 Configuration
S3_BUCKET = "your-s3-bucket"
S3_REGION = "your-region"
S3_ACCESS_KEY = "your-access-key"
S3_SECRET_KEY = "your-secret-key"
S3_FOLDER = "ecom-image-ai"
```
Replace your-mysql-host, your-username, and AWS credentials with actual values.


## 📌 Index Product Images

Run the following script to fetch images from AWS S3, extract features, and index them in FAISS:

```sh
python index_images.py
```
✔ This will create product_index.faiss and product_ids.npy.


## 📌 Start the Flask API

Run the following command to start the API server:

```sh
python image_search.py
```
The API will be available at:
http://127.0.0.1:5000


## 📌 Usage Instructions

Send a POST request with an image file:

```sh
curl -X POST -F "image=@query.jpg" http://127.0.0.1:5000/search
```
