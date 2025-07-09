import os
import time
import mysql.connector
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch

# Load API key
load_dotenv()


# === 1. Initialize Pinecone ===
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# === 2. Load Hugging Face embedding model ===
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")


def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return (embedding / embedding.norm(dim=-1, keepdim=True)).squeeze().tolist()


# === 3. Connect to MySQL and fetch data ===
conn = mysql.connector.connect(
    host="localhost", user="root", password="", database="dream skin nepal"
)
cursor = conn.cursor(dictionary=True)
cursor.execute(
    "SELECT `ID`, `Product Title`, `Slug Url`, `Product Content`, `Ingredients`, `How to use`, `Product Price` FROM posts"
)
rows = cursor.fetchall()

# === 4. Prepare and upsert records in batches ===
records = []
for row in rows:
    text = f"{row['Product Title']} {row['Product Content']} {row['Ingredients']} {row['How to use']}"
    vector = embed(text)
    records.append(
        {
            "id": str(row["ID"]),
            "values": vector,
            "metadata": {
                "title": row["Product Title"],
                "slug": row["Slug Url"],
                "price": row["Product Price"],
            },
        }
    )

# # Upsert in batches of 96
for i in range(0, len(records), 96):
    batch = records[i : i + 96]
    index.upsert(batch)
    time.sleep(1)  # rate limiting

print("✅ All product vectors uploaded successfully.")

cursor.close()
conn.close()
