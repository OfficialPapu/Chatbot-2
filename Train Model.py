import os
import time
import mysql.connector
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch

start_time = time.time()

# === 1. Load Environment Variables ===
load_dotenv()

# === 2. Initialize Pinecone Vector Database ===
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

# === 3. Load Hugging Face Embedding Model (BAAI/bge-base-en-v1.5) ===
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

# === 4. Define Text Embedding Function ===
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return (embedding / embedding.norm(dim=-1, keepdim=True)).squeeze().tolist()

# === 5. Connect to MySQL Database and Prepare Cursor ===
conn = mysql.connector.connect(host="localhost", user="root", password="", database="dream skin nepal")
cursor = conn.cursor(dictionary=True)

# === 6. Define Generic Upsert Function for Pinecone ===
def process_and_upsert(query, get_text_fn, get_metadata_fn, embed_fn, index, batch_size=96):
    cursor.execute(query)
    rows = cursor.fetchall()

    records = []
    for row in rows:
        text = get_text_fn(row)
        vector = embed_fn(text)
        records.append({
            "id": str(row["ID"]),
            "values": vector,
            "metadata": get_metadata_fn(row),
        })

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert(batch)
        time.sleep(1)  # Sleep to respect rate limits

# === 7. Upsert Product Data into Pinecone ===
process_and_upsert(
    query="SELECT `ID`, `Product Title`, `Slug Url`, `Product Content`, `Product Price` FROM posts",
    get_text_fn=lambda row: f"{row['Product Title']} {row['Product Content']}",
    get_metadata_fn=lambda row: {
        "title": row["Product Title"],
        "price": row["Product Price"],
        "slug": row["Slug Url"],
    },
    embed_fn=embed,
    index=index
)

# === 8. Upsert FAQ Data into Pinecone ===
process_and_upsert(
    query="SELECT `ID`, `Question`, `Answer` FROM `ai`",
    get_text_fn=lambda row: f"{row['Question']} {row['Answer']}",
    get_metadata_fn=lambda row: {
        "Answer": row["Answer"]
    },
    embed_fn=embed,
    index=index
)

cursor.close()
conn.close()

end_time = time.time()
total_time = round(end_time - start_time, 2)
print(f"✅ All product and FAQ vectors uploaded successfully in {total_time} seconds.")
