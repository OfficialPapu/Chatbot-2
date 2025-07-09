import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai 
import torch

start_time = time.time()
# Load API key
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- PINECONE & TRANSFORMERS SETUP ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Load model for embeddings
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")


def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return (embedding / embedding.norm(dim=-1, keepdim=True)).squeeze().tolist()


# Embed and query
query_text = "my skin type is oily and what are the price of it"

print("🧠 Generating response...", end="", flush=True)

vector = embed(query_text)
results = index.query(vector=vector, top_k=5, include_metadata=True)

def format_context(matches):
    context = ""
    for match in matches:
        meta = match["metadata"]
        context += f"- {meta['title']} (Rs. {meta['price']}): http://dreamskinnepal.com/{meta['slug']}\n"
    return context

context_text = format_context(results["matches"])
prompt = f"""You are a skincare expert. Based on the following product list, recommend a product for someone asking about: "{query_text}"

Product List:
{context_text}

Respond in a helpful and friendly tone in 2-3 lines. Add product names if appropriate."""

print("\r✅ Response generated!\n")
response = gemini_model.generate_content(prompt)

# --- OUTPUT ---
end_time = time.time()
elapsed_time = end_time - start_time

print("✅ Gemini Response:\n")
print(response.text.strip())
print(f"\nTime taken: {elapsed_time:.2f} seconds")