import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import torch

start_time = time.time()

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# === 1. Load Embedding Model (Hugging Face - BAAI/bge-base-en-v1.5) ===
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

# === 2. Define Embedding Function ===
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return (embedding / embedding.norm(dim=-1, keepdim=True)).squeeze().tolist()

# === 3. Input User Query ===
query_text = "send Order to Loved Ones"

print("🧠 Generating response...", end="", flush=True)

# === 4. Generate Embedding and Query Pinecone ===
vector = embed(query_text)
results = index.query(vector=vector, top_k=5, include_metadata=True)

# === 5. Format Retrieved Context from Pinecone Matches ===
def format_context(matches):
    product_list = []
    faq_list = []
    for match in matches:
        meta = match["metadata"]
        if "title" in meta and "slug" in meta:
            product_list.append(f"- {meta['title']} (Rs. {meta['price']}): http://dreamskinnepal.com/{meta['slug']}")
        elif "Answer" in meta:
            faq_list.append(f"Q: {match['id'].replace('-', ' ')}\nA: {meta['Answer']}")
    return "\n".join(product_list), "\n".join(faq_list)

product_context, faq_context = format_context(results["matches"])

# === 6. Construct Prompt for Gemini ===
prompt_parts = []

if product_context:
    prompt_parts.append(f"🛍️ **Related Products:**\n{product_context}")

if faq_context:
    prompt_parts.append(f"❓ **Similar Questions Answered:**\n{faq_context}")

prompt = f"""You are a helpful and knowledgeable Dream Skin expert.

A user asked: "{query_text}"

{'\n\n'.join(prompt_parts)}

Please respond clearly in 2-3 lines. Add product names or recommendations if relevant."""

print("\r✅ Response generated!\n")

# === 7. Generate Final Answer from Gemini ===
response = gemini_model.generate_content(prompt)

print("✅ Gemini Response:\n")
print(response.text.strip())

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTime taken: {elapsed_time:.2f} seconds")
