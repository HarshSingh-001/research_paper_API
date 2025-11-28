def extract_arxiv_id(raw_id: str):
    # Remove known prefixes
    prefixes = ["abs-", "cmp-lg-", "hep-ph/", "cs/"]
    for p in prefixes:
        raw_id = raw_id.replace(p, "")
    
    return raw_id.strip()


def build_pdf_url(arxiv_id: str):
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def build_abs_url(arxiv_id: str):
    return f"https://arxiv.org/abs/{arxiv_id}"




# backend/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load model and data
model = SentenceTransformer("sentence_model")
embeddings = np.load("embeddings.npy")
df = pickle.load(open("data.pkl", "rb"))

app = FastAPI(title="SciBERT Research Paper Recommendation API")

class QueryInput(BaseModel):
    query: str
    top_k: int = 5

@app.post("/recommend")
async def recommend(input: QueryInput):
    query_embedding = model.encode([input.query])

    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Sort and pick top K
    top_indices = similarities.argsort()[-input.top_k:][::-1]

    results = []
    for idx in top_indices:
        raw_id = df.iloc[idx]["id"]
        arxiv_id = extract_arxiv_id(raw_id)

        results.append({
            "id": arxiv_id,
            "title": df.iloc[idx]["title"],
            "category": df.iloc[idx]["category"],
            "pdf_url": build_pdf_url(arxiv_id),
            "abs_url": build_abs_url(arxiv_id)
        })


    return {"query": input.query, "results": results}
