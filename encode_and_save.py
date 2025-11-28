# encode_and_save.py

import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("ArXivFinal_3_dataset.csv")
df.dropna(subset=["summary", "title","first_author"], inplace=True)
df.drop_duplicates(subset=["id"], inplace=True)

# Load scientific Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Best scientifc paper embeddings

# Create embeddings for summaries
embeddings = model.encode(
    df["summary"].tolist(),
    batch_size=32,
    show_progress_bar=True
)

# Save artifacts
model.save("sentence_model")  
np.save("embeddings.npy", embeddings)
pickle.dump(df, open("data.pkl", "wb"))


print("🎯 SciBERT Model + Embeddings saved successfully!")
