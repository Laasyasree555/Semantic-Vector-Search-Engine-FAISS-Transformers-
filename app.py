import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS Index
index = faiss.read_index("faiss_index.bin")

# Load Passages
with open("passages.pkl", "rb") as f:
    passages = pickle.load(f)

# Streamlit UI
st.title("🔍 Semantic Search Engine (FAISS + Transformers)")

query = st.text_input("Enter your search query:")

if st.button("Search"):
    query_emb = model.encode([query])
    distances, indices = index.search(query_emb, k=5)
    
    st.subheader("Top Results:")
    
    for score, idx in zip(distances[0], indices[0]):
        st.write(f"**Score:** {float(score):.4f}")
        st.write(f"**Passage:** {passages[idx]}")
        st.write("---")
