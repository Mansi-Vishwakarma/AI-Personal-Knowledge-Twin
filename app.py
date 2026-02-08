import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("AI Personal Knowledge Twin - MVP")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.texts = []

model = SentenceTransformer("all-MiniLM-L6-v2")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    new_texts = []

    for file in uploaded_files:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                new_texts.append(text)

    embeddings = model.encode(new_texts)
    dim = embeddings.shape[1]

    if st.session_state.index is None:
        st.session_state.index = faiss.IndexFlatL2(dim)

    st.session_state.index.add(np.array(embeddings))
    st.session_state.texts.extend(new_texts)

    st.success(f"{len(new_texts)} pages added to memory")
    st.write("Total memory chunks:", st.session_state.index.ntotal)

st.subheader("Ask a question from your documents")

question = st.text_input("Enter your question")

if question and st.session_state.index is not None:
    q_embedding = model.encode([question])
    D, I = st.session_state.index.search(np.array(q_embedding), k=1)

    answer = st.session_state.texts[I[0][0]]

    st.markdown("Answer")
    st.write(answer)


