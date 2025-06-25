import streamlit as st
import os
from rag_pipeline import answer_query, retrieve_docs, llm_model
from vector_database import upload_pdf, process_uploaded_pdf

st.set_page_config(page_title="RAG PDF Q&A", layout="centered")

st.title("📄 Ask Questions from your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)
user_query = st.text_area("Enter your query", placeholder="Type your question here...", height=150)
ask_question = st.button("Ask Question")

if ask_question:
    if uploaded_file:
        st.chat_message("user").write(user_query)

        # Save and process uploaded PDF
        saved_file_path = upload_pdf(uploaded_file)
        faiss_db = process_uploaded_pdf(saved_file_path)

        # Retrieve and answer
        retrieved_docs = faiss_db.similarity_search(user_query, k=3)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        st.chat_message("ai").write(response)
    else:
        st.error("Please upload a PDF file before asking a question.")
