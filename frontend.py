import streamlit as st
from vector_database import upload_pdf, process_uploaded_pdf
from rag_pipeline import answer_query, llm_model

st.set_page_config(page_title="📄 RAG PDF Q&A", layout="centered")
st.title("📄 Ask Questions from your PDF")

# Upload Section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)

# User Question Input
user_query = st.text_area("Enter your query", placeholder="Type your question here...", height=150)

# Ask Button
if st.button("Ask Question"):
    if uploaded_file and user_query:
        with st.spinner("Processing PDF and generating answer..."):

            # Step 1: Save and embed the uploaded PDF
            saved_file_path = upload_pdf(uploaded_file)
            faiss_db = process_uploaded_pdf(saved_file_path)

            # Step 2: Run similarity search + LLM response
            retrieved_docs = faiss_db.similarity_search(user_query, k=3)
            response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

            # Step 3: Display conversation
            st.chat_message("user").write(user_query)
            st.chat_message("ai").write(response)
    else:
        st.error("Please upload a PDF and enter a question.")
