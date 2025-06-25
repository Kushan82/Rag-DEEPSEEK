import streamlit as st
from rag_pipeline import answer_query, retrieve_docs, llm_model
uploaded_file = st.file_uploader("Upload a file", type=["pdf"],accept_multiple_files=False)

user_query = st.text_area("Enter your query", placeholder="Type your question here...",height=150)

ask_question = st.button("Ask Question")

if ask_question:

    if uploaded_file:
        st.chat_message("user").write(user_query)

        retrieved_docs=retrieve_docs(user_query)
        response=answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        #fixed_response = "This is a fixed response for demonstration purposes."
        st.chat_message("ai help").write(response)
    else:
        st.error("Please upload a PDF file before asking a question.")