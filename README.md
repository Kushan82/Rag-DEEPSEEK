# 📄 RAG-DEEPSEEK: Chat with Your PDF using RAG + Ollama + FAISS

This project enables users to **ask questions from any PDF document** using a Retrieval-Augmented Generation (RAG) pipeline. It uses:

- 💬 `LangChain` for chaining together embeddings, retrieval, and prompting.
- 🧠 `Ollama` for local embedding and model inference.
- 🔍 `FAISS` for fast vector similarity search.
- 🌐 `Streamlit` for a simple and interactive web UI.

---

## 📦 Features

- ✅ Upload your own PDF (on the fly)
- ✅ Automatically chunk and embed document using Ollama
- ✅ Retrieve relevant chunks using FAISS
- ✅ Use `deepseek-r1-distill-llama-70b` (via Groq) to answer queries based on retrieved context
- ✅ Local file handling with temporary vector store generation

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Rag-DEEPSEEK.git
cd Rag-DEEPSEEK

2. Create Virtual Environment & Install Requirements
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

3. Environment Variables
Create a .env file in the root directory with the following:

🚀 Running the App
streamlit run frontend.py

📂 Project Structure
Rag-DEEPSEEK/
├── frontend.py              # Streamlit UI
├── rag_pipeline.py          # Core RAG logic and LLM integration
├── vector_database.py       # File upload, chunking, embedding, FAISS logic
├── requirements.txt
├── .env
└── pdfs/                    # Directory to store uploaded PDFs

🧠 How It Works
    1. User uploads a PDF
    2. vector_database.py:
        Saves the PDF
        Loads and splits it into chunks
        Uses nomic-embed-text from Ollama to embed it
        Saves in a temporary FAISS vector store
    3. User enters a question
    4. App searches for relevant chunks and passes them to deepseek-r1-distill-llama-70b LLM (via Groq)
    5. Answer is generated and shown in the chat window


🤖 Requirements
Python 3.8+
Ollama running locally with:
    ollama pull nomic-embed-text
Groq API key (optional if you only want embedding)


