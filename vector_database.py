from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import faiss
import os
import uuid
pdf_directory = "pdfs/"

def upload_pdf(file):
    os.makedirs(pdf_directory, exist_ok=True)
    file.name = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(pdf_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings

def process_uploaded_pdf(file_path):
    documents = load_pdf(file_path)
    text_chunks = create_chunks(documents)
    embedding_model = get_embedding_model("nomic-embed-text")
    faiss_db = faiss.FAISS.from_documents(text_chunks, embedding_model)
    faiss_db.save_local("vectorstore/db_faiss")
    return faiss_db
