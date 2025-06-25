from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import faiss


pdf_directory = "pdfs/"
def upload_pdf(file):
   with open(pdf_directory + file.name, "wb") as f:
       f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path = 'Updated_Dataset_Description_Guide.pdf'
documents = load_pdf(file_path)

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,add_start_index=True)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
#print("chunks count", len(text_chunks))

ollama_model_name = "nomic-embed-text"
def get_embedding_model(ollama_model_name):
    embeddings= OllamaEmbeddings(model=ollama_model_name)
    return embeddings

FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = faiss.FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)