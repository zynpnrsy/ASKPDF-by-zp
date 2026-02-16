#lokalde OLLAMA ile çalışırken chunk + embedding processlerini incelemk için kullanıldı.
#Yalnızca terminalde gözükür.
#sample_pdf ile kullanılmalı


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print(f"Total pages: {len(documents)}")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

chunk_texts = [chunk.page_content for chunk in chunks]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = embedding_model.encode(chunk_texts)

print("Embedding shape:", embeddings.shape)