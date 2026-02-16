#Bu kod lokalde OLLAMA ile çalışırken chunk processlerini görüntülemek için kullanıldı.
#Yalnızca terminalde gözükür
#sample_pdf ile kullanılmalı

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
print("First chunk:\n")
print(chunks[0].page_content)
