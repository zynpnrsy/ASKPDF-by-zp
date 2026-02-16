#Lokalde OLLAMA + Gradio ile çalışan versiyon.
#sample_pdf ile kullanılmalı


import numpy as np
import faiss
import subprocess
import gradio as gr

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Load PDF
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


# Chunking
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# Embedding + FAISS
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings


# Retrieval
def retrieve(query, chunks, index, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]


# Ollama LLM
"""#
def ask_llm(context, question):
    prompt = f"""
"""#
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know, this pdf does not include this info."

Context:
{context}

Question:
{question}

Answer:
"""
"""#
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True
    )

    return result.stdout
#"""

def ask_llm(context, question):
    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    return result.stdout




# Full RAG Pipeline
print("Loading PDF...")
text = load_pdf("sample.pdf")
chunks = chunk_text(text)
print(f"Total chunks: {len(chunks)}")

print("Creating FAISS index...")
index, embeddings = create_vector_store(chunks)
print("System ready.")


def ask_question(question):
    retrieved_chunks = retrieve(question, chunks, index)

    print("Retrieved chunks:")
    for chunk in retrieved_chunks:
        print("-----")
        print(chunk[:200])

    context = "\n\n".join(retrieved_chunks)

    answer = ask_llm(context, question)
    return answer


# Gradio UI
demo = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=3, placeholder="Ask something about the PDF..."),
    outputs="text",
    title="Local PDF RAG Assistant by zeyneppinarsoy"
)

demo.launch()
