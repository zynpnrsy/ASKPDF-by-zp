# ASKPDF-by-zp
Intelligent PDF Question Answering
Many critical pieces of information, from academic articles to corporate reports, are stored in PDF format; however, accessing this information is often cumbersome and time-consuming.
Moreover, in many companies, uploading sensitive internal data to cloud-based models like ChatGPT is prohibited due to security and GDPR policies. Therefore, LLM systems that can analyze data without taking it outside the organization are becoming an important need.
AskPDF is an artificial intelligence solution I developed to respond precisely to this need and to speed up the document access process.


Hugging Face: https://huggingface.co/spaces/zeyneppinarsoy/ASKPDF
https://medium.com/@zeyneppinarsoy/askpdf-local-pdf-rag-assistant-52ae251f03ab

Notes: 
app.py > local ollama llm

app_g.py > hugging face gradio api

test_loaders are for chunking & embeddings uses langchain
