#  RAG Chatbot with FAISS and Moondream

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and interact with its content using natural language questions.

This system combines semantic embeddings, vector similarity search (FAISS), and a locally deployed Large Language Model (Moondream via Ollama) to generate accurate, document-grounded responses.

---

## Project Objective

Large Language Models (LLMs) cannot access private or user-uploaded documents by default.  
This project solves that limitation by implementing a Retrieval-Augmented Generation (RAG) pipeline that:

- Extracts content from uploaded PDFs
- Converts text into vector embeddings
- Performs similarity search using FAISS
- Injects relevant context into the LLM
- Generates accurate, grounded responses

---

## Features

- 📂 Upload any PDF document
- 📄 Extract text using PyPDF2
- ⚠ Handles warnings for scanned or non-extractable pages
- ✂ Splits document into semantic chunks
- 🧠 Generates embeddings using HuggingFace Sentence Transformers
- 📦 Stores embeddings in FAISS vector database
- 🔎 Performs similarity search for relevant chunks
- 🤖 Uses Ollama (Moondream LLM) for response generation
- 🔒 Fully local deployment (no external API required)

---

##  System Architecture

1. User uploads a PDF  
2. Text extraction using PyPDF2  
3. Document text split into chunks  
4. Embedding generation using Sentence Transformers  
5. FAISS vector index creation  
6. User query converted into embedding  
7. Similarity search retrieves relevant chunks  
8. Retrieved context sent to Moondream LLM  
9. Context-aware answer generated  

---

## Technology Stack

### Frontend
- Streamlit

### Backend / Processing
- Python

### AI Components
- LangChain
- Sentence Transformers (HuggingFace)
- FAISS (Facebook AI Similarity Search)
- Ollama (Moondream LLM)

### PDF Processing
- PyPDF2

---
## install dependencies
pip install streamlit PyPDF2 faiss-cpu langchain sentence-transformers ollama

## Pull Moondream Model
ollama pull moondream:latest

##  Run the Application
streamlit run app.py
