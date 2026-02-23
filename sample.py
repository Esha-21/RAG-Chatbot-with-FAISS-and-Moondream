# ui framework to build the web app.
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text
        else:
            st.warning(f"⚠️ Page {page_num} has no extractable text (maybe scanned image).")
    return text


# Function to create FAISS vector store
def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    st.write("📑 Number of chunks created:", len(chunks))
    st.write("🔹 First chunk preview:", chunks[0][:1000] if chunks else "No chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)
    st.success("✅ FAISS vector store created and saved locally.")


# Load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    st.success("✅ FAISS vector store loaded.")
    return vector_store


# Build QA Chain
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()

    llm = Ollama(model="moondream:latest")  # ensure model exists in `ollama list`
    st.info("🔗 QA Chain initialized with Moondream:latest model.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain



# Streamlit App
st.title("📘 RAG Chatbot with FAISS and Moondream:latest")
st.write("Upload a PDF and ask questions based on its content.")


uploaded_file = st.file_uploader("📂 Upload your PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = f"uploaded/{uploaded_file.name}"
    os.makedirs("uploaded", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    st.write("🔹 Extracted text length:", len(text))
    st.text_area("📖 PDF Extract Preview", text[:1000], height=200)

    if len(text.strip()) > 0:
        st.info("⚙️ Creating FAISS vector store...")
        create_faiss_vector_store(text)

        st.info("🤖 Initializing chatbot...")
        qa_chain = build_qa_chain()
        st.success("✅ Chatbot is ready!")

        question = st.text_input("💬 Ask a question about the uploaded PDF:")
        if question:
            st.info(f"🔍 Querying the document with: `{question}`")
            try:
                answer = qa_chain.run(question)
                st.success(f"📝 Answer: {answer}")
            except Exception as e:
                st.error(f"❌ Error while running QA chain: {e}")

    else:
        st.error("❌ No text extracted from the PDF. Try another file.")
