# Date : September 30, 2024
# Author: Ardya Dipta Nandaviri (ardyadipta@gmail.com)
# aspiration & credits to :
# * Build RAG Application with Gemini using Langchain by Karndeep Singh (youtube)
# * Gen AI projects Gen AI by Krish Naik on Udemy

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as generativeai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
generativeai.configure(api_key=api_key)


def extract_pdf_content(pdf_files):
    """
    Extracts the text content from the uploaded PDF files.

    Args:
        pdf_files (list): List of uploaded PDF files.

    Returns:
        str: The concatenated text content extracted from all pages of the PDF files.
    """
    combined_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            combined_text += page.extract_text()
    return combined_text


def split_text_into_chunks(content):
    """
    Splits a large text into smaller chunks for processing.

    Args:
        content (str): The large block of text to be split into smaller chunks.

    Returns:
        list: A list of text chunks, each within the specified size and overlap limits.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
    return splitter.split_text(content)


def generate_vector_index(text_segments):
    """
    Creates a vector store of text chunks and saves it to a FAISS index.

    Args:
        text_segments (list): List of text chunks to be embedded and indexed.

    Returns:
        None: The FAISS index is saved locally as 'faiss_index_store'.
    """
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(text_segments, embedding=embed_model)
    vector_db.save_local("faiss_index_store")


def create_qa_chain():
    """
    Creates a question-answering chain using the Gemini model.

    The chain is built using a custom prompt template and a language model to answer questions based on context.

    Returns:
        Chain: A LangChain question-answering chain configured with a Gemini model.
    """
    custom_prompt = """
    Please provide a detailed answer based on the context given. If the context does not contain the answer, simply state, 
    "The answer is not available in the provided context." Please do not guess.\n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt_template = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
    return qa_chain


def handle_user_query(query):
    """
    Handles the user input by searching for relevant documents and generating an answer.

    Args:
        query (str): The user's question or query input.

    Returns:
        None: The answer is displayed in the Streamlit interface.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_db = FAISS.load_local("faiss_index_store", embedding_model, allow_dangerous_deserialization=True)
    matched_docs = faiss_db.similarity_search(query)
    qa_chain = create_qa_chain()
    result = qa_chain({"input_documents": matched_docs, "question": query}, return_only_outputs=True)

    st.write("AI Response:", result["output_text"])


def main():
    """
    Main function for running the Streamlit app.

    Sets up the interface, handles user input, and processes the uploaded PDF files.

    Returns:
        None
    """
    st.set_page_config(page_title="PDF Q&A with Gemini")
    st.header("Chat with Gemini AI 🤖 Who Consumed Knowledge from your PDF")

    user_query = st.text_input("Enter a question based on your uploaded PDFs:")

    if user_query:
        handle_user_query(user_query)

    with st.sidebar:
        st.title("Upload Section:")
        uploaded_pdfs = st.file_uploader("Select PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing your files..."):
                extracted_text = extract_pdf_content(uploaded_pdfs)
                text_segments = split_text_into_chunks(extracted_text)
                generate_vector_index(text_segments)
                st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()
