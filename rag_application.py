import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_community.document_loaders.csv_loader import CSVLoader

from utils import *
load_dotenv()

# Title of web app
st.title("RAG Application")

# Data send to Qdrant vector store :
qdrant_key = os.getenv("qdrant_key")
URL = os.getenv("URL")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
def send_to_qdrat(documents,embeddings):
    qdrant = Qdrant.from_documents(
            documents,
            embeddings,
            url=URL,
            prefer_grpc=False,
            api_key=qdrant_key,
            collection_name="streamlit_files"
        )
    return qdrant

# Qdrant vector store client side :
qdrant_key = os.getenv("qdrant_key")
URL = os.getenv("URL")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(
    url=URL,
    api_key=qdrant_key,
)
vector_db = Qdrant(
    client=client, collection_name="streamlit_files",
    embeddings=embedding_model,
)

# Main function for getting files and input :
def main_fun():
    # Create 'uploads' directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    # File upload
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "csv", "docx","txt"])
    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the file based on its type
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension == "pdf":
            loader = PyMuPDFLoader(file_path)  
            pages = loader.load()
        elif file_extension == "csv":
            loader = CSVLoader(file_path=file_path)
            pages = loader.load()
        elif file_extension == "docx":
            file_loader = Docx2txtLoader(file_path)
            pages = file_loader.load()
        else:
            st.error("Unsupported file format.")
            return

        def split_doc(pages):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=70,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_documents(pages)
            return texts

        text_new = split_doc(pages)
        # documents = [text_new(doc.page_content, metadata={"source": "abd"}) for doc in text_new]

        st.write(f"Number of chunks extracted: {len(text_new)}")
        # st.write(f"chunks{documents}")
        send_to_qdrat(text_new,embeddings)
        st.write("chunks uploaded to Qdrant vector store")
        bot_id = st.text_input("Enter your Bor id in (integer)")
        input_query = st.text_input("Enter query here ....")
        if st.button("submit"):
            # Geting conv_retrieval_chain from utils.py file 
            result = conv_retrieval_chain(vector_db,input_query,bot_id)
            input_query = result["question"]
            ai_response = result["answer"]
            if result :
                st.write("Question : ", input_query)
                st.write("Response : ",ai_response)
            else : 
                print("No response from vector db please try again")
    else:
        st.write("Please upload a file.")

if __name__ == '__main__':
    main_fun()
