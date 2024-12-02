# pdf_text_splitter.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio

# Function to load and split the PDF
def load_and_split_pdf(file_path, chunk_size=1024, chunk_overlap=10):
    # Initialize the PDF loader with the given file path
    loader = PyPDFLoader(file_path)
    
    # Load the PDF asynchronously
    pages = []
    for page in loader.alazy_load():
        pages.append(page)

    # Initialize the text splitter with the given parameters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split the loaded pages into smaller chunks
    all_splits = text_splitter.split_documents(pages)
    
    return all_splits
