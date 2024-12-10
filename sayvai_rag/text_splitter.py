from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader, JSONLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# Function to load and split the PDF
def load_and_split_files(file_path, chunk_size=500, chunk_overlap=20):

    file_extension = os.path.splitext(file_path)[1]

    if file_extension == ".txt":
        loader = TextLoader(file_path)

    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif file_extension == ".json":
        loader = JSONLoader(file_path)

    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif file_extension == ".html":
        loader = UnstructuredHTMLLoader(file_path)

    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path)

    else :
        loader = CSVLoader(file_path)
    
    
    # Load the File synchronously using .load() instead of .alazy_load()
    pages = loader.load()

    # Initialize the text splitter with the given parameters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split the loaded pages into smaller chunks
    all_splits = text_splitter.split_documents(pages)
    
    return all_splits