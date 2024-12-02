import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sayvai_rag.config import create_vector_store
from sayvai_rag.search import search_vector_store
from sayvai_rag.milvus_vector_store import create_user_store
from sayvai_rag.text_splitter import load_and_split_pdf

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
os.environ["MILVUS_URI"] = "test.db"

# Initialize FastAPI app
app = FastAPI(
    title="sayvai-rag-backend",
    description="This is a sayvai backend rag app",
    version="0.1",
)

# Pydantic models for input validation
class Config(BaseModel):
    query: str
    user_name: str
    doc_name: str | None = None
    
    
class CreateConfig(BaseModel):
    user_name: str
    doc_name: str

# Root route
@app.get("/")
async def root():
    return {"message": "API working. Welcome to the Sayvai rag app."}

# Chat route for querying vector store
@app.post("/chat/")
async def chat(config: Config):
    vector_store = create_user_store(
        embeddings,
        connection_args={"uri": os.environ["MILVUS_URI"]},  # Fix the argument here
        collection_name=config.user_name,
        document_name=config.doc_name
    )
    return search_vector_store(vector_store, config.query)

# Route for uploading and creating the vector store from a PDF
@app.post("/chat/create/")
async def insert(config: CreateConfig, file: UploadFile = File(...)):
    # Save the uploaded pdf 
    file_path = f"documents/{config.doc_name}"
    with open(file_path, "wb") as f:
        file_content = await file.read
        f.write(file_content)    
    # Split the PDF into chunks
    documents = await load_and_split_pdf(file_path)  # Ensure this returns a list of document chunks
    
    # Create vector store for the user
    vector_store = create_vector_store(
        embeddings,
        connection_args={"uri": os.environ["MILVUS_URI"]},
        collection_name=config.user_name,
        document_name=config.doc_name,
        documents=documents
    )
    
    # Optional: Test the vector store by performing a search after insertion
    search_result = search_vector_store(vector_store, "what is the twin city of Kovai")
    print(search_result)
    
    return {"message": "Inserted successfully and vector store created."}
