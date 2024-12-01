# fastapi app 
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sayvai_rag.config import create_vector_store
from sayvai_rag.search import search_vector_store

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
os.environ["MILVUS_URI"] = "../sayvai.db"

app = FastAPI(
    title="sayvai-rag-backend",
    description="This is a sayvai backend rag app",
    version="0.1",
)

class Config(BaseModel):
    query: str
    user_name: str
    doc_name: str | None = None


@app.get("/")
async def root():
    return {"message": "API working Welcome to Sayvai rag app"}

@app.post("/chat/")
async def chat(config: Config):
    vector_store = create_vector_store(embeddings,
                                       connection_args= {"ui": os.environ["MILVUS_URI"]},
                                       collection_name= config.user_name, 
                                       document_name = config.doc_name
                                       )
    return search_vector_store(vector_store, config.query)
    


