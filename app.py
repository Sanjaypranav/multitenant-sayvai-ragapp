import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sayvai_rag.config import create_vector_store
from sayvai_rag.search import search_vector_store
from sayvai_rag.milvus_vector_store import create_user_store
from sayvai_rag.text_splitter import load_and_split_files
from typing import TypedDict
from sayvai_rag.utils import format_docs
from fastapi.responses import StreamingResponse

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
os.environ["MILVUS_URI"] = "sayvai.db"

# Initialize FastAPI app
app = FastAPI(
    title="sayvai-rag-backend",
    description="This is a sayvai backend rag app",
    version="0.1",
)

# Pydantic models for input validation
class Config(BaseModel):
    query: str ='What is this document about'
    user_name: str 
    doc_name: str | None = None

class CreateConfig(BaseModel):
    user_name: str
    doc_name: str

class State(TypedDict):
    question: str
    collection_name: str
    docs_name: str

# global graph_build, graph_status
# graph_build = None
# graph_status = False


# def build_graph_rag(collection_name):
#     os.environ["USER_NAME"] = collection_name
#     from sayvai_rag.agent import SayvaiRagAgent
#     agent = SayvaiRagAgent(model="gpt-4o-mini", config={})
#     graph_build = agent.build_graph()
#     graph_status = True
#     return "Graph is built"


# Root route
@app.get("/")
def root():
    return {"message": "API working. Welcome to the Sayvai rag app."}

# Chat route for querying vector store
# @app.post("/chat")
# def chat(config: Config):
#     vector_store = create_vector_store(
#         embeddings,
#         connection_args={"uri": os.environ["MILVUS_URI"]},
#         collection_name=config.user_name,
#         document_name=config.doc_name
#     )
#     index = {"user_name": config.user_name, "doc_name": config.doc_name}
#     return format_docs(search_vector_store(vector_store, config.query, index=index))

@app.post("/chatbot")
def chatbot(config: Config):
    # if not graph_status:
    #     build_graph_rag(config.user_name)
    #     if graph_status:
    #         pass
    #     else:
    #         raise HTTPException(status_code=500, detail="Failed to build graph.")
    os.environ["USER_NAME"] = config.user_name
    from sayvai_rag.agent import SayvaiRagAgent
    agent = SayvaiRagAgent(model="groq-llama3-groq-8b-8192-tool-use-preview", config={"thread_id" : "1"})
    graph_build = agent.build_graph()
    return StreamingResponse(agent.chatter(graph=graph_build, input_message=config.query))


# Route for uploading and creating the vector store from a PDF
@app.post("/create")
def insert(
    user_name: str = Form(...),
    doc_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Ensure documents directory exists
        os.makedirs("documents", exist_ok=True)
        file_extension = os.path.splitext(file.filename)[1]

        # Save the uploaded PDF synchronously
        file_path = f"documents/{doc_name}{file_extension}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())  # Read and write file contents synchronously

        # Ensure the file pointer is closed after reading
        file.file.close()

        # Split the PDF into chunks
        documents = load_and_split_files(file_path)  # Ensure this function returns an iterable (e.g., list of chunks)

        # Create vector store for the user
        vector_store = create_user_store(
            embeddings,
            connection_args={"uri": os.environ["MILVUS_URI"]},
            collection_name=user_name,
            document_name=doc_name,
            documents=documents  # Convert to list if necessary
        )
        os.remove(file_path)

        return {"message": "Inserted successfully and vector store created."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
