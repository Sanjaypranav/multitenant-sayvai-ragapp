import os
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from sayvai_rag.config import create_vector_store

# Initialize OpenAI embeddings and LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o-mini")

# Ensure the Milvus URI is set
os.environ["MILVUS_URI"] = "sayvai.db"  # Replace with your actual URI

# Define prompt for question-answering
prompt = PromptTemplate(
    template="Answer the user's question as best as possible. If the user asks about the document, retrieve it using the retrieval tool.\n{context}\n{question}",
    input_variables=["context", "question"]
)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Initialize vector store
def initialize_vector_store(collection_name: str):
    return create_vector_store(
        embeddings=embeddings,
        connection_args={"uri": os.environ["MILVUS_URI"]},
        collection_name=collection_name,
        document_name=None
    )

# Define the retrieve function
def retrieve(state: State, vector_store):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Define the generate function
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build the StateGraph
def build_graph(collection_name):
    # Initialize the vector store
    vector_store = initialize_vector_store(collection_name)
    
    # Define a wrapper for the retrieve function
    def retrieve_with_store(state: State):
        return retrieve(state, vector_store)
    
    # Initialize memory saver
    memory = MemorySaver()  # Manage states
    
    # Create a StateGraph and add the retrieve and generate nodes
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve_with_store)  # Explicitly add the retrieve node
    graph_builder.add_node("generate", generate)  # Explicitly add the generate node
    
    # Define the sequence of execution and state transitions
    graph_builder.add_edge(START, "retrieve")  # Start the process with "retrieve"
    graph_builder.add_edge("retrieve", "generate")  # After "retrieve", move to "generate"
    
    # Compile the graph
    graph = graph_builder.compile(checkpointer=memory, interrupt_after=["generate"])
    return graph

# Main chatter function
def chatter(graph, input_message):
    for message, metadata in graph.stream(
        {"question": input_message},
        stream_mode="messages",
        config={"configurable": {"thread_id": "abc123"}}
    ):
        if metadata["langgraph_node"] == "generate":
            yield message.content

# Example Usage
# if __name__ == "__main__":
#     collection_name = "my_document_collection"  # Replace with your collection name
#     graph = build_graph(collection_name)
    
#     input_message = "What does the document say about AI?"
#     for response in chatter(graph, input_message):
#         print(response)
