from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from sayvai_rag.config import create_vector_store
import os
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
os.environ["MILVUS_URI"] = "sayvai.db"

llm = ChatOpenAI(model="gpt-4o-mini")
# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

vector_store = create_vector_store(
        embeddings,
        connection_args={"uri": os.environ["MILVUS_URI"]},
        collection_name='subash',
        document_name= None
    )

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

while True:
    print("\n")
    input_message = input("User: ")

    for message, metadata in graph.stream(
        {"question": input_message},
        stream_mode="messages",
        config=config
    ):
        if metadata["langgraph_node"] == "generate":
            print(message.content,end=' ')
