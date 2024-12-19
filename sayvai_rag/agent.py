from typing import Dict

from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from sayvai_rag.config import create_vector_store
import os
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.llms import BaseLLM
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from sayvai_rag.utils import format_docs


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# os.environ["MILVUS_URI"] = "sayvai.db"

# llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
# Define prompt for question-answering
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
# Con


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class SayvaiRagAgent:
    def __init__(self, model: str):
        self.llm = self.get_llm(model)
        self.vector_store = None

    def get_llm(self, model) -> BaseLLM:
        """
        Get the language model for the agent.
        Args:
            model: if groq use groq-modelname, openai use modelname, ollama if ollama-modelname

        Returns:
            BaseLLM: Language model for the agent.
        """
        if model[:3] == "gpt":
            return ChatOpenAI(model=model, streaming=True)
        if model[:4] == "groq":
            return ChatGroq(model=model[5:], streaming=True)
        if model[:6] == "ollama":
            return ChatOllama(model=model[7:], streaming=True)
        
    def init_vector_store(self, collection_name: str):
        self.vector_store = create_vector_store(
        embeddings,
        connection_args={"uri": os.environ["MILVUS_URI"]},
        collection_name=collection_name,
        document_name=None
        )
    
    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def build_graph(self, collection_name):
        self.init_vector_store(collection_name)
        memory = MemorySaver()
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile(checkpointer=memory, interrupt_after=["generate"])

    def chatter(self, input_message: str, config: Dict = {"thread_id" : "1"}):
        for message, metadata in self.graph.stream(
                {"question": input_message},
                stream_mode="messages",
                config=config
        ):
            if metadata["langgraph_node"] == "generate":
                yield message.content
