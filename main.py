import openai
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

if not openai.api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Define State structure for memory
class State(TypedDict):
    messages: list
    memory: list

# Initialize the StateGraph and MemorySaver
graph_builder = StateGraph(State)
memory_saver = MemorySaver()

# Function to get a response from OpenAI's API
def get_openai_response(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"

# Chatbot logic with memory handling
def chatbot(state: State):
    memory = state.get("memory", [])
    user_input = state["messages"][-1].strip()

    memory.append(f"User: {user_input}")
    conversation_context = "\n".join(memory)
    response = get_openai_response(conversation_context + "\nAssistant:")

    memory.append(f"Assistant: {response}")
    state["memory"] = memory

    return {"messages": [response]}

# Define chatbot node
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph
graph = graph_builder.compile(checkpointer=memory_saver)
