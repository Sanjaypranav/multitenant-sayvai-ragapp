from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path
import json
from main import chatbot  # Import chatbot logic from main.py

app = FastAPI()

# Serve the index.html file at the root endpoint
@app.get("/")
def get_ui():
    index_file = Path(__file__).parent / "index.html"
    return HTMLResponse(index_file.read_text())

# WebSocket endpoint for chatbot communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = {"messages": [], "memory": []}
    try:
        while True:
            # Receive message from the frontend
            data = await websocket.receive_text()
            state["messages"].append(data)

            # Get response from chatbot logic
            result = chatbot(state)
            response_message = result["messages"][0]

            # Send the response back to the frontend
            await websocket.send_text(json.dumps({"message": response_message}))

    except WebSocketDisconnect:
        print("Client disconnected")

# Run the FastAPI server when the script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
