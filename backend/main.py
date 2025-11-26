import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from data import ppg_dict_to_dataframe, store_ppg_dataframe_to_csv
from infer import Inferer
from typing import List



class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages to connected clients."""
    def __init__(self):
        """Initializes the ConnectionManager with an empty list of active connections."""
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accepts a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Removes a WebSocket connection."""
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Sends a message to all connected WebSocket clients."""
        for connection in self.active_connections:
            try:
                print(f"Broadcasting message to a client: {connection.base_url}")
                print(f"Message content (truncated): {message[:100]}...")
                await connection.send_text(message)
            except WebSocketDisconnect:
                print(f"Client disconnected during broadcast: {connection.base_url}.")
                self.disconnect(connection)


app = FastAPI()
frontend_port = os.environ.get('FRONTEND_PORT') or 8080
FRONTEND_ORIGINS = [
    f"http://localhost:{frontend_port}",
    f"http://127.0.0.1:{frontend_port}",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
manager = ConnectionManager()

model_path = os.environ.get('PPG_MODEL_PATH') or None
inferer: Inferer = None
if model_path is not None:
    print(f"Using PPG model path from env: {model_path}")
    inferer = Inferer(model_path)

# Directory to store received PPG CSVs. Read from env `PPG_DATA_DIR`,
# fallback to the repository `data` folder (project root /data).
project_root = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('PPG_DATA_DIR') or (project_root / 'data'))
DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to stream PPG data to connected clients."""
    # Basic network check: only allow localhost clients
    client = websocket.client
    if client is not None:
        client_host = client[0]
        if client_host not in ("127.0.0.1", "::1", "localhost"):
            # refuse connection
            await websocket.close(code=1008)
            return

    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/")
async def receive_data(request: Request, data: dict):
    """Receives and processes PPG data in JSON format."""
    try:
        data = ppg_dict_to_dataframe(data)
    except Exception as e:
        print(f"Error while parsing JSON: {e}")
        return {"status": "error", "message": f"Error while parsing JSON: {e}"}

    print(f"Received data with {len(data)} samples.")

    if inferer is not None:
        results = inferer.classify(data)
        if results is not None:
            print("Classification results:")
            for channel in results:
                label = results[channel]["label"]
                confidence = results[channel]["confidence"]
                print(f"  Channel {channel}: {label} (confidence: {confidence:.3f})")


    # Broadcast to connected WebSocket clients.
    await manager.broadcast(data.to_json(orient="split"))

    # Save received dataframe to CSV.
    try:
        filepath = store_ppg_dataframe_to_csv(str(DATA_DIR), data)
        print(f"Saved received PPG data to CSV: {filepath}")
    except Exception as e:
        print(f"Error saving PPG data to CSV: {e}")

    return {"status": "ok", "received": True}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
