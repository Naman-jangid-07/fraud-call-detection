from fastapi import WebSocket
from typing import List
import json

class AlertManager:
    def __init__(self):
        # Store active connections to users
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_alert(self, fraud_data: dict):
        """Broadcasts the fraud score and status to the UI."""
        message = json.dumps(fraud_data)
        for connection in self.active_connections:
            await connection.send_text(message)

# Global instance to be used by the FastAPI router
manager = AlertManager()