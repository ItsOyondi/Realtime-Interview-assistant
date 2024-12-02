import asyncio
import websockets

async def test_server(websocket, path):
    print("Client connected")
    await websocket.send("Hello from server")
    await websocket.close()

async def main():
    server = await websockets.serve(test_server, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()  # Keeps the server running

asyncio.run(main())
