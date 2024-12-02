import streamlit as st
import asyncio
import websockets
import numpy as np
from queue import Queue
import threading

# Queues and shared variables
audio_queue = Queue()
stop_signal = threading.Event()

# WebSocket server for audio capture
async def audio_server(websocket, path):
    async for message in websocket:
        try:
            # Convert the received message to audio data
            audio_data = np.array(eval(message), dtype=np.float32)
            audio_queue.put(audio_data)
        except Exception as e:
            print(f"WebSocket error: {e}")
            stop_signal.set()

# Function to start WebSocket server
def start_audio_server():
    # Function to run the asyncio WebSocket server in a separate thread
    def run_server():
        try:
            # Run an asyncio event loop in this thread
            asyncio.run(websocket_main())
        except Exception as e:
            print(f"Error running WebSocket server: {e}")

    # Start the server thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

# Main asyncio function for the WebSocket server
async def websocket_main():
    server = await websockets.serve(audio_server, "localhost", 8080)
    print("WebSocket server started on ws://localhost:8080")
    await server.wait_closed()

# Streamlit UI
if st.button("Start WebSocket Server"):
    start_audio_server()
    st.success("WebSocket server started on ws://localhost:8080!")
