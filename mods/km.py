import streamlit as st
import asyncio
import websockets
import numpy as np
import time
from queue import Queue
import threading
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from whisper import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Queues and shared variables
audio_queue = Queue()
transcription_results = []
stop_signal = threading.Event()

# WebSocket server for audio capture
async def audio_server(websocket, path=None):
    async for message in websocket:
        try:
            # Convert the received message to audio data
            audio_data = np.array(eval(message), dtype=np.float32)
            audio_queue.put(audio_data)
        except Exception as e:
            logger.error(f"Error in audio_server: {e}", exc_info=True)
            stop_signal.set()


# Main asyncio function to start the WebSocket server
async def websocket_main():
    server = await websockets.serve(audio_server, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

# Function to start WebSocket server in a thread
def start_audio_server():
    def run_server():
        asyncio.run(websocket_main())

    # Start the WebSocket server in a new thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

# JavaScript for browser audio capture
def inject_audio_capture_js():
    st.components.v1.html(
        """
        <script>
            (async () => {
                const maxRetries = 5;
                let retries = 0;

                async function connectWebSocket() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        const audioContext = new AudioContext();
                        const mediaStreamSource = audioContext.createMediaStreamSource(stream);
                        const processor = audioContext.createScriptProcessor(4096, 1, 1);

                        mediaStreamSource.connect(processor);
                        processor.connect(audioContext.destination);

                        const webSocket = new WebSocket('ws://localhost:8765');
                        
                        webSocket.onopen = () => {
                            console.log("WebSocket connection opened.");
                            processor.onaudioprocess = (audioProcessingEvent) => {
                                if (webSocket.readyState === WebSocket.OPEN) {
                                    const audioData = audioProcessingEvent.inputBuffer.getChannelData(0);
                                    const audioArray = Array.from(audioData);
                                    webSocket.send(JSON.stringify(audioArray));
                                }
                            };
                        };

                        webSocket.onerror = (error) => {
                            console.error("WebSocket error:", error);
                            webSocket.close();
                        };

                        webSocket.onclose = () => {
                            console.log("WebSocket connection closed.");
                        };
                    } catch (error) {
                        if (retries < maxRetries) {
                            retries++;
                            console.warn(`Retrying WebSocket connection (${retries}/${maxRetries})...`);
                            setTimeout(connectWebSocket, 1000);
                        } else {
                            console.error("Failed to connect to WebSocket server after maximum retries.", error);
                        }
                    }
                }

                connectWebSocket();
            })();
        </script>
        """,
        height=0,
    )

# Capture audio function
def capture_audio():
    while not stop_signal.is_set():
        try:
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                # print(f"Captured audio chunk: {audio_chunk[:10]}")  # Debugging output
        except Exception as e:
            logger.error(f"Error in capture_audio: {e}", exc_info=True)
            stop_signal.set()
            break

# Audio processing and transcription function
def process_audio(sr=16000, thresh=15, k=3, model_name="base"):
    model = load_model(model_name)
    speaker_audio_buffer = []
    current_speaker = None

    while not stop_signal.is_set():
        try:
            if audio_queue.empty():
                time.sleep(0.1)
                continue

            audio_chunk = audio_queue.get()
            if audio_chunk is None:
                break

            # Voice Activity Detection (VAD)
            frame_length = 2048
            hop_length = 512
            energy = librosa.feature.rms(y=audio_chunk, frame_length=frame_length, hop_length=hop_length).flatten()
            threshold = np.percentile(energy, thresh)
            vad_mask = energy > threshold

            voiced_audio = audio_chunk[vad_mask]
            if len(voiced_audio) == 0:
                if current_speaker is not None and len(speaker_audio_buffer) > 0:
                    accumulated_audio = np.concatenate(speaker_audio_buffer)
                    result = model.transcribe(accumulated_audio)
                    transcription_results.append(f"{current_speaker}: {result['text']}")
                    speaker_audio_buffer.clear()
                    current_speaker = None
                continue

            # MFCC feature extraction and speaker clustering
            mfccs = librosa.feature.mfcc(y=voiced_audio, sr=sr, n_mfcc=12)
            scaler = StandardScaler()
            mfccs_scaled = scaler.fit_transform(mfccs.T)

            kmeans = KMeans(n_clusters=k, random_state=42)
            speaker_labels = kmeans.fit_predict(mfccs_scaled)

            predominant_speaker = np.argmax(np.bincount(speaker_labels)) if len(speaker_labels) > 0 else -1
            speaker_label = "others"
            if predominant_speaker == 1:
                speaker_label = "interviewee"
            elif predominant_speaker == 2:
                speaker_label = "interviewer"

            if current_speaker == speaker_label:
                speaker_audio_buffer.append(voiced_audio)
            else:
                if current_speaker is not None and len(speaker_audio_buffer) > 0:
                    accumulated_audio = np.concatenate(speaker_audio_buffer)
                    result = model.transcribe(accumulated_audio)
                    transcription_results.append(f"{current_speaker}: {result['text']}")
                    speaker_audio_buffer.clear()

                current_speaker = speaker_label
                speaker_audio_buffer.append(voiced_audio)

            # Update transcription results in the UI
            st.session_state.transcriptions = transcription_results.copy()

        except Exception as e:
            transcription_results.append(f"Error: {e}")
            stop_signal.set()
            break

# # Streamlit UI
# if st.button("Start Recording"):
#     stop_signal.clear()
#     start_audio_server()
#     time.sleep(1)  # Allow the WebSocket server to start
#     inject_audio_capture_js()
#     threading.Thread(target=capture_audio, daemon=True).start()
#     threading.Thread(target=process_audio, daemon=True).start()

# if st.button("Stop Recording"):
#     stop_signal.set()

# # Display transcriptions in real-time
# if "transcriptions" not in st.session_state:
#     st.session_state.transcriptions = []

# st.write("Transcriptions:")
# for transcription in st.session_state.transcriptions:
#     st.write(transcription)
