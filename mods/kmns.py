import streamlit as st
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import whisper
from speech_recognition import Microphone, Recognizer
import threading
import queue
import time

# Global variables
audio_queue = queue.Queue()
stop_signal = threading.Event()
transcription_results = []

# Load Whisper model
@st.cache_resource
def load_model(model_name="base"):
    return whisper.load_model(model_name)

# Audio capture function
def capture_audio(sr=16000, duration=5):
    recognizer = Recognizer()
    mic = Microphone(sample_rate=sr)

    while not stop_signal.is_set():
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.record(source, duration=duration)
                # Convert audio to numpy array
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                audio_queue.put(audio_data)
        except Exception as e:
            st.warning(f"Audio capture error: {e}")
            audio_queue.put(None)
            stop_signal.set()

# Audio processing and transcription function
def process_audio(sr=16000, thresh=15, k=2, model_name="base"):
    model = load_model(model_name)

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

            frames = np.arange(len(energy)) * hop_length
            vad_audio = np.zeros_like(audio_chunk)
            for i in range(len(frames) - 1):
                vad_audio[frames[i]:frames[i + 1]] = vad_mask[i]
            vad_audio[frames[-1]:] = vad_mask[-1]

            voiced_audio = audio_chunk[vad_audio > 0]
            if len(voiced_audio) == 0:
                continue

            # MFCC feature extraction
            mfccs = librosa.feature.mfcc(y=voiced_audio, sr=sr, n_mfcc=12)
            scaler = StandardScaler()
            mfccs_scaled = scaler.fit_transform(mfccs.T)

            # Speaker clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            speaker_labels = kmeans.fit_predict(mfccs_scaled)

            # Transcription with Whisper
            result = model.transcribe(voiced_audio)
            transcript = result["text"]

            # Map transcript to speakers
            segments = result.get("segments", [])
            for segment in segments:
                start, end = segment["start"], segment["end"]
                text = segment["text"]
                start_frame = int(start * sr / len(audio_chunk) * len(speaker_labels))
                end_frame = int(end * sr / len(audio_chunk) * len(speaker_labels))
                speaker_frames = speaker_labels[start_frame:end_frame]
                speaker = np.argmax(np.bincount(speaker_frames)) if len(speaker_frames) > 0 else -1

                speaker_label = "interviewee" if speaker == 1 else "interviewer"
                transcription_results.append(f"{speaker_label}: {text}")

        except Exception as e:
            transcription_results.append(f"Error: {e}")
            stop_signal.set()
            break

# Streamlit App
def main():
    st.title("Real-Time Transcription with Whisper")

    # Streamlit session state
    if "transcriptions" not in st.session_state:
        st.session_state.transcriptions = []

    # Sidebar
    model_name = st.sidebar.selectbox("Whisper Model", ["base", "small", "medium", "large"], index=0)
    start_button = st.sidebar.button("Start Transcription")
    stop_button = st.sidebar.button("Stop Transcription")

    # Start transcription
    if start_button:
        stop_signal.clear()
        st.session_state.transcriptions.clear()
        threading.Thread(target=capture_audio, args=(16000, 5), daemon=True).start()
        threading.Thread(target=process_audio, args=(16000, 15, 2, model_name), daemon=True).start()

    # Stop transcription
    if stop_button:
        stop_signal.set()
        st.warning("Stopping transcription...")

    # Display transcriptions
    st.header("Transcriptions")
    transcription_area = st.empty()

    while not stop_signal.is_set():
        if transcription_results:
            st.session_state.transcriptions.extend(transcription_results)
            transcription_results.clear()
        transcription_area.text("\n".join(st.session_state.transcriptions))
        time.sleep(0.5)

        
if __name__ == "__main__":
    main()