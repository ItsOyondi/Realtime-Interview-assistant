import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os

# Directory to save audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Counter for audio file names
if "audio_counter" not in st.session_state:
    st.session_state["audio_counter"] = 0

# Streamlit interface
st.title("Audio Recorder and Saver")
st.write("Click the button to record audio. The audio will be saved locally.")

# Record audio
audio_bytes = audio_recorder()

if audio_bytes:
    # Increment the counter
    st.session_state["audio_counter"] += 1

    # Save audio file
    file_path = os.path.join(
        AUDIO_DIR, f"recording_{st.session_state['audio_counter']}.mp3"
    )
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    st.success(f"Audio saved to {file_path}")

    # Play the audio
    st.audio(audio_bytes, format='audio/mp3')
