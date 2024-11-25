import streamlit as st
import numpy as np
import sounddevice as sd
import io
import wave
import whisper
import tempfile
import os

# Load Whisper model (You can use "base", "small", "medium", or "large" based on your needs)
model = whisper.load_model("base")

# Function to record audio
def record_audio(duration=18, sample_rate=44100):
    st.write("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is done
    return audio_data

# Function to convert recorded audio to byte stream
def audio_to_bytes(audio_data, sample_rate=44100):
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wf:
        wf.setnchannels(1)  
        wf.setsampwidth(2) 
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    byte_io.seek(0)  
    return byte_io

def transcribe_audio(byte_io):
    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(byte_io.read())
        temp_file_path = temp_file.name  

    result = model.transcribe(temp_file_path)
    
    os.remove(temp_file_path)
    
    return result['text']

# # Button to start recording
# if st.button("Start Recording"):
#     # Record audio from the microphone (5 seconds duration)
#     audio_data = record_audio(duration=5)
    
#     # Convert the audio to byte stream and play it
#     byte_io = audio_to_bytes(audio_data)
#     st.audio(byte_io, format="audio/wav")
    
#     st.success("Recording complete!")
    
#     # Transcribe the audio using Whisper
#     st.write("Transcribing the audio...")
#     transcription = transcribe_audio(byte_io)
#     st.write("Transcription:")
#     st.write(transcription)

# else:
#     st.info("Click the button to start recording.")
