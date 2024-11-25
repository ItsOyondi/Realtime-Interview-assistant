import whisper
import pyaudio
import wave
import numpy as np
import time

# Configuration for real-time audio capture
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono channel
RATE = 16000  # Sample rate for Whisper compatibility
CHUNK = 1024  # Buffer size
OUTPUT_FILENAME = "outputs/recorded_audio.wav"
RECORD_DURATION = 18 
SILENCE_THRESHOLD = 1000  # Silence threshold to determine speech peresence

# Initialize the Whisper model
model = whisper.load_model("base")  # You can choose a different model, like "small", "medium", etc.

# Function to capture and process audio in real-time for a specific duration
def record_and_transcribe():
    audio = pyaudio.PyAudio()
    
    # Open the audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Listening for 30 seconds...")

    frames = []
    start_time = time.time()  # Record the start time
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # Stop recording after 30 seconds
        if time.time() - start_time > RECORD_DURATION:
            print("Recording time reached 30 seconds, stopping...")
            break
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a file
    save_audio(frames)
    
    # Transcribe the saved audio
    transcription = transcribe_audio(OUTPUT_FILENAME)
    print(f"Final Transcription: {transcription}")
    return transcription

# Function to save the audio frames to a file
def save_audio(frames):
    with wave.open(OUTPUT_FILENAME, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Function to transcribe the audio using Whisper
def transcribe_audio(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

# Run the audio capture and transcription process
# record_and_transcribe()
