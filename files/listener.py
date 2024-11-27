import speech_recognition as sr
import sounddevice as sd
import wavio
import numpy as np

recognizer = sr.Recognizer()
def get_speeach():
    
    # Capture the audio from the microphone
    with sr.Microphone() as source:
        print("Please speak something:")
        
        # Adjust for ambient noise (optional but helpful in noisy environments)
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Record the audio
        audio = recognizer.listen(source)

        try:
            # Convert speech to text using Google's speech recognition API
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            # If there's an error in the API request
            print(f"Could not request results; {e}")

#Get recorded version
def record_audio(duration=5, fs=44100):
    print("Recording audio for 5 seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    audio = np.squeeze(audio)
    wavio.write("outputs/output.wav", audio, fs, sampwidth=2)
    print("Recording finished.")
    return "output.wav"

def get_recorded(file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")


speech = get_speeach()
print(f"You said: {speech}")
