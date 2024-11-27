import os
import openai
import dotenv
import whisper_listener
import streamlit as st

# Load environment variables
dotenv.load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Verify the API key is set
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize OpenAI client with Groq's API
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=groq_api_key
)

# Function to get the response from the Groq API
def get_groq_chat_response(question, model="gemma2-9b-it"):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give an answer to this question as an interviewee in data science interview: Make the answer brief and keep it professional. " + question},
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except openai.OpenAIError as e:
        st.error(f"Error calling Groq API: {e}")
        return None

# Streamlit app UI
st.title("Real-Time Interview Assistant")
st.text("Developed by Joe Oyondi")
st.write("Record your question, and get an immediate response with help of AI!")

# Automatically record and transcribe
st.info("Click the button below to record your question.")

if st.button("Record and Get Response"):
    try:
        # Record and transcribe question
        st.info("Recording... Speak now!")
        question = whisper_listener.record_and_transcribe()
        st.success("Recording complete! Transcription in progress...")
        
        # Display the transcribed question
        st.subheader("Question:")
        st.write(question)

        # Fetch response from the Groq API
        response = get_groq_chat_response(question)
        
        # Display the response
        if response:
            st.subheader("Response:")
            st.write(response)
        else:
            st.error("No response received from the Groq API.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
###################################

# Function to record audio from the microphone
# def record_audio(duration=17):
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
#     frames = []
    
#     st.info("Recording... Speak now!")
#     for _ in range(0, int(16000 / 1024 * duration)):
#         data = stream.read(1024)
#         frames.append(data)
    
#     st.success("Recording complete!")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     # Convert the frames to bytes
#     audio_data = b''.join(frames)
#     return audio_data

# Function to convert raw audio bytes to numpy array for whisper
# def audio_bytes_to_np_array(audio_data, rate=16000):
#     # Convert the raw bytes into a numpy array of int16 format (16-bit PCM)
#     audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to float range (-1.0 to 1.0)
#     return audio_np
