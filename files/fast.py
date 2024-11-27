import os
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import openai
import webstr as ws
import io

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client with Groq's API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY
)

# Function to get the response from the Groq API
def get_groq_chat_response(question, primer, model="llama-3.2-3b-preview"):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": primer + " " + question},
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except openai.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {e}")

# Load resume text
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

resume_text = read_file("resume.txt")

# Primer template
primer_template = f"""
    Give an answer to this question as an interviewee in a data science interview:
    ** Make the answer brief but detailed and professional.
    ** If asked about introducing myself, write a brief, and best elevator pitch using my experience in this text {resume_text}.
    ** The elevator pitch should be shorter, sound more senior.
    ** You may use this information for some cases of my experiences:
"""

# Endpoint to handle recorded audio and process the response
@app.post("/process-audio/")
async def process_audio(audio_data: bytes = Form(...)):
    try:
        # Convert audio bytes to a byte stream
        byte_io = io.BytesIO(audio_data)

        # Transcribe the audio using Whisper
        question = ws.transcribe_audio(byte_io)

        # Generate response using Groq API
        response = get_groq_chat_response(question, primer_template)

        # Return the transcribed question and generated response
        return {
            "question": question,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Root endpoint for testing
@app.get("/")
def root():
    return {"message": "Welcome to the Real-Time Interview Assistant API!"}
