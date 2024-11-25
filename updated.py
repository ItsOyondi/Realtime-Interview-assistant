import os
import openai
import streamlit as st
import webstr as ws
from io import BytesIO
from docx import Document
import spacy
import base64
import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events


# Initialize OpenAI client with Groq's API
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set or not available in secrets.")

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY
)

def get_groq_chat_response(question, primer, max_words, model="llama-3.2-3b-preview"):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": primer + " " + question},
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        response = chat_completion.choices[0].message.content
        # Truncate response to the specified number of words
        truncated_response = " ".join(response.split()[:max_words])
        return truncated_response
    except openai.OpenAIError as e:
        st.error(f"Error calling Groq API: {e}")
        return None

def read_file(file):
    return file.read().decode("utf-8")

def read_word_document(file):
    try:
        doc = Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return ""

def extract_skills(text):
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ORG"]]
        return skills
    except OSError:
        st.error("The spaCy model 'en_core_web_sm' is not installed. Please install it with: `python -m spacy download en_core_web_sm`")
        return []

st.set_page_config(layout="wide")

st.title("RealTime Interview Assistant")
# Sidebar
st.sidebar.title("Settings")
st.sidebar.info("Adjust the following:")
response_length = st.sidebar.slider("Response Length (words)", 50, 500, 150)
model_choice = st.sidebar.selectbox("Choose Model", ["llama-3.2-3b-preview", "llama-2-13b"])
# Create three columns with specified widths
left_col, middle_col, right_col = st.columns([1, 1, 2])

with left_col:
    st.subheader("Uploads")
    uploaded_resume = st.file_uploader("Resume", type=["txt", "docx"])
    text = ""
    if uploaded_resume is not None:
        if uploaded_resume.type == "text/plain":
            text = read_file(uploaded_resume)
        elif uploaded_resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = read_word_document(BytesIO(uploaded_resume.read()))

    uploaded_job_desc = st.file_uploader("Job Description", type=["txt", "docx"])
    jobd = ""
    if uploaded_job_desc is not None:
        if uploaded_job_desc.type == "text/plain":
            jobd = read_file(uploaded_job_desc)
        elif uploaded_job_desc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            jobd = read_word_document(BytesIO(uploaded_job_desc.read()))

    if jobd:
        skills = extract_skills(jobd)
        st.markdown("### Key Skills from Job Description")
        st.write(", ".join(skills))

with middle_col:
    st.subheader("Record & Process")
    st.info("Click the button below to record your question.")
    
    if st.button("🎙️ Start Recording"):
        with st.spinner("Recording in progress..."):
            try:
                audio_data = ws.record_audio(duration=18)
                byte_io = ws.audio_to_bytes(audio_data)
                st.success("Recording complete!")
                st.write("Processing question...")
                question = ws.transcribe_audio(byte_io)

                primer = f"""
                    ** You are an interview and preparation assistant. 
                    ** You can generate new interview questions based on the job description provided here: {jobd}.
                    ** You can answer questions as an interviewee excellently and professionally. Avoid giving detailed examples and stick on the main point.
                    ** Give brief and concise answers.
                    ** If asked about introducing myself, write a brief, and best elevator pitch using my experience in this text {text}.  
                    ** You may use this information for some cases of my experiences:
                    """
                # Retrieve response length from sidebar
                # response_length = st.sidebar.slider("Response Length (words)", 50, 500, 150)
                response = get_groq_chat_response(question, primer, response_length)

                st.session_state["question"] = question
                st.session_state["response"] = response
            except Exception as e:
                st.error(f"An error occurred: {e}")

with right_col:
    st.subheader("Question & Response")
    if "question" in st.session_state:
        st.text_area("Question", st.session_state["question"], height=70)
    else:
        st.text_area("Question", "Your question will appear here after recording.", height=80)

    if "response" in st.session_state:
        st.text_area("Response", st.session_state["response"], height=200)
    else:
        st.text_area("Response", "The response will appear here after processing.", height=300)



st.markdown("### Was this response helpful?")
if st.button("👍 Yes"):
    st.success("Thank you for your feedback!")
if st.button("👎 No"):
    st.warning("We'll work to improve!")
