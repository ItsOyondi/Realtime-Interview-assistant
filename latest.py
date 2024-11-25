import os
import openai
import streamlit as st
# import webstr as ws
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
            {"role": "system", "content": primer},
            {"role": "user", "content":  question},
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        response = chat_completion.choices[0].message.content
        
        return response
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

def main():
    st.set_page_config(layout="wide")

    st.title("RealTime Interview Assistant")
    # Sidebar
    st.sidebar.title("Settings")
    st.sidebar.info("Adjust the following: ")
    response_length = st.sidebar.slider("Response Length (words)", 50, 500, 150)
    model_choice = st.sidebar.selectbox("Choose Model", ["llama-3.2-3b-preview","llama-3.1-70b-versatile","mixtral-8x7b-32768", "llama-2-13b", "gemma2-9b-it"])
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
        
        if 'stt_button' not in st.session_state:
            # Create a Speak button and store it in session state
            st.session_state.stt_button = Button(label="Speak", width=100)
        
        st.bokeh_chart(st.session_state.stt_button)

        # Define custom JavaScript to start speech recognition
        st.session_state.stt_button.js_on_event("button_click", CustomJS(code="""
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onresult = function (e) {
                var value = '';
                for (var i = e.resultIndex; i < e.results.length; i++) {
                    if (e.results[i].isFinal) {
                        value += e.results[i][0].transcript;
                    }
                }
                
                if (value !== '') {
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                }
            };
            
            recognition.start();
        """))

        # Capture the custom event and display the recognized text
        result = streamlit_bokeh_events(
            st.session_state.stt_button,
            events="GET_TEXT",
            key="speech",
            refresh_on_update=True,
            override_height=75,
            debounce_time=0,
        )
        if result is not None:
            if "GET_TEXT" in result:
                question = result.get("GET_TEXT")
                primer = f"""
                    You are an interview preparation assistant. 
                    - Your role is to generate and answer interview questions based on the provided job description ({jobd}) and resume ({text}).
                    - Respond to questions professionally and excellently, as an interviewee would.
                    - Your response must not exceed the specified word limit. The word limit is specified here: {response_length}.
                    - Ensure answers are brief, concise, and focused on the main points, avoiding unnecessary details or examples.
                    - Base all your answers, including the elevator pitch, on my experience as outlined in the provided text.
                    - Be specific, avoid filler words, and answer each question exactly as asked.
                    - Use the above text provided for context about my experiences when applicable.
                    """

                # Retrieve response length from sidebar
                response = get_groq_chat_response(question, primer, response_length)

                st.session_state["question"] = question
                st.session_state["response"] = response

    with right_col:
        st.subheader("Question & Response")
        if "question" in st.session_state:
            st.markdown("### Question:")
            st.write(st.session_state["question"])
        else:
            st.markdown("### Question:")
            st.write("Your question will appear here after recording.")

        if "response" in st.session_state:
            st.write("### Response:")
            st.write(st.session_state["response"])
        else:
            st.write("### Response:")
            st.write("The response will appear here after processing.")



    st.markdown("### Was this response helpful?")
    if st.button("👍 Yes"):
        st.success("Thank you for your feedback!")
    if st.button("👎 No"):
        st.warning("We'll work to improve!")

if __name__ == "__main__":
    main()