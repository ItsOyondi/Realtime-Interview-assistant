import os
import openai
import streamlit as st
from io import BytesIO
from docx import Document
import spacy
import base64
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import asyncio
import time

# Initialize OpenAI client with Groq's API
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set or not available in secrets.")

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY
)

# Load spaCy model once to improve performance
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    st.error("The spaCy model 'en_core_web_sm' is not installed. Install it with: `python -m spacy download en_core_web_sm`")


async def get_groq_chat_response_async(question, primer, model="gemma2-9b-it"):
    #Asynchronous API call for faster response.
    try:
        messages = [
            {"role": "system", "content": primer},
            {"role": "user", "content": question},
        ]
        chat_completion = await asyncio.to_thread(
            client.chat.completions.create,
            messages=messages,
            model=model
        )
        response = chat_completion.choices[0].message.content
        return response
    except openai.OpenAIError as e:
        st.error(f"Error calling Groq API: {e}")
        return None


def read_file(file):
    """Read plain text file."""
    return file.read().decode("utf-8")


def read_word_document(file):
    """Read Word document and extract text."""
    try:
        doc = Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return ""


def extract_skills(text):
    """Extract skills and organizations from text using spaCy."""
    if not nlp:
        return []
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ORG"]]
    return set(skills)


async def main():
   
    # Set page configuration
    st.set_page_config(page_title="Interview Assistant", layout="wide")
    
    # Add custom CSS to style the file uploader widget
    st.markdown(
        """
        <style>
            .file-upload-container {
                max-width: 150px; /* Set the desired width */
                margin: 0 auto; /* Center align if needed */
                font-size: 12px; /* Reduce font size */
            }
            .file-upload-container .uploadedFileName {
                font-size: 10px; /* Adjust the size of uploaded filename text */
            }
            .file-upload-container button {
                padding: 2px 8px; /* Adjust padding of the upload button */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Sidebar settings
    st.sidebar.subheader("⚙️ Settings")
    # st.sidebar.info("Adjust the following:")
    response_length = st.sidebar.slider("Response Length (words)", 50, 500, 100)
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["llama-3.2-3b-preview", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "llama-2-13b", "gemma2-9b-it", "gpt-3.5-turbo"],
    )

    # st.sidebar.info("🎯 Customize your experience using these settings.")

    st.sidebar.subheader("📄 Upload Files")

    # Resume upload
    uploaded_resume = st.sidebar.file_uploader("Upload Resume", type=["txt", "docx"], key="resume_uploader")
    text = ""
    if uploaded_resume:
        if uploaded_resume.type == "text/plain":
            text = read_file(uploaded_resume)
        elif uploaded_resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = read_word_document(BytesIO(uploaded_resume.read()))

    # Job description upload
    uploaded_job_desc = st.sidebar.file_uploader("Upload Job Description", type=["txt", "docx"], key="job_desc_uploader")
    jobd = ""
    if uploaded_job_desc:
        if uploaded_job_desc.type == "text/plain":
            jobd = read_file(uploaded_job_desc)
        elif uploaded_job_desc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            jobd = read_word_document(BytesIO(uploaded_job_desc.read()))
    
    
    

    # Columns layout
    left_col, right_col = st.columns([3, 1])

    with left_col:
        st.title("🎤 RealTime Interview Assistant")
        st.markdown(
            "###### Prepare for interviews with AI-powered insights. Upload your resume and job description, record your question, and get expert-level responses in real-time."
        )
        # st.info("Click the button below to record your question.")
        if "stt_button" not in st.session_state:
            # Create a Speak button and store it in session state
            st.session_state.stt_button = Button(label="Ask Question", width=70)

        # JavaScript to start speech recognition
        st.session_state.stt_button.js_on_event(
            "button_click",
            CustomJS(code=""" 
                var recognition = new webkitSpeechRecognition();
                recognition.continuous = false;
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
            """),
        )
        result = streamlit_bokeh_events(
            st.session_state.stt_button,
            events="GET_TEXT",
            key="speech",
            refresh_on_update=True,
            override_height=50,
            debounce_time=0,
        )

        # Capture the custom event and display the recognized text
        # st.subheader("📜 Q&A")
        st.markdown("#### Response:")
        st.write(st.session_state.get("response", "The response will appear here after processing."))
        
        #add timer to return time taken to get response
        start_time = time.time()  
        # context = text[:2500] + jobd[:1000]
        if result and "GET_TEXT" in result:
            question = result.get("GET_TEXT")
            st.session_state["question"] = question

            primer = f"""
                    You are a professional interviewee, skilled at answering questions with clarity, confidence, and relevance.
                    - Your role is to provide thoughtful, well-structured, and natural answers based solely on the provided information.
                    - You are interviewing for this job ({jobd[:1000]}).
                    - Craft your responses to be professional yet conversational, showcasing my experience, skills, and enthusiasm for the role.
                    - Tailor your answers to align with the job requirements, strictly using specific examples from my experience in the provided text: {text[:2500]}.
                    - Do not add any external information or speculation. Base your responses only on the provided job description and personal experience.
                    - Keep your answers concise and engaging, within the specified word limit: {response_length}.
                    - Focus on demonstrating a deep understanding of the role, industry, and how my background uniquely qualifies me for the position.
                    - If you're unsure of how to respond, refer back to the provided experience for guidance.
                """
            response = await get_groq_chat_response_async(question, primer, model_choice)

            st.session_state["response"] = response
            
        end_time = time.time()  
        elapsed_time = end_time - start_time  
        #display elapsed time
        st.markdown(f"#### Response Time:")
        st.write(f"{elapsed_time:.2f} seconds")

    with right_col:
        #show question here
        st.markdown("#### Question:")
        st.info(st.session_state.get("question", "Your question will appear here after recording."))
        st.subheader("🔑 Skills from job description")
        if jobd:
            skills = extract_skills(jobd)
            with st.expander("Extracted Skills"):
                st.write(", ".join(skills))

    # Feedback Section
    st.markdown("### 🙋 Was this response helpful?")
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

    with feedback_col1:
        if st.button("👍 Yes"):
            st.success("Thanks for your feedback!")
    with feedback_col2:
        if st.button("👎 No"):
            st.warning("We'll strive to improve.")
    with feedback_col3:
        st.button("🤔 Neutral")


if __name__ == "__main__":
    asyncio.run(main())
