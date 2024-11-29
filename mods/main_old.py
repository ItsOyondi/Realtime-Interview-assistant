import streamlit as st
import dotenv
import os
from PyPDF2 import PdfReader  
import openai

# Load environment variables
dotenv.load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize Groq client
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=groq_api_key
)

def extract_text_from_pdf(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            st.error(f"Error: File not found at {pdf_path}")
            return None

        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or "" 
            return text.strip()  
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return None

def get_groq_chat_response(messages, model="gemma2-9b-it"):
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except openai.OpenAIError as e:
        st.error(f"Error calling Groq API: {e}")
        return None

def process_pdf_and_get_response(pdf_path, query, model="gemma2-9b-it"):
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    if pdf_text is None:
        return None

    # Truncate text if too long for API
    max_chars = 10000 
    if len(pdf_text) > max_chars:
        st.warning(f"PDF content is too long. Truncating to {max_chars} characters.")
        pdf_text = pdf_text[:max_chars]

    # Prepare messages for the Groq API
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about provided PDF documents."},
        {"role": "user", "content": f"Here's the content of a PDF:\n{pdf_text}\n\nMy question is: {query}"},
    ]

    # Get response from Groq API
    try:
        response = get_groq_chat_response(messages, model)
        return response
    except Exception as e:
        st.error(f"Error getting Groq response: {e}")
        return None

# Streamlit UI components
st.title("PDF Document Sumarizer")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Query input
query = st.text_input("Enter your question", "Give me a summary of the document.")

# Button to start processing
if uploaded_file is not None and query:
    process_button = st.button("Process PDF")

    if process_button:
        # Save uploaded PDF temporarily
        pdf_path = f"./temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the PDF and get the response
        with st.spinner('Processing...'):
            response = process_pdf_and_get_response(pdf_path, query)

        if response:
            st.subheader("Answer:")
            st.write(response)
        else:
            st.warning("No response received or an error occurred.")
else:
    st.info("Please upload a PDF and enter a question.")
