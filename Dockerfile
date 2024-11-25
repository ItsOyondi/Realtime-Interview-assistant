# Use a lightweight Python base image
FROM python:3.12-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    python3-dev \
    && apt-get clean
# Set the working directory
WORKDIR /app

# Copy files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8501

# Define environment variable to avoid Streamlit prompt
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit app
CMD ["streamlit", "run", "main.py"]
