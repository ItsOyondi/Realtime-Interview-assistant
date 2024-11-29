import streamlit as st
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.embed import components
from streamlit_server_state import server_state
import numpy as np
import asyncio

# Streamlit WebSocket for receiving audio chunks
if "audio_chunks" not in server_state:
    server_state.audio_chunks = []

# Bokeh Visualization
source = ColumnDataSource(data={"x": [], "y": []})
plot = figure(title="Real-time Audio Visualization", height=300, x_axis_label="Time", y_axis_label="Amplitude")
plot.line(x="x", y="y", source=source, line_width=2)

bokeh_script, bokeh_div = components(plot)

# Streamlit UI
st.title("Streamlit + Bokeh Audio Visualization")
st.write("Click the **Start Recording** button below to begin capturing audio.")

st.components.v1.html(f"""
<!DOCTYPE html>
<html>
    <head>
        <script>
            let audioContext, mediaStream, socket;

            function startAudioCapture() {{
                socket = new WebSocket("ws://localhost:8501/audio");
                socket.onopen = () => console.log("WebSocket connected!");

                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                navigator.mediaDevices.getUserMedia({{ audio: true }}).then((stream) => {{
                    mediaStream = audioContext.createMediaStreamSource(stream);
                    const processor = audioContext.createScriptProcessor(4096, 1, 1);

                    processor.onaudioprocess = (event) => {{
                        const audioData = event.inputBuffer.getChannelData(0);
                        socket.send(audioData.buffer);
                    }};

                    mediaStream.connect(processor);
                    processor.connect(audioContext.destination);
                }});
            }}

            function stopAudioCapture() {{
                if (mediaStream) {{
                    mediaStream.mediaStream.getTracks().forEach((track) => track.stop());
                }}
                if (socket) {{
                    socket.close();
                }}
                if (audioContext) {{
                    audioContext.close();
                }}
            }}
        </script>
    </head>
    <body>
        <button onclick="startAudioCapture()">Start Recording</button>
        <button onclick="stopAudioCapture()">Stop Recording</button>
    </body>
</html>
""", height=150)

# Display the Bokeh plot in Streamlit
st.write(bokeh_div, unsafe_allow_html=True)
st.write(bokeh_script, unsafe_allow_html=True)

# Update Bokeh plot with received audio chunks
async def update_audio_plot():
    while True:
        if server_state.audio_chunks:
            chunk = server_state.audio_chunks.pop(0)
            audio_data = np.frombuffer(chunk, dtype=np.float32)
            x = np.linspace(0, len(audio_data), len(audio_data))
            source.stream({"x": x, "y": audio_data}, rollover=500)
        await asyncio.sleep(0.05)

asyncio.run(update_audio_plot())
