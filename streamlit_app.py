# import os
# import streamlit as st
# import whisper
# import pyaudio
# import wave
# import openai
# from dotenv import load_dotenv


# st.title("Thought Summarization App")

# # Use Replit secrets storage for the API key
# load_dotenv()
# openai.api_key = os.environ["OPENAI_API_KEY"]

# model = whisper.load_model("base")

# st.write("Whisper Model Loaded!")
# audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

# temp_audio_file_path = "./output.wav"

# if audio_file is not None:
#     audio_bytes = audio_file.read()
#     with open(temp_audio_file_path, "wb") as audio_out:
#         audio_out.write(audio_bytes)
#     st.write(f"Audio file written at: {temp_audio_file_path}")

# if st.sidebar.button("Transcribe/Summarize Audio") and audio_file is not None:
#     st.write("Transcribing Audio...")
#     result = model.transcribe(temp_audio_file_path)
#     st.write("Thought Summaries:")
#     text = result["text"]
#     st.write("Transcription finished, original text: ")
#     st.write(text)
#     prompt = (f"summarize this text: {text}")
#     st.write("Summarizing...")

#     # Get a response from GPT-3
#     # Define the system message
#     system_msg = 'Act as an expert summarizer who correctly captures the key points of the following transcription'

#     # Define the user message
#     # user_msg = 'Create a small dataset about total sales over the last year. The format of the dataset should be a data frame with 12 rows and 2 columns. The columns should be called "month" and "total_sales_usd". The "month" column should contain the shortened forms of month names from "Jan" to "Dec". The "total_sales_usd" column should contain random numeric values taken from a normal distribution with mean 100000 and standard deviation 5000. Provide Python code to generate the dataset, then provide the output in the format of a markdown table.'


#     # Create a dataset using GPT
#     response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
#                                             messages=[{"role": "system", "content": system_msg},
#                                             {"role": "user", "content": text}])

#     # Get the summary from the response
#     summary = response["choices"][0]["message"]["content"]

#     # Print the summary
#     st.write("Final Summary:")
#     st.write(summary)

# import os
# from pathlib import Path
# import tempfile
# from dotenv import load_dotenv
# import openai
# import whisper
# import streamlit as st

# # Constants
# ALLOWED_AUDIO_EXTENSIONS = ["wav", "mp3", "m4a"]
# OPENAI_MODEL = "gpt-3.5-turbo"

# # Streamlit Configuration
# st.set_page_config(
#     page_title="Thought Summarizer",
#     layout='wide',
#     initial_sidebar_state="expanded",
# )

# def main():
#     # Load Environment Variables
#     load_dotenv()
#     openai.api_key = os.getenv("OPENAI_API_KEY")

#     # Load Whisper Model
#     whisper_model = load_model()
    
#     # Sidebar
#     st.sidebar.title("Thought Summarizer")
#     st.sidebar.markdown("This app transcribes and summarizes audio files using Whisper ASR and GPT-3")
    
#     # Main
#     st.title("Upload an Audio File")
#     uploaded_audio_file = st.file_uploader("", type=ALLOWED_AUDIO_EXTENSIONS)
    
#     if uploaded_audio_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(uploaded_audio_file.getvalue())
#             temp_audio_file_path = tmp.name

#         if st.sidebar.button("Transcribe and Summarize Audio"):
#             # Transcribe Audio
#             transcription = transcribe_audio(whisper_model, temp_audio_file_path)

#             # Summarize Transcription
#             summary = summarize_text(transcription)
            
#             # Display Results
#             display_results(transcription, summary)
    
# def load_model():
#     st.sidebar.info("Loading Whisper model...")
#     model = whisper.load_model("base")
#     st.sidebar.success("Whisper model loaded successfully!")
#     return model

# def transcribe_audio(model, audio_path):
#     st.sidebar.info("Transcribing Audio...")
#     result = model.transcribe(audio_path)
#     st.sidebar.success("Transcription finished!")
#     return result["text"]

# def summarize_text(text):
#     st.sidebar.info("Summarizing...")
#     system_msg = 'Act as an expert summarizer who correctly captures the key points of the following transcription'
#     response = openai.ChatCompletion.create(
#         model=OPENAI_MODEL,
#         messages=[{"role": "system", "content": system_msg},
#                   {"role": "user", "content": text}]
#     )
#     st.sidebar.success("Summarization completed!")
#     return response.choices[0].message.content

# def display_results(transcription, summary):
#     st.markdown("## Transcription")
#     st.write(transcription)
#     st.markdown("## Summary")
#     st.write(summary)

# if __name__ == "__main__":
#     main()

import os
import tempfile
from dotenv import load_dotenv
import openai
import whisper
import streamlit as st

# Constants
ALLOWED_AUDIO_EXTENSIONS = ["wav", "mp3", "m4a"]
OPENAI_MODEL = "gpt-4"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit Configuration
st.set_page_config(
    page_title="Audible Abstract",
    layout='wide',
)

def main():

    # Load Whisper Model
    whisper_model = load_model()

    # App Header
    st.title("Audible Abstract")
    st.markdown("Transform your **audio voice memos** into **concise summaries** with our state-of-the-art OpenAI Whisper ASR and GPT-4. Please upload your audio files below.")

    uploaded_audio_file = st.file_uploader("", type=ALLOWED_AUDIO_EXTENSIONS)
    
    if uploaded_audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio_file.getvalue())
            temp_audio_file_path = tmp.name

        if st.button("Transcribe and Summarize Audio"):
            # Transcribe Audio
            transcription = transcribe_audio(whisper_model, temp_audio_file_path)

            # Display Transcription
            display_transcription(transcription)

            # Summarize Transcription
            summary = summarize_text(transcription)
            
            # Display Summary
            display_summary(summary)

def load_model():
    # with st.spinner("Loading Whisper model..."):
    model = whisper.load_model("base")
    # st.success("Whisper model loaded successfully!")
    return model

def transcribe_audio(model, audio_path):
    with st.spinner("Transcribing Audio..."):
        result = model.transcribe(audio_path)
    st.success("Transcription finished!")
    return result["text"]

def summarize_text(text):
    with st.spinner("Summarizing..."):
        system_msg = 'Act as an expert summarizer who correctly captures the key points of the following transcription'
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": text}]
        )
    st.success("Summarization completed!")
    return response.choices[0].message.content

def display_transcription(transcription):
    st.markdown("## 🎙️ Transcription:")
    st.write(transcription)
    st.markdown("---")  # separator

def display_summary(summary):
    st.markdown("## 📝 Summary:")
    st.write(summary)

if __name__ == "__main__":
    main()
