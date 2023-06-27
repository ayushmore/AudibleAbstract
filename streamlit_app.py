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
openai.api_key = st.secrets["OPENAI_API_KEY"]

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
