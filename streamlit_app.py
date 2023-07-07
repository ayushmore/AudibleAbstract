import os
import tempfile
from dotenv import load_dotenv
import openai
import whisper
import streamlit as st
import streamlit_analytics

ALLOWED_AUDIO_EXTENSIONS = ["wav", "mp3", "m4a"]
OPENAI_MODEL = "gpt-3.5-turbo"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

streamlit_analytics.start_tracking()

st.set_page_config(
    page_title=" 🗣️ Audible Abstract ",
    layout='wide',
)

def main():

    whisper_model = load_model()

    st.title(" 🗣️ Audible Abstract ")
    st.markdown("Transform your **audio voice memos** into **concise summaries** with our state-of-the-art OpenAI Whisper ASR and GPT-4. Please upload your audio files below.")

    uploaded_audio_file = st.file_uploader("", type=ALLOWED_AUDIO_EXTENSIONS)
    
    if uploaded_audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio_file.getvalue())
            temp_audio_file_path = tmp.name

        if st.button("Transcribe and Summarize Audio"):
            transcription = transcribe_audio(whisper_model, temp_audio_file_path)

            display_transcription(transcription)

            summary = summarize_text(transcription)
            
            display_summary(summary)

def load_model():
    model = whisper.load_model("base")
    return model

def transcribe_audio(model, audio_path):
    with st.spinner("Transcribing Audio..."):
        result = model.transcribe(audio_path)
    st.success("Transcription finished!")
    return result["text"]

def summarize_text(text):
    with st.spinner("Summarizing..."):
        system_msg = 'You are the world\'s best professional summarizer. Your role is to summarize and condense the key points and content in the text so that it distills the main messages and flow of the text. The transcript is provided in the following text: '
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
    st.markdown("---")

def display_summary(summary):
    st.markdown("## 📝 Summary:")
    st.write(summary)

streamlit_analytics.stop_tracking()

if __name__ == "__main__":
    main()
