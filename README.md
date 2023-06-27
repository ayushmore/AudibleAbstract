# Audible Abstract 🗣️

Audible Abstract is a tool that leverages OpenAI's cutting-edge Whisper ASR and GPT-4 models to convert audio voice memos into concise text summaries.

This tool can help you transform lengthy audio records, meeting notes, or voice memos into easily digestible written summaries. It could be an ideal solution for students, professionals, researchers, or anyone in need of summarizing their voice memos.

## How to Use

Access the live application at [https://audibleabstract.streamlit.app/](https://audibleabstract.streamlit.app/).

## Features

1. **Audio Transcription**: Transform your voice memos into text using OpenAI's Whisper ASR model.
2. **Text Summarization**: Condense your transcription into a concise summary with OpenAI's GPT-4 model.
3. **Supported Audio Formats**: The application can process `.wav`, `.mp3`, and `.m4a` audio file formats.

## Installation

To install Audible Abstract, follow these steps:

- Clone the repository:

```bash
git clone https://github.com/yourusername/audible-abstract.git
```

- Install the required dependencies:

```bash
pip install -r requirements.txt
```

- Setup your OpenAI API key:

```bash
echo OPENAI_API_KEY=your_openai_api_key > .env
```

## Usage

To use Audible Abstract, navigate to the application URL on your local server or deployed server:

1. Upload your audio file.
2. Click on the "Transcribe and Summarize Audio" button.

The application will transcribe your audio and present a summary.

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to make a pull request.

