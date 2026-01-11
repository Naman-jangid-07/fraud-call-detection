# 🛡️ Real-Time Audio Fraud Detection System

An AI-powered system designed to analyze audio recordings and detect fraudulent activity in real-time. By combining **Speech-to-Text (STT)** with **Large Language Models (LLMs)**, this system identifies scam patterns, urgency tactics, and financial threats to protect users.

---

## 🏗️ Project Architecture



The system follows a modular architecture:
1. **Frontend**: Streamlit-based UI for audio upload and real-time alerts.
2. **Backend**: FastAPI handling requests and orchestrating the AI pipeline.
3. **Transcription**: OpenAI Whisper converting audio to processed text.
4. **Analysis**: Google Gemini AI performing pattern recognition and risk scoring.

---

## 💻 Tech Stack

- **Frameworks**: FastAPI, Streamlit
- **AI/ML**: Google Gemini (gemini-2.0-flash), OpenAI Whisper
- **NLP**: NLTK, TextBlob
- **Audio Processing**: Librosa, PyDub
- **Data**: Pandas, NumPy, Scikit-learn

---

## 📁 Repository Structure

```text
fraud-call-detector/
├── api/                  # FastAPI Backend (Fraud analysis logic)
├── src/                  # Core Source Code (Scrapers & Data Preprocessing)
├── data/                 # Data schemas (Raw data excluded via .gitignore)
├── app.py                # Streamlit Frontend application
├── requirements.txt      # Project dependencies
└── config.yaml           # Configuration settings
🚀 Key Features
Hybrid Detection: Combines keyword-based rules with Gemini AI’s semantic reasoning.

Sentiment Analysis: Evaluates urgency and emotional pressure in the speaker's voice.

Explainable AI: Provides a "Risk Level" along with a detailed reasoning for why a call was flagged.

Multi-Format Support: Handles WAV, MP3, and OGG files.
🛠️ Installation & Setup
Clone the repository:

Bash

git clone [https://github.com/Naman-jangid-07/fraud-call-detection.git](https://github.com/Naman-jangid-07/fraud-call-detection.git)
cd fraud-call-detection
Set up Environment Variables: Create a .env file and add your Gemini API Key:

Plaintext

GEMINI_API_KEY=your_key_here
Install Dependencies:

Bash

pip install -r requirements.txt
Run the Application:

Start the API: python run_api.py

Start the UI: streamlit run app.py

📈 Future Roadmap
[ ] Real-time phone line integration.

[ ] Support for regional languages.

[ ] Improved noise cancellation for better transcription.

Disclaimer: This project is part of my initial coding journey and is intended for educational purposes.