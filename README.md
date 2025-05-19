🎥 YouTube RAG Chatbot
An AI-powered chatbot that answers user questions about any YouTube video using Retrieval-Augmented Generation (RAG). Built with Streamlit, LangChain, FAISS, and Google Gemini Pro.

🚀 Features
🔍 Extracts transcript from YouTube video using video ID

🧠 Uses semantic search with FAISS for relevant chunk retrieval

🤖 Generates accurate, context-aware answers with Gemini Pro

💻 Simple, interactive Streamlit UI

🛠️ Tech Stack
Streamlit – For building the web UI

LangChain – For chaining components

FAISS – For vector similarity search

SentenceTransformers – For creating embeddings

YouTube Transcript API – To fetch video transcripts

Google Generative AI – Gemini Pro for language generation

🧪 How It Works
User inputs a YouTube video ID and their query

Transcript is fetched (if available)

Transcript is chunked and embedded

Relevant chunks are retrieved based on the query

Gemini Pro generates an answer grounded in retrieved context

⚙️ Installation
bash
Copy
Edit
git clone https://github.com/your-username/youtube-rag-chatbot.git
cd youtube-rag-chatbot
pip install -r requirements.txt
Set your Google API key in a .env file:

env
Copy
Edit
GOOGLE_API_KEY=your_google_api_key
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
📝 Example
Ask: “What is the main topic discussed in the video?”
Provide: YouTube ID like dQw4w9WgXcQ
Get: An AI-generated answer based on the actual content

📌 Limitations
Works only with videos that have English captions enabled

Requires a valid Google API key for Gemini Pro access

