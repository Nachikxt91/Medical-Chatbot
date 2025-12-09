---
title: Medical AI Chatbot
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - medical
  - chatbot
  - rag
  - langchain
  - ai
---

# 🏥 Medical AI Chatbot

A retrieval‑augmented medical assistant that answers health‑related questions using a curated medical knowledge base and a modern LLM backend.  
This instance is deployed on Hugging Face Spaces at:

> https://huggingface.co/spaces/Nachikxt91/medical-chatbot

---

## ✨ Features

- **RAG (Retrieval‑Augmented Generation)** over a medical knowledge base  
- **Dynamic chain‑of‑thought style reasoning** for complex queries  
- **Emergency intent detection** to warn users and suggest contacting professionals  
- **Conversational memory** so follow‑up questions stay in context  
- **Safety‑first responses** with disclaimers and scope limits  

---

## 🧱 Tech Stack

- **Python** for core implementation  
- **LangChain** for retrieval and tool orchestration  
- **Groq / LLM backend** for fast generation  
- **Vector store** (e.g. Pinecone / FAISS) for document embeddings  
- **Web framework** (FastAPI or Flask) exposed as an HTTP API  
- **Docker** as the runtime for Hugging Face Spaces  

The exact stack may vary, but the Space is configured to run via Docker on port `7860`.

---


## 🚀 Running Locally

You can run the same image locally before pushing to Spaces.

### 1. Clone the repo

git clone <your-repo-url>
cd <your-repo-folder>

text

### 2. Set environment variables

Create a `.env` file in the project root (do **not** commit real secrets):

GROQ_API_KEY=your_groq_key
MODEL_NAME=llama-3.1-8b-instant
VECTOR_DB_API_KEY=your_vector_db_key # optional
VECTOR_DB_INDEX=medical-chatbot # optional

text

For Hugging Face Spaces, put these keys in the **Space settings → Variables** UI rather than in the repo.

### 3. Install and run (non‑Docker)

pip install -r requirements.txt

FastAPI example
uvicorn app.main:app --host 0.0.0.0 --port 7860

text

Then open `http://localhost:7860` in your browser or hit the API from a client.

---

## 🐳 Docker Usage

The Space is configured to use Docker (`sdk: docker` with `app_port: 7860`).

### Build and run locally

docker build -t medical-chatbot .
docker run -p 7860:7860 --env-file .env medical-chatbot

text

Hugging Face Spaces will perform a similar build automatically from this `Dockerfile`.

---

## 📡 API Example

Assuming the app exposes a `/chat` endpoint:

curl -X POST "http://localhost:7860/chat"
-H "Content-Type: application/json"
-d '{
"messages": [
{"role": "user", "content": "I have a headache and mild fever. What could it be?"}
]
}'

text

Example response (shape, not actual medical advice):

{
"role": "assistant",
"content": "I am not a doctor, but common causes of headache and mild fever include...",
"metadata": {
"sources": [
"UpToDate summary on headaches",
"WHO general guidance..."
]
}
}

text

Adjust the path and payload to match your actual API.

---

## ⚠️ Medical Disclaimer

- This chatbot is **not a doctor** and **does not provide medical diagnosis or treatment**.  
- All responses are for **information and education only**.  
- Users should always consult a licensed healthcare professional for medical decisions, emergencies, or diagnosis.  
- In case of an emergency (e.g., chest pain, difficulty breathing, severe bleeding), users should **call emergency services immediately** and **not rely on this chatbot**.

---

## 🧪 Testing

Basic testing commands (adapt as needed):

Unit tests
pytest

Linting
ruff check .

text

---

## 🔐 Security & Privacy

- No real API keys are stored in the repository; they are injected via environment variables.  
- Do not log sensitive user data in production.  
- If using an external vector DB, ensure transport is encrypted (HTTPS/TLS).  

---

## 📜 License

Specify your license here, for example:

MIT License
Copyright (c) 2025 <Your Name>

text

---

## 🙋‍♂️ Maintainer

Built by an AI/ML engineer focusing on production‑grade RAG systems and healthcare‑oriented chatbots.  
Feel free to open issues or pull requests to improve the system.
