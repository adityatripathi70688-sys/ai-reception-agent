# AI Reception Agent

An AI-powered reception and call-triage system that converts customer conversations into structured operational insights using **LLMs, RAG, speech-to-text, and PostgreSQL**.

## 🚀 Live Demo

[Open the Live App](https://ai-reception-agent-ojwm8cyrb9x4sjwwhchbmn.streamlit.app/)

## ✨ Features

* 🎙️ **Speech-to-Text** — Transcribes audio using Groq Whisper.
* 🤖 **LLM-Based Call Analysis** — Extracts structured information from conversations.
* 🧠 **RAG-Based Policy Retrieval** — Retrieves relevant company policies before generating decisions.
* 📊 **Call Triage** — Identifies intent, sentiment, urgency, and risk score.
* 👤 **Caller Information Extraction** — Extracts caller name and callback number when available.
* 🏢 **Department Assignment** — Determines the appropriate department for the request.
* ⚡ **Action Recommendation** — Generates the required operational follow-up.
* 🗄️ **PostgreSQL Logging** — Stores analyzed calls and operational metadata.
* 📈 **Streamlit Dashboard** — Displays results and historical call records.
* 🔄 **LLM Fallback Handling** — Supports Groq and Anthropic providers.

## 🏗️ How It Works

```text
Customer Conversation
        ↓
Audio / Text Input
        ↓
Groq Whisper
(Speech-to-Text)
        ↓
RAG Policy Retrieval
        ↓
LLM Analysis
(Groq / Anthropic)
        ↓
Structured JSON Extraction
        ↓
Validation & Guardrails
        ↓
PostgreSQL
        ↓
Streamlit Dashboard
```

## 🧠 RAG Pipeline

The system uses Retrieval-Augmented Generation to provide relevant operational policy context to the LLM before analyzing a conversation.

### Process

1. Customer conversation is received.
2. The system searches the internal policy knowledge base.
3. Relevant policy content is retrieved using semantic similarity.
4. Retrieved policy context is provided to the LLM.
5. The LLM generates structured triage information.
6. The result is validated and stored in PostgreSQL.

The current implementation uses **Sentence Transformers (`all-MiniLM-L6-v2`)** with a lightweight cosine-similarity fallback.

## 📋 Extracted Information

For each conversation, the system can identify:

| Field           | Description                    |
| --------------- | ------------------------------ |
| Caller Name     | Identified customer name       |
| Callback Number | Contact number                 |
| Intent          | Main reason for the call       |
| Sentiment       | Customer sentiment             |
| Urgency         | Low / Medium / High            |
| Risk Score      | 1–10 risk assessment           |
| Department      | Responsible department         |
| Action Required | Recommended operational action |
| RAG Policy      | Relevant policy context        |

## 🛠️ Tech Stack

| Technology                | Purpose                  |
| ------------------------- | ------------------------ |
| **Python**                | Core application         |
| **Streamlit**             | Web interface            |
| **Groq API**              | LLM inference            |
| **Groq Whisper**          | Speech-to-text           |
| **Anthropic API**         | Alternative LLM provider |
| **RAG**                   | Policy-aware retrieval   |
| **Sentence Transformers** | Semantic embeddings      |
| **NumPy**                 | Similarity calculations  |
| **PostgreSQL**            | Call-log database        |
| **psycopg2**              | PostgreSQL integration   |
| **Pandas**                | Data processing          |
| **Requests**              | API communication        |

## 📁 Project Structure

```text
ai-reception-agent/
│
├── app.py              # Streamlit application
├── ai_agent.py         # LLM, transcription & pipeline logic
├── rag_utils.py        # RAG retrieval system
├── database.py         # PostgreSQL operations
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
```

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/adityatripathi70688-sys/ai-reception-agent.git
cd ai-reception-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
DB_HOST=localhost
DB_NAME=triage_db
DB_USER=postgres
DB_PASS=YOUR_POSTGRES_PASSWORD
DB_PORT=5432

GROQ_API_KEY=YOUR_GROQ_API_KEY
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
```

### 5. Run the application

```bash
streamlit run app.py
```

## 🔐 Environment Variables

The application requires:

* `GROQ_API_KEY`
* `ANTHROPIC_API_KEY` *(optional if using only Groq)*
* `DB_HOST`
* `DB_NAME`
* `DB_USER`
* `DB_PASS`
* `DB_PORT`

**Never commit your `.env` file or API keys to GitHub.**

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud** with **Neon PostgreSQL** as the cloud database.

Production architecture:

```text
Streamlit Cloud
      │
      ├── Groq API
      │     ├── LLM
      │     └── Whisper
      │
      ├── RAG Pipeline
      │
      └── Neon PostgreSQL
            └── Call Logs
```

## 🔮 Future Improvements

* Real-time voice conversation
* Vector database integration using pgvector/Pinecone/Weaviate
* Authentication and role-based dashboards
* Automated email/SMS follow-ups
* Advanced analytics and reporting
* CRM integration
* Multi-language conversation support

## 👨‍💻 Author

**Aditya Tripathi**
