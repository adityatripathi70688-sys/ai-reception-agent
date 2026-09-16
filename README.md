\# AI Reception Agent
An AI-powered customer call triage system that combines \*\*LLMs, Retrieval-Augmented Generation (RAG), speech-to-text, and PostgreSQL\*\* to analyze customer calls, retrieve relevant internal policies, and route issues to the appropriate department.
\## Features
\* Upload customer transcripts or audio files.
\* Convert audio to text using \*\*Groq Whisper\*\*.
\* Use \*\*Groq or Anthropic\*\* for AI-based call analysis.
\* \*\*Retrieval-Augmented Generation (RAG)\*\* for retrieving relevant internal policies.
\* Extract structured information from calls:
&#x20; \* Caller name
&#x20; \* Callback number
&#x20; \* Intent
&#x20; \* Sentiment
&#x20; \* Urgency
&#x20; \* Risk score
&#x20; \* Assigned department
&#x20; \* Required action
\* Store call logs in \*\*PostgreSQL\*\*.
\* View statistics and historical call logs through a Streamlit dashboard.
\* Search and filter previous calls.
\* Includes fallback handling for failed API calls and invalid LLM responses.
\## Tech Stack
| Technology            | Usage                          |
| --------------------- | ------------------------------ |
| Python                | Core application               |
| Streamlit             | Web interface and dashboard    |
| Groq API              | LLM inference                  |
| Anthropic API         | Alternative LLM provider       |
| Groq Whisper          | Speech-to-text                 |
| \*\*RAG\*\*               | Internal policy retrieval      |
| Sentence Transformers | Text embeddings                |
| NumPy                 | Vector similarity calculations |
| PostgreSQL            | Call-log storage               |
| psycopg2              | PostgreSQL connectivity        |
| Pandas                | Data processing                |
| Requests              | API communication              |
\## How It Works
```text
Customer Transcript / Audio
&#x20;         ↓
&#x20;   Groq Whisper
&#x20;   (Audio input)
&#x20;         ↓
&#x20;    RAG Retrieval
&#x20;         ↓
&#x20;Relevant Internal Policy
&#x20;         ↓
&#x20;   LLM Triage Agent
&#x20;  Groq / Anthropic
&#x20;         ↓
&#x20;  JSON Validation
&#x20;         ↓
&#x20;Structured Triage Data
&#x20;         ↓
&#x20;     PostgreSQL
&#x20;         ↓
&#x20;  Streamlit Dashboard
```
\### RAG Pipeline
The application uses \*\*Retrieval-Augmented Generation (RAG)\*\* to provide the LLM with relevant internal policy information before analyzing a customer call.
The current implementation:
1\. Splits the internal policy document into chunks.
2\. Generates embeddings using \*\*`all-MiniLM-L6-v2`\*\*.
3\. Calculates similarity between the customer issue and policy chunks.
4\. Retrieves the most relevant policy context.
5\. Provides that context to the LLM during call analysis.
If Sentence Transformers cannot be loaded, the application falls back to a TF-based cosine similarity approach.
\## Project Structure
```text
AI\_Agent\_Project/
├── ai\_agent.py       # AI pipeline, LLM calls and transcription
├── app.py            # Streamlit application
├── database.py       # PostgreSQL operations
├── rag\_utils.py      # RAG and policy retrieval
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── .gitignore
└── README.md
```
\## Setup
\### 1. Create a virtual environment
```bash
python -m venv venv
```
Windows PowerShell:
```powershell
.\\venv\\Scripts\\Activate.ps1
```
\### 2. Install dependencies
```bash
pip install -r requirements.txt
```
\### 3. Configure environment variables
Create a local `.env` file:
```env
GROQ\_API\_KEY=your\_groq\_api\_key
ANTHROPIC\_API\_KEY=your\_anthropic\_api\_key
DB\_HOST=localhost
DB\_NAME=triage\_db
DB\_USER=postgres
DB\_PASS=your\_postgresql\_password
DB\_PORT=5432
```
Never commit API keys or database credentials to GitHub.
\### 4. Start the application
```bash
streamlit run app.py
```
\## Future Improvements
\* Move the policy knowledge base to a dedicated vector database.
\* Improve RAG retrieval and ranking.
\* Add authentication and role-based access.
\* Add automated tests and production monitoring.
\* Deploy the application to a cloud environment.
\## Author
\*\*Aditya Tripathi\*\*
