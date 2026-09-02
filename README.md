# Sahakari Sahayak — Multilingual Cooperative Governance & Legal Assistance Chatbot

An AI-powered, multilingual chatbot that helps cooperative members, farmers, and
rural stakeholders get guidance on cooperative laws and by-laws, Ministry of
Cooperation schemes, PMFBY and agricultural support, financial literacy, and
grievance redressal — built with **Streamlit**, **Google Gemini**, **LangChain**,
**ChromaDB**, and **PyPDF**.

## ⚠️ About the API key

Never hardcode your Gemini API key into any file in this project. The app reads
it from Streamlit's secrets manager (or a session-only input box for quick local
testing) — see **Setup** below. If a key has ever been pasted into a chat, doc,
or committed to a public repo, treat it as compromised and regenerate it in
[Google AI Studio](https://aistudio.google.com/app/apikey).

## Features

- 💬 Chat interface answering cooperative-governance, PACS, scheme, PMFBY, and
  financial-literacy questions
- 📚 Retrieval-augmented generation (RAG) over your own PDF/TXT/MD documents —
  upload the actual Cooperative Societies Act text, state by-laws, official
  scheme circulars, etc., and answers ground themselves in those documents
  instead of guessing
- 🌐 Multilingual — auto-detect or pick a response language (English, Hindi,
  Gujarati, Marathi, Tamil, Telugu, Kannada, Bengali, Punjabi, Malayalam, Odia)
- 🎤 Voice input (record a question, transcribed via Gemini's native audio
  understanding) and 🔊 optional spoken replies (via gTTS)
- 📝 A guided "File a Grievance" form that drafts a structured grievance letter
- ☁️ Ready to deploy on Streamlit Community Cloud

## Project structure

```
coop-chatbot/
├── app.py                        # Streamlit app (entry point)
├── requirements.txt
├── .streamlit/
│   ├── config.toml               # theme
│   └── secrets.toml.example      # copy -> secrets.toml locally; never commit the real one
├── src/
│   ├── config.py                 # models, languages, prompts, paths
│   ├── rag_engine.py             # ingestion, chunking, vector store, chains
│   └── voice.py                  # speech-to-text / text-to-speech helpers
├── data/
│   ├── knowledge_base/           # put your source PDFs/TXT/MD here
│   └── chroma_db/                # auto-generated vector index (not committed)
└── README.md
```

## Setup (local)

1. **Python 3.10+** recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   (format looks like `AIza...`).
4. Copy the secrets template and add your key:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # then edit .streamlit/secrets.toml and paste your real key
   ```
5. Run it:
   ```bash
   streamlit run app.py
   ```

If you skip step 4, the app will show a password-style input box in the
sidebar where you can paste a key for that browser session only (nothing is
written to disk) — handy for a quick test.

## Adding your real knowledge base

The app ships with a small placeholder file
(`data/knowledge_base/starter_faq.md`) just so retrieval works immediately.
For a real deployment, replace/add to it with authoritative sources, e.g.:

- Your state's Cooperative Societies Act and model by-laws
- Official Ministry of Cooperation scheme guidelines and circulars
- PMFBY operational guidelines for the current season
- PACS handbooks / member service guides

You can either drop files directly into `data/knowledge_base/` before
deploying, or use the **file uploader in the app's sidebar** at runtime, then
click **"🔄 Rebuild knowledge base index"** to re-embed everything into
ChromaDB. Uploads at runtime are stored on the app's local disk — on
Streamlit Community Cloud that storage is **ephemeral** (wiped on redeploy/
restart), so for anything you want to persist long-term, commit it into
`data/knowledge_base/` in the repo instead.

## Deploying to Streamlit Community Cloud

1. Push this folder to a **public or private GitHub repo**. Double-check
   `.streamlit/secrets.toml` is *not* in the repo (it's git-ignored by
   default here) — only `secrets.toml.example` should be committed.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **"New app"**,
   and point it at your repo, branch, and `app.py`.
3. Before (or right after) deploying, open **Settings → Secrets** in the
   Streamlit Cloud dashboard and paste:
   ```toml
   GOOGLE_API_KEY = "your-real-key-here"
   ```
   This is the only place your real key should ever live.
4. Deploy. First load will read the built-in starter knowledge base; add your
   real documents and click "Rebuild knowledge base index" as described above.

### Why `pysqlite3-binary` is in requirements.txt

Streamlit Community Cloud's base image often ships an older system `sqlite3`
than ChromaDB requires (`>= 3.35.0`), causing a `RuntimeError` on startup.
`requirements.txt` includes `pysqlite3-binary`, and the top of `app.py` swaps
it in for the standard library's `sqlite3` before ChromaDB is imported. This
is the standard community fix for this exact deployment error.

## Notes on accuracy

The system prompt instructs Gemini to answer from your ingested documents
first, and to clearly flag anything it's answering from general knowledge —
and to never invent specific numbers, helpline numbers, or portal URLs it
isn't grounded on. Even so, this is a guidance tool, not an official system
of record: for anything with real legal, financial, or scheme-eligibility
stakes, point users to their PACS, their state's Registrar of Cooperative
Societies, or the official Ministry of Cooperation / PMFBY channels to
confirm current details.
