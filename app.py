"""
Sahakari Sahayak — Multilingual Cooperative Governance & Legal Assistance Chatbot
Streamlit + Google Gemini + LangChain + ChromaDB + PyPDF

Run locally:
    streamlit run app.py

Deploy: push this folder to a GitHub repo, deploy on Streamlit Community
Cloud, and add GOOGLE_API_KEY under the app's Settings -> Secrets.
See README.md for full steps.
"""

# --- ChromaDB on Streamlit Cloud needs a newer sqlite3 than the base image
# ships. This swap must happen before chromadb is imported anywhere
# (directly or via langchain_chroma). Safe no-op if pysqlite3 isn't
# installed (e.g. on Windows during local dev).
try:
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

from src import config, rag_engine, voice

st.set_page_config(
    page_title="Sahakari Sahayak | Cooperative Governance Assistant",
    page_icon="🌾",
    layout="centered",
)

# ---------------------------------------------------------------------------
# API key handling — never hardcoded. Prefers Streamlit secrets (for
# deployment), falls back to a session-only text box (for quick local testing).
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
    if key:
        return key
    return st.session_state.get("manual_api_key")


with st.sidebar:
    st.markdown("## 🌾 Sahakari Sahayak")
    st.caption("Cooperative Governance & Legal Assistance")

    api_key = get_api_key()
    if not api_key:
        st.warning("No Gemini API key configured.", icon="🔑")
        manual_key = st.text_input(
            "Enter a Gemini API key for this session",
            type="password",
            help="For deployment, set GOOGLE_API_KEY in Streamlit Cloud's "
            "Settings → Secrets instead of typing it here. Keys entered here "
            "are kept only in this browser session and are never saved to disk.",
        )
        if manual_key:
            st.session_state["manual_api_key"] = manual_key
            st.rerun()
        st.stop()

    st.divider()

    language_label = st.selectbox("Response language", list(config.LANGUAGES.keys()))
    language_choice, tts_lang = config.LANGUAGES[language_label]

    voice_reply = st.toggle("🔊 Read answers aloud", value=False)

    st.divider()
    st.markdown("**Knowledge base**")
    kb_files = [
        p.name
        for p in config.KNOWLEDGE_BASE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in rag_engine.SUPPORTED_EXTENSIONS
    ]
    st.caption(f"{len(kb_files)} source document(s) in `data/knowledge_base/`")
    with st.expander("Files"):
        for f in kb_files:
            st.write(f"- {f}")
        if not kb_files:
            st.write("_none yet_")

    uploaded = st.file_uploader(
        "Add PDF / TXT / MD documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            (config.KNOWLEDGE_BASE_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s). Click rebuild below to index them.")
        st.rerun()

    if st.button("🔄 Rebuild knowledge base index", use_container_width=True):
        with st.spinner("Reading documents and building the vector index…"):
            vectorstore, n_chunks = rag_engine.rebuild_vectorstore(api_key)
        if vectorstore is None:
            st.info("Knowledge base folder is empty — nothing to index.")
        else:
            st.success(f"Indexed {n_chunks} chunks from {len(kb_files)} document(s).")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.caption(
        "Not a substitute for official guidance. For binding decisions, "
        "confirm details with your PACS, cooperative registrar, or the "
        "official Ministry of Cooperation / PMFBY channels."
    )

# ---------------------------------------------------------------------------
# Load the vector store (if one has been built) and the retrieval chain for
# the current language. Opening a persisted Chroma collection is cheap (it
# doesn't re-embed anything), so this runs fresh on every rerun rather than
# being cached — that keeps it correct if the user swaps their API key or
# rebuilds the index mid-session, at a negligible performance cost.
# ---------------------------------------------------------------------------
vectorstore = rag_engine.load_existing_vectorstore(api_key)
chain, retriever = rag_engine.build_answer_chain(api_key, vectorstore, language_choice)

# ---------------------------------------------------------------------------
# Tabs: Chat / Voice / File a Grievance
# ---------------------------------------------------------------------------
tab_chat, tab_voice, tab_grievance = st.tabs(["💬 Chat", "🎤 Voice", "📝 File a Grievance"])

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of (role, content)
if "lc_history" not in st.session_state:
    st.session_state["lc_history"] = []  # LangChain-style (role, content) pairs


def render_history():
    for role, content in st.session_state["messages"]:
        with st.chat_message(role):
            st.markdown(content)


def handle_user_turn(user_text: str):
    st.session_state["messages"].append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, sources = rag_engine.run_chain(
                    chain, retriever is not None, user_text, st.session_state["lc_history"]
                )
                if isinstance(answer, list) and len(answer) > 0 and isinstance(answer[0], dict):
                    answer = answer[0].get('text', str(answer))
                elif isinstance(answer, dict):
                    answer = answer.get('text', str(answer))
            except Exception as exc:  # noqa: BLE001
                answer = (
                    "Sorry, I ran into an error talking to Gemini. Please check "
                    f"that the API key is valid and try again.\n\n`{exc}`"
                )
                sources = []
        st.markdown(answer)
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} passages used)"):
                for doc in sources:
                    st.caption(f"**{doc.metadata.get('source', 'unknown')}**")
                    st.text(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))

        if voice_reply and tts_lang:
            with st.spinner("Generating audio…"):
                audio_bytes = voice.text_to_speech(answer, tts_lang)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

    st.session_state["messages"].append(("assistant", answer))
    st.session_state["lc_history"].append(("human", user_text))
    st.session_state["lc_history"].append(("ai", answer))


with tab_chat:
    st.caption(
        "Ask about cooperative laws, PACS services, Ministry of Cooperation "
        "schemes, PMFBY, financial literacy, or how to raise a grievance."
    )
    render_history()
    if prompt := st.chat_input("Type your question…"):
        handle_user_turn(prompt)

with tab_voice:
    st.caption("Record a question in your own language — it will be transcribed and answered.")
    try:
        from audio_recorder_streamlit import audio_recorder

        audio_bytes = audio_recorder(text="Tap to record", icon_size="2x", pause_threshold=2.0)
        if audio_bytes:
            with st.spinner("Transcribing…"):
                try:
                    transcript = voice.transcribe_audio(api_key, audio_bytes)
                except Exception as exc:  # noqa: BLE001
                    transcript = ""
                    st.error(f"Could not transcribe audio: {exc}")
            if transcript:
                st.info(f"Heard: “{transcript}”")
                if st.button("Send this as my question", use_container_width=True):
                    handle_user_turn(transcript)
    except ImportError:
        st.warning(
            "Voice recording needs the `audio-recorder-streamlit` package. "
            "It's in requirements.txt, so this should only happen in an "
            "incomplete local setup — run `pip install -r requirements.txt`."
        )
    st.divider()
    st.caption("Recent conversation:")
    render_history()

with tab_grievance:
    st.caption(
        "Fill in what you know — leave anything blank if you're not sure, "
        "and the draft will mark it for you to fill in later."
    )
    with st.form("grievance_form"):
        coop_name = st.text_input("Cooperative society / PACS name")
        location = st.text_input("District / State")
        category = st.selectbox(
            "Category",
            [
                "Loan / credit issue",
                "Membership rights",
                "Election / governance dispute",
                "Crop insurance (PMFBY) claim",
                "Scheme benefit not received",
                "Staff conduct",
                "Other",
            ],
        )
        description = st.text_area("What happened? (include dates if you remember them)")
        desired_resolution = st.text_input("What resolution are you asking for?")
        submitted = st.form_submit_button("Draft grievance letter")

    if submitted:
        with st.spinner("Drafting…"):
            try:
                draft = rag_engine.draft_grievance(
                    api_key,
                    language_choice if language_choice != "auto" else "English",
                    coop_name,
                    location,
                    category,
                    description,
                    desired_resolution,
                )
                st.text_area("Draft grievance letter", draft, height=350)
                st.download_button(
                    "⬇️ Download as .txt",
                    draft,
                    file_name="grievance_draft.txt",
                    use_container_width=True,
                )
                st.caption(
                    "Review and fill in any [bracketed] blanks before submitting "
                    "this to your cooperative's registrar office or the relevant "
                    "grievance channel."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not draft the letter: {exc}")
