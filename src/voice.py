"""
Voice helpers for rural / low-literacy users.

Speech-to-text: Gemini models natively understand audio, so we send the
recorded clip straight to Gemini and ask for a plain transcript — no separate
STT service or API key needed beyond the one Gemini key the app already uses.

Text-to-speech: gTTS (Google Translate's free TTS endpoint) turns the answer
back into speech. It needs outbound internet access, which Streamlit Cloud
has, so no extra credentials are required.
"""

from __future__ import annotations

import io

from google import genai
from google.genai import types
from gtts import gTTS

from . import config


def transcribe_audio(api_key: str, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Send a short audio clip to Gemini and get back a plain-text transcript."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=config.CHAT_MODEL,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            "Transcribe exactly what is said in this audio clip. Reply with "
            "only the transcript text, in the original language(s) spoken — "
            "no translation, no commentary, no extra formatting.",
        ],
    )
    return (response.text or "").strip()


def text_to_speech(text: str, lang_code: str | None) -> bytes | None:
    """Convert text to an in-memory MP3. Returns None if lang_code is unset
    or the given text is empty (e.g. auto-detect with an unmappable language)."""
    if not text or not lang_code:
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang_code).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:  # noqa: BLE001
        print(f"[voice] TTS failed: {exc}")
        return None
