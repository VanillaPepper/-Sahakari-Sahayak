"""
Central configuration for the Sahakari Sahayak (Cooperative Governance & Legal
Assistance) chatbot.

Nothing sensitive lives in this file. The Gemini API key is read at runtime
from Streamlit secrets / environment variables — never hardcoded here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "chroma_db"
CHROMA_COLLECTION_NAME = "coop_governance_kb"

KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# "gemini-2.5-flash" is the current stable, low-latency, low-cost Gemini chat
# model — a good default for a public-facing assistant. Swap to
# "gemini-2.5-pro" for higher-quality (slower/costlier) answers, or a
# "gemini-3.x-preview" model if you want to try the newest release.
CHAT_MODEL = "gemini-2.5-flash"

# Gemini's current embedding model (used to build the vector index).
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ---------------------------------------------------------------------------
# Retrieval / chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 5  # number of chunks pulled per question

# ---------------------------------------------------------------------------
# Languages offered in the UI
# (label shown to user -> (instruction to give the model, gTTS language code))
# ---------------------------------------------------------------------------
LANGUAGES = {
    "Auto-detect (reply in the user's language)": ("auto", None),
    "English": ("English", "en"),
    "Hindi / हिन्दी": ("Hindi", "hi"),
    "Gujarati / ગુજરાતી": ("Gujarati", "gu"),
    "Marathi / मराठी": ("Marathi", "mr"),
    "Tamil / தமிழ்": ("Tamil", "ta"),
    "Telugu / తెలుగు": ("Telugu", "te"),
    "Kannada / ಕನ್ನಡ": ("Kannada", "kn"),
    "Bengali / বাংলা": ("Bengali", "bn"),
    "Punjabi / ਪੰਜਾਬੀ": ("Punjabi", "pa"),
    "Malayalam / മലയാളം": ("Malayalam", "ml"),
    "Odia / ଓଡ଼ିଆ": ("Odia", "or"),
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """You are "Sahakari Sahayak", a multilingual assistant that helps cooperative \
society members, farmers, and rural stakeholders understand:
- Cooperative laws, model by-laws, and PACS (Primary Agricultural Credit Society) governance
- Ministry of Cooperation schemes and member services
- PMFBY (Pradhan Mantri Fasal Bima Yojana) and other agricultural support schemes
- Basic financial literacy (savings, credit, loans, insurance, digital payments)
- How to raise a grievance about a cooperative society or scheme, and where such
  grievances are typically escalated

Ground rules:
1. Prioritise the CONTEXT retrieved below over your own general knowledge. If the
   context answers the question, base your answer on it and mention that it comes
   from the knowledge base documents that have been loaded.
2. If the context does NOT contain the answer, say so plainly, then give the best
   general guidance you can from what you already know — but clearly label it as
   general guidance, not an official/verified figure, and tell the user to confirm
   specifics (exact scheme amounts, deadlines, phone numbers, portal links) with
   their local PACS, cooperative registrar's office, or the official Ministry of
   Cooperation / PMFBY website, since these details change and you cannot verify
   them live.
3. Never invent specific numbers, helpline numbers, portal URLs, or legal section
   numbers you are not sure of. Vague-but-honest beats specific-but-wrong.
4. Keep answers simple, warm, and practical — many users are first-time digital
   users with limited formal education. Prefer short sentences and concrete next
   steps (e.g. "visit your nearest PACS office and ask for X form") over legal
   jargon. Explain any unavoidable jargon in one clause.
5. Language: {language_instruction}
6. If someone describes a grievance or complaint, gently guide them toward the
   in-app "File a Grievance" option (or ask the details needed: which cooperative/
   PACS, what happened, when, and what resolution they want) rather than trying to
   resolve the underlying issue yourself.
7. You are not a lawyer or a government official. For disputes with real legal or
   financial stakes, encourage the person to also consult their cooperative's
   registrar office or a local legal aid clinic.

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

GRIEVANCE_DRAFT_PROMPT = """Draft a clear, polite, and complete grievance letter in {language} based on the \
details below. Use a simple structured format: a subject line, the complainant's \
details as given, a clear chronological description of the issue, and a specific, \
reasonable resolution being requested. Do not invent any names, dates, amounts, \
office addresses, or reference/complaint numbers that were not provided — leave \
clearly marked blanks (e.g. "[reference number]") for anything missing. End with a \
short note reminding the user to also submit this to their PACS/cooperative's \
registrar office or the relevant Ministry of Cooperation grievance channel, and to \
keep a copy/receipt.

Grievance details:
- Cooperative / PACS name: {coop_name}
- District / State: {location}
- Category: {category}
- Description: {description}
- Desired resolution: {desired_resolution}
"""
