"""
RAG (retrieval-augmented generation) engine for the cooperative governance
chatbot.

Responsibilities:
  - Load PDFs / text / markdown files from the knowledge base folder
  - Split them into chunks and embed them with Gemini embeddings into ChromaDB
  - Build a LangChain retrieval chain (retriever + Gemini chat model)
  - Provide a simple fallback (no-retrieval) chain for when the knowledge base
    is empty, so the app is still useful on first run
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _load_single_file(path: Path) -> list[Document]:
    """Load one file into LangChain Documents, tagging the source filename.

    PDFs are read directly with pypdf (one Document per page, so citations
    can point to a page number); .txt/.md files are read as a single
    Document. This avoids depending on langchain-community, which has been
    sunset/archived upstream.
    """
    docs: list[Document] = []
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_num},
                    )
                )
    else:
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def load_knowledge_base_documents(kb_dir: Path = config.KNOWLEDGE_BASE_DIR) -> list[Document]:
    """Load every supported file under the knowledge base directory."""
    documents: list[Document] = []
    for path in sorted(kb_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                documents.extend(_load_single_file(path))
            except Exception as exc:  # noqa: BLE001 - surface but keep going
                print(f"[ingest] Skipped {path.name}: {exc}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=api_key)


def get_chat_model(api_key: str, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=False,
    )


def rebuild_vectorstore(api_key: str):
    """
    Wipe and rebuild the persisted Chroma collection from whatever is
    currently in the knowledge base folder. Returns (vectorstore, num_chunks).
    Returns (None, 0) if the knowledge base is empty.
    """
    from langchain_chroma import Chroma

    documents = load_knowledge_base_documents()
    if not documents:
        return None, 0

    chunks = split_documents(documents)

    # Fresh start each rebuild so deleted/edited source files don't leave
    # stale chunks behind.
    if config.CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(api_key)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )
    return vectorstore, len(chunks)


def load_existing_vectorstore(api_key: str) -> Optional["Chroma"]:  # noqa: F821
    """Load a previously-persisted Chroma collection, if one exists."""
    from langchain_chroma import Chroma

    if not any(config.CHROMA_PERSIST_DIR.glob("**/*")):
        return None
    embeddings = get_embeddings(api_key)
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )
    # A persisted-but-empty collection should behave like "no vectorstore".
    try:
        if vectorstore._collection.count() == 0:  # noqa: SLF001
            return None
    except Exception:  # noqa: BLE001
        pass
    return vectorstore


def _language_instruction(language_choice: str) -> str:
    if language_choice == "auto":
        return (
            "Reply in the same language the user just wrote in. If they mix "
            "languages, mirror that mix naturally."
        )
    return f"Reply in {language_choice}, regardless of what language the user wrote in."


def build_answer_chain(api_key: str, vectorstore, language_choice: str, k: int = config.RETRIEVER_K):
    """
    Build an LCEL chain: retriever -> stuff context into system prompt -> Gemini.
    If vectorstore is None, builds a context-free (general knowledge) chain
    instead, so the app still works before any documents are ingested.
    """
    llm = get_chat_model(api_key)
    system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(
        language_instruction=_language_instruction(language_choice),
        context="{context}",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    if vectorstore is None:
        # No knowledge base yet: answer from general knowledge, with an
        # explicit "no documents loaded" notice baked into the context slot.
        no_context_prompt = prompt.partial(
            context="(No knowledge base documents are loaded yet — answer from "
            "general knowledge and say so.)"
        )
        chain = no_context_prompt | llm
        return chain, None

    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain, retriever


def run_chain(chain, has_retriever: bool, user_input: str, chat_history: list) -> tuple[str, list[Document]]:
    """
    Invoke either kind of chain built by build_answer_chain and normalise the
    return shape to (answer_text, source_documents).
    """
    if has_retriever:
        result = chain.invoke({"input": user_input, "chat_history": chat_history})
        return result["answer"], result.get("context", [])
    else:
        result = chain.invoke({"input": user_input, "chat_history": chat_history})
        return result.content, []


def draft_grievance(api_key: str, language: str, coop_name: str, location: str,
                     category: str, description: str, desired_resolution: str) -> str:
    llm = get_chat_model(api_key, temperature=0.2)
    prompt = config.GRIEVANCE_DRAFT_PROMPT.format(
        language=language,
        coop_name=coop_name or "[not provided]",
        location=location or "[not provided]",
        category=category or "[not provided]",
        description=description or "[not provided]",
        desired_resolution=desired_resolution or "[not provided]",
    )
    response = llm.invoke(prompt)
    return response.content
