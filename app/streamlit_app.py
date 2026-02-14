from __future__ import annotations
import html

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Any

import streamlit as st
from dotenv import load_dotenv

from tools.retriever import build_or_update_index
from agents.graph import run_task
from schemas.state import AppState

load_dotenv()

APP_TITLE = "Agentic Assistant"
SAMPLE_DOCS_DIR = Path("data/sample_docs")
CHROMA_DIR = Path("data/chroma")
FINGERPRINT_FILE = CHROMA_DIR / ".fingerprint"


# -----------------------------
# Styling
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
.block-container {max-width: 980px; padding-top: 1.25rem;}

.hero {
  padding: 18px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.10);
}
.hero h1 {margin: 0; font-size: 26px; line-height: 1.1;}
.hero p {margin: 6px 0 0 0; opacity: 0.85; font-size: 13px;}

.card {
  padding: 14px 14px;
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
}

.small {font-size: 12px; opacity: 0.8;}

/* Citation blocks (stable, no expanders) */
.cite {
  padding: 12px 12px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  margin: 10px 0;
}
.cite .meta {
  font-size: 12px;
  opacity: 0.85;
  margin-bottom: 6px;
}
.cite .snippet {
  font-size: 13px;
  line-height: 1.35;
  white-space: pre-wrap;
  max-height: 140px;      /* prevents page explosions */
  overflow-y: auto;       /* scroll inside citation */
  padding-right: 6px;
}

/* Make tabs less jumpy */
div[data-testid="stTabs"] { background: transparent; }
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Indexing helpers
# -----------------------------
def ensure_dirs() -> None:
    SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def docs_fingerprint(doc_dir: Path) -> str:
    files = sorted([p for p in doc_dir.rglob("*") if p.is_file()])
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def read_fingerprint() -> Optional[str]:
    if FINGERPRINT_FILE.exists():
        v = FINGERPRINT_FILE.read_text(encoding="utf-8").strip()
        return v or None
    return None


def write_fingerprint(fp: str) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_FILE.write_text(fp, encoding="utf-8")


def save_uploaded_files(uploaded_files) -> int:
    ensure_dirs()
    n = 0
    for f in uploaded_files or []:
        out_path = SAMPLE_DOCS_DIR / f.name
        with out_path.open("wb") as out:
            out.write(f.getbuffer())
        n += 1
    return n


def clear_docs_and_index() -> None:
    if SAMPLE_DOCS_DIR.exists():
        shutil.rmtree(SAMPLE_DOCS_DIR)
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    ensure_dirs()


def ensure_index_ready() -> str:
    ensure_dirs()
    doc_files = [p for p in SAMPLE_DOCS_DIR.rglob("*") if p.is_file()]
    if not doc_files:
        return "No docs uploaded yet."

    current = docs_fingerprint(SAMPLE_DOCS_DIR)
    previous = read_fingerprint()
    if previous == current:
        return "Index is up to date."

    _, num = build_or_update_index(str(SAMPLE_DOCS_DIR), str(CHROMA_DIR))
    write_fingerprint(current)
    return f"Indexed {num} chunks (docs changed)."


# -----------------------------
# LangGraph return normalization
# -----------------------------
def as_app_state(x: Any) -> AppState:
    if isinstance(x, AppState):
        return x
    if isinstance(x, dict):
        return AppState(**x)
    raise TypeError(f"Unexpected state type: {type(x)}")


# -----------------------------
# Session
# -----------------------------
def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}
    if "last_state" not in st.session_state:
        st.session_state.last_state = None
    if "kb_status" not in st.session_state:
        st.session_state.kb_status = ""


# -----------------------------
# UI helpers
# -----------------------------
def render_latest_details_under_answer(state: AppState) -> None:
    """
    Render plan/citations/trace in a stable layout (no expanders).
    This goes under the latest assistant message.
    """
    st.markdown("")

    tabs = st.tabs(["Citations", "Plan", "Trace"])

    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if state.citations:
            for i, c in enumerate(state.citations, start=1):
                st.markdown(
                    f"""
<div class="cite">
  <div class="meta"><b>[{i}]</b> <code>{c.doc_id}</code> — {c.location}</div>
  <div class="snippet">{html.escape(c.snippet)}</div>

</div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div class='small'>No citations (likely: Not found in sources).</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if state.plan:
            for i, step in enumerate(state.plan, start=1):
                st.write(f"{i}. {step}")
        else:
            st.write("_No plan_")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if state.agent_logs:
            st.dataframe([e.model_dump() for e in state.agent_logs], use_container_width=True, height=260)
        else:
            st.write("_No logs_")
        st.write(f"Verifier retries: **{state.verifier_fail_count} / {state.verifier_max_retries}**")
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    inject_css()
    init_session()
    ensure_dirs()

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

        st.markdown("---")
        st.markdown("### Knowledge base")
        uploaded = st.file_uploader(
            "Upload docs (.txt, .pdf)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Indexed automatically when you submit a task.",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save", use_container_width=True):
                n = save_uploaded_files(uploaded)
                st.session_state.kb_status = f"Saved {n} file(s)."
        with c2:
            if st.button("Reset", use_container_width=True):
                clear_docs_and_index()
                st.session_state.kb_status = "Cleared docs + index."

        doc_count = len([p for p in SAMPLE_DOCS_DIR.rglob("*") if p.is_file()])
        st.markdown(f"<div class='small'>Docs: <b>{doc_count}</b></div>", unsafe_allow_html=True)

        if st.session_state.kb_status:
            st.info(st.session_state.kb_status)

    # Header
    st.markdown(
        """
<div class="hero">
  <h1>Agentic Assistant</h1>
  <p>Grounded workflow with citations + verifier. Upload docs, then ask a task.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Chat stream
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Show details under the latest assistant answer (ONLY once, and stable)
    if st.session_state.last_state and st.session_state.messages:
        if st.session_state.messages[-1]["role"] == "assistant":
            with st.container():
                render_latest_details_under_answer(st.session_state.last_state)

    # Bottom input
    user_task = st.chat_input("Type a task…")

    if user_task:
        st.session_state.messages.append({"role": "user", "content": user_task})

        with st.spinner("Preparing knowledge base…"):
            st.session_state.kb_status = ensure_index_ready()

        with st.spinner("Running agents…"):
            result = run_task(user_task=user_task, persist_dir=str(CHROMA_DIR), model=model)
            state = as_app_state(result)

        st.session_state.last_state = state

        assistant_msg = state.final_output or state.draft_output or "_No output_"
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})

        st.rerun()


if __name__ == "__main__":
    main()


