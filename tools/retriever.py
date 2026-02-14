from __future__ import annotations

import os
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)

try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


def _load_documents(sample_docs_dir: str) -> List[Document]:
    """
    Loads documents from data/sample_docs/.
    Supports .txt by default; supports PDF if pypdf and PyPDFLoader are installed.
    """
    docs: List[Document] = []

    # Text files
    txt_loader = DirectoryLoader(
        sample_docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=True,
    )
    docs.extend(txt_loader.load())

    # PDFs (optional)
    if PDF_AVAILABLE:
        pdf_paths: List[str] = []
        for root, _, files in os.walk(sample_docs_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, f))
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

    return docs


def _split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = splitter.split_documents(docs)

    # Attach a stable doc_id & chunk_id for citations
    for i, d in enumerate(split_docs):
        src = d.metadata.get("source", "unknown_source")
        page = d.metadata.get("page", None)
        d.metadata["doc_id"] = os.path.basename(src)
        d.metadata["chunk_id"] = i
        if page is not None:
            d.metadata["location"] = f"page {page}, chunk {i}"
        else:
            d.metadata["location"] = f"chunk {i}"
    return split_docs


def get_vectorstore(
    persist_dir: str,
    collection_name: str = "agentic_assistant",
) -> Chroma:
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def build_or_update_index(
    sample_docs_dir: str,
    persist_dir: str,
    collection_name: str = "agentic_assistant",
) -> Tuple[Chroma, int]:
    """
    Loads docs from sample_docs_dir, splits, and upserts into Chroma persistent store.
    Returns (vectorstore, num_chunks_indexed).
    """
    os.makedirs(sample_docs_dir, exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)

    raw_docs = _load_documents(sample_docs_dir)
    if not raw_docs:
        vs = get_vectorstore(persist_dir, collection_name)
        return vs, 0

    split_docs = _split_documents(raw_docs)
    vs = get_vectorstore(persist_dir, collection_name)

    # Add documents (Chroma will store embeddings)
    vs.add_documents(split_docs)
    return vs, len(split_docs)


def retrieve(
    query: str,
    persist_dir: str,
    k: int = 6,
    collection_name: str = "agentic_assistant",
) -> List[Document]:
    """
    Retrieve top-k docs from persistent Chroma store.
    """
    vs = get_vectorstore(persist_dir, collection_name)
    retriever = vs.as_retriever(search_kwargs={"k": k})

# Newer LangChain retrievers are Runnables
    docs = retriever.invoke(query)

# Back-compat: ensure list[Document]
    return list(docs)

