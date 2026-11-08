"""Streamlit frontend that uses the FastAPI backend."""
from __future__ import annotations

import os
import hmac
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
# First-time model initialization and CPU embedding can take several minutes.
# Use a short connection timeout but allow long server-side indexing to finish.
API_TIMEOUT = (30, int(os.getenv("DOCUMIND_API_READ_TIMEOUT", "900")))

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")

# Optional access gate for public portfolio deployments. Set APP_ACCESS_PASSWORD
# in the deployment environment; do not hard-code a password in this file.
access_password = os.getenv("APP_ACCESS_PASSWORD", "").strip()
if access_password and not st.session_state.get("authenticated", False):
    st.title("📚 DocuMind AI")
    st.caption("Private portfolio demonstration")
    entered = st.text_input("Demo access password", type="password")
    if st.button("Unlock", type="primary"):
        if hmac.compare_digest(entered, access_password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.title("📚 DocuMind AI")
st.caption("Upload  PDFs, ask questions across documents, and inspect page-level evidence.")


def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


try:
    health = api_get("/health")
    st.sidebar.success(f"Backend connected: {health.get('status', 'unknown')}")
except requests.RequestException as exc:
    st.error(f"Cannot connect to the DocuMind backend at {API_URL}. Start the API server first. Details: {exc}")
    st.stop()

st.sidebar.header("Upload documents")
files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
if st.sidebar.button("Upload and index PDFs", type="primary", disabled=not files):
    progress = st.sidebar.progress(0)
    for i, uploaded in enumerate(files, start=1):
        try:
            response = requests.post(
                f"{API_URL}/api/documents",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=API_TIMEOUT,
            )
            response.raise_for_status()
            item = response.json()
            st.sidebar.success(f"{item['filename']}: {item['message']} ({item['chunks_count']} chunks)")
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", str(exc))
            st.sidebar.error(f"Failed to process {uploaded.name}: {detail}")
        progress.progress(i / len(files))
    st.rerun()

try:
    documents = api_get("/api/documents")
except requests.RequestException as exc:
    st.error(f"Could not load indexed documents: {exc}")
    st.stop()

if not documents:
    st.info("Upload one or more PDFs using the sidebar to begin.")
    st.stop()

st.subheader("Indexed documents")
st.dataframe(
    [
        {
            "Document": doc["filename"],
            "Pages": doc["total_pages"],
            "OCR pages": doc["ocr_pages"],
            "Chunks": doc["chunks_count"],
            "Indexed": doc["created_at"],
        }
        for doc in documents
    ],
    use_container_width=True,
    hide_index=True,
)

labels = {doc["id"]: f"{doc['filename']} ({doc['chunks_count']} chunks)" for doc in documents}
selected_ids = st.multiselect(
    "Select documents to search",
    options=[doc["id"] for doc in documents],
    default=[doc["id"] for doc in documents],
    format_func=lambda doc_id: labels.get(doc_id, doc_id),
)

with st.form("ask_form"):
    question = st.text_area(
        "Your question",
        placeholder="Example: Compare the scholarship continuation requirements across all uploaded university policies.",
        height=100,
    )
    passages = st.slider("Evidence passages per document", 3, 10, 6)
    submitted = st.form_submit_button("Ask selected documents", type="primary", disabled=not selected_ids)

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                response = requests.post(
                    f"{API_URL}/api/ask",
                    json={
                        "question": question.strip(),
                        "document_ids": selected_ids,
                        "passages_per_document": passages,
                    },
                    timeout=API_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()
            st.subheader("Answer")
            st.markdown(result.get("answer", "No answer returned."))
            if result.get("citations_ok"):
                st.success("Citation validation passed.")
            elif result.get("found"):
                st.warning("Please review the citations; validation did not pass.")
            st.caption("Documents searched: " + ", ".join(result.get("documents_searched", [])))
            st.subheader("Retrieved evidence")
            for rank, item in enumerate(result.get("evidence", []), start=1):
                title = f"{rank}. {item.get('doc', 'Unknown document')} — physical PDF page {item.get('page', 'Unknown')}"
                with st.expander(title, expanded=(rank == 1)):
                    st.write(item.get("text", ""))
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", str(exc))
            st.error(f"Question request failed: {detail}")
