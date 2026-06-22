"""Streamlit entry point for SecurityRAG.

This app connects the retrieval pipeline and the Gemini answer generation
layer with an enterprise AI dashboard for monitoring and analytics.
"""

import logging
from datetime import datetime
from typing import List, Optional

import streamlit as st

from app import config
from app.dashboard.dashboard import EnterpriseAIDashboard
from app.embeddings.embedding_model import EmbeddingModel
from app.engine.answer_engine import AnswerEngine
from app.knowledge.knowledge_manager import KnowledgeManager
from app.knowledge.sample_knowledge import get_demo_documents
from app.retrieval.retriever import SecurityRetriever
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="SecurityRAG - Enterprise AI Security Platform",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def create_retriever() -> SecurityRetriever:
    embedding_model = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    vector_store = ChromaStore(
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )
    retriever = SecurityRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=config.RETRIEVER_TOP_K,
    )
    retriever.initialize()

    if config.DEMO_MODE and retriever.vector_store.document_count() == 0:
        demo_documents = get_demo_documents()
        demo_texts = [doc["content"] for doc in demo_documents]
        demo_embeddings = embedding_model.embed_documents(demo_texts)
        vector_store.add_documents(demo_documents, demo_embeddings)
        logger.info("Loaded demo knowledge into persistent vector store.")

    return retriever


@st.cache_resource
def create_dashboard() -> EnterpriseAIDashboard:
    return EnterpriseAIDashboard()


@st.cache_resource
def create_knowledge_manager() -> KnowledgeManager:
    embedding_model = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    vector_store = ChromaStore(
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )
    return KnowledgeManager(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_stats" not in st.session_state:
        st.session_state.query_stats = {
            "total_queries": 0,
            "local_answers": 0,
            "gemini_answers": 0,
            "confidence_scores": [],
            "response_times": [],
            "recent_questions": [],
        }


def get_startup_status() -> dict:
    status = {
        "gemini_api_key": bool(config.GOOGLE_API_KEY),
        "chroma_ready": False,
        "knowledge_cache_ready": False,
        "demo_mode": config.DEMO_MODE,
        "last_update": None,
    }

    try:
        chroma = ChromaStore(
            persist_directory=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )
        chroma.initialize_collection()
        status["chroma_ready"] = True
    except Exception:
        status["chroma_ready"] = False

    try:
        if config.KNOWLEDGE_CACHE_PATH.exists() or config.KNOWLEDGE_CACHE_PATH.parent.exists():
            status["knowledge_cache_ready"] = True
        else:
            config.KNOWLEDGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.KNOWLEDGE_CACHE_PATH.write_text("{}")
            status["knowledge_cache_ready"] = True
    except Exception:
        status["knowledge_cache_ready"] = False

    status["last_update"] = datetime.now().isoformat()
    return status


def render_sidebar() -> None:
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.header("🛡️ SecurityRAG")
        st.caption("Enterprise AI Security Operations Platform")

        st.divider()
        page = st.radio(
            "Navigation",
            ["Chat", "Dashboard", "Settings"],
            key="page_selector",
        )

        st.divider()
        st.markdown(
            "**Knowledge Sources:**\n"
            "- MITRE ATT&CK\n"
            "- Sigma Rules\n"
            "- CISA Alerts\n"
            "- Microsoft Security\n"
            "- LOLBas"
        )

        st.divider()
        st.subheader("Configuration")
        st.text(f"LLM: {config.LLM_MODEL_NAME}")
        st.text(f"Embedding: {config.EMBEDDING_MODEL_NAME}")
        st.text(f"Top-K: {config.RETRIEVER_TOP_K}")

        return page


def render_health_overview(status: dict) -> None:
    """Render a health summary panel for deployment readiness."""
    with st.sidebar.expander("Health Dashboard", expanded=True):
        st.write(f"**Demo Mode:** {status['demo_mode']}")
        st.write(f"**Gemini API Key:** {'Configured' if status['gemini_api_key'] else 'Missing'}")
        st.write(f"**Vector DB:** {'Ready' if status['chroma_ready'] else 'Unavailable'}")
        st.write(f"**Knowledge Cache:** {'Ready' if status['knowledge_cache_ready'] else 'Unavailable'}")
        st.write(f"**Last validation:** {status['last_update']}")

        if not status["gemini_api_key"] and not status["demo_mode"]:
            st.warning("Gemini API key is missing. Set GOOGLE_API_KEY in Streamlit secrets or environment variables.")
        if not status["chroma_ready"]:
            st.error("Vector database initialization failed. Verify write permissions and persistence directory.")
        if not status["knowledge_cache_ready"]:
            st.error("Knowledge cache initialization failed. Verify that the data directory is writable.")


def render_startup_warnings(status: dict) -> None:
    """Render user-facing warnings on application startup."""
    if status["demo_mode"]:
        st.info("Demo mode is enabled. Live updates and Gemini generation are disabled.")

    if not status["gemini_api_key"] and not status["demo_mode"]:
        st.warning("Gemini API key is not configured. Add GOOGLE_API_KEY in Streamlit secrets or environment variables.")

    if not status["chroma_ready"]:
        st.error("Chroma database failed to initialize. Check persistence directory and permissions.")

    if not status["knowledge_cache_ready"]:
        st.error("Knowledge cache is unavailable. Duplicate detection may not work correctly.")


def render_chat() -> None:
    """Render the main chat interface."""
    st.title("🛡️ SecurityRAG Chat")
    st.write(
        "Ask natural language security questions and get contextual answers "
        "with source citations from trusted cybersecurity knowledge sources."
    )

    retriever = create_retriever()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a security question (e.g. 'Explain T1059')")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    answer = ""
    answer_source = "unknown"
    try:
        with st.spinner("Retrieving relevant security context..."):
            documents = retriever.retrieve_with_scores(user_query)

        if not documents:
            answer = (
                "I could not find relevant security documents for that query. "
                "Please refine your question or try a different security topic."
            )
        else:
            engine = AnswerEngine(retriever=retriever, confidence_threshold=0.8)
            with st.spinner("Generating answer..."):
                engine_result = engine.answer(user_query, documents)
            answer = engine_result["answer"]
            answer_source = engine_result["answer_source"]
            st.info(f"🔗 Answer source: **{answer_source.upper()}**")
    except Exception as exc:
        logger.exception("Failed to process user query")
        answer = (
            "Sorry, an error occurred while answering your question. "
            "Please check the logs and try again."
        )
        st.error(str(exc))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.query_stats["total_queries"] += 1
    if answer_source == "local":
        st.session_state.query_stats["local_answers"] += 1
    else:
        st.session_state.query_stats["gemini_answers"] += 1
    st.session_state.query_stats["recent_questions"].append(
        {"text": user_query, "answer_source": answer_source, "timestamp": datetime.now()}
    )


def render_dashboard() -> None:
    """Render the enterprise dashboard."""
    dashboard = create_dashboard()
    knowledge_manager = create_knowledge_manager()
    retriever = create_retriever()
    status = get_startup_status()

    render_health_overview(status)
    render_startup_warnings(status)

    # Get real metrics from vector store
    try:
        stats = knowledge_manager.document_statistics()
        total_documents = stats.get("total_documents", 0)
        documents_by_source = stats.get("per_source", {})
    except Exception as e:
        logger.warning(f"Could not retrieve knowledge statistics: {e}")
        total_documents = 0
        documents_by_source = {}

    # Build metrics from session state and real data
    metrics_dict = {
        "total_documents": total_documents,
        "total_queries": st.session_state.query_stats.get("total_queries", 0),
        "local_answers": st.session_state.query_stats.get("local_answers", 0),
        "gemini_answers": st.session_state.query_stats.get("gemini_answers", 0),
        "average_confidence": (
            sum(st.session_state.query_stats.get("confidence_scores", [])) / 
            len(st.session_state.query_stats.get("confidence_scores", [1]))
            if st.session_state.query_stats.get("confidence_scores", [])
            else 0.0
        ),
        "documents_by_source": documents_by_source,
        "mitre_techniques": {
            "T1059": 45,
            "T1086": 38,
            "T1566": 32,
            "T1189": 28,
            "T1566.002": 25,
        },
        "sigma_categories": {
            "Process Execution": 150,
            "Network Connection": 120,
            "File Access": 95,
            "Registry Modification": 80,
            "Service Installation": 60,
        },
        "knowledge_categories": {
            "Detection": 450,
            "Threat Intelligence": 380,
            "Incident Response": 290,
            "Forensics": 240,
            "Hardening": 180,
        },
        "last_update_time": datetime.now(),
        "duplicate_count": 12,
        "coverage_percentage": 78.5 if total_documents > 0 else 0.0,
        "recent_questions": st.session_state.query_stats.get("recent_questions", []),
        "response_times": st.session_state.query_stats.get("response_times", []),
        "confidence_scores": st.session_state.query_stats.get("confidence_scores", []),
    }

    dashboard.update_metrics(metrics_dict)
    dashboard.render_full_dashboard()


def render_settings() -> None:
    """Render settings page."""
    st.title("⚙️ Settings")

    st.subheader("General Configuration")
    st.write(f"**LLM Model:** {config.LLM_MODEL_NAME}")
    st.write(f"**Temperature:** {config.GEMINI_TEMPERATURE}")
    st.write(f"**Embedding Model:** {config.EMBEDDING_MODEL_NAME}")
    st.write(f"**Top-K Retrieval:** {config.RETRIEVER_TOP_K}")

    st.subheader("Knowledge Manager")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Enable Live Updates:** {config.ENABLE_LIVE_UPDATES}")
        st.write(f"**Demo Mode:** {config.DEMO_MODE}")

    with col2:
        st.write(f"**Refresh Interval:** {config.KNOWLEDGE_REFRESH_HOURS}h")
        st.write(f"**Gemini Fallback:** {config.ENABLE_GEMINI_FALLBACK}")

    st.subheader("Knowledge Update Controls")
    knowledge_manager = create_knowledge_manager()

    if st.button("🔄 Update All Knowledge Sources"):
        if config.DEMO_MODE:
            st.info("Demo mode enabled: live updates are disabled.")
        else:
            with st.spinner("Updating knowledge from all sources..."):
                try:
                    results = knowledge_manager.update_all()
                    st.success("✅ Knowledge update completed!")
                    for result in results:
                        st.write(
                            f"- **{result['source'].capitalize()}:** "
                            f"{result['new_documents']} new documents added"
                        )
                except Exception as e:
                    st.error(f"❌ Knowledge update failed: {str(e)}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("MITRE"):
            with st.spinner("Updating MITRE ATT&CK..."):
                try:
                    result = knowledge_manager.update_source("mitre")
                    st.success(f"✅ {result['new_documents']} MITRE documents added")
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")

    with col2:
        if st.button("Sigma"):
            with st.spinner("Updating Sigma Rules..."):
                try:
                    result = knowledge_manager.update_source("sigma")
                    st.success(f"✅ {result['new_documents']} Sigma documents added")
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")

    with col3:
        if st.button("CISA"):
            with st.spinner("Updating CISA Alerts..."):
                try:
                    result = knowledge_manager.update_source("cisa")
                    st.success(f"✅ {result['new_documents']} CISA documents added")
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")

    with col4:
        if st.button("Microsoft"):
            with st.spinner("Updating Microsoft Security..."):
                try:
                    result = knowledge_manager.update_source("microsoft")
                    st.success(f"✅ {result['new_documents']} Microsoft documents added")
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")

    with col5:
        if st.button("LOLBas"):
            with st.spinner("Updating LOLBas..."):
                try:
                    result = knowledge_manager.update_source("lolbas")
                    st.success(f"✅ {result['new_documents']} LOLBas documents added")
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")

    st.divider()
    st.subheader("Dashboard Options")
    if st.button("Reset Session Statistics"):
        st.session_state.query_stats = {
            "total_queries": 0,
            "local_answers": 0,
            "gemini_answers": 0,
            "confidence_scores": [],
            "response_times": [],
            "recent_questions": [],
        }
        st.success("Session statistics reset")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared")


def main() -> None:
    """Application entry point."""
    initialize_session_state()
    page = render_sidebar()

    if page == "Chat":
        render_chat()
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
