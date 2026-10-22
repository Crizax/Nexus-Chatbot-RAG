# --- Fix for ChromaDB SQLite3 version on Streamlit Cloud ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file locally
load_dotenv()

DATA_FILE = "hospital_data.json"
DB_DIR = "./nexus_health_db"

# Priority order of active Groq production models (with automatic fallback)
PREFERRED_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

st.set_page_config(
    page_title="Nexus Health Assistant",
    page_icon="🏥",
    layout="wide"
)

# Resolve API Key silently in the background (.env or Streamlit Cloud Secrets)
api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

# --- Clean Patient-Facing Sidebar (No API Key or Model Selectors) ---
with st.sidebar:
    st.header("🏥 Nexus Health")
    st.caption("452 Innovation Blvd, Tech City")

    st.divider()
    st.subheader("🚨 Quick Contacts")
    st.markdown(
        "- **Emergency:** `(555) 010-9999`\n"
        "- **Main Desk:** `(555) 010-2000`\n"
        "- **Appointments:** `(555) 010-2020`\n"
        "- **Billing Office:** `(555) 010-2055`\n"
        "- **24/7 Crisis Line:** `(555) 010-9888`"
    )

    st.divider()
    st.subheader("🕒 Visiting Hours")
    st.markdown(
        "- **General Wards:**\n  11:00 AM – 1:00 PM\n  5:00 PM – 8:00 PM\n"
        "- **ICU / NICU:**\n  Immediate family only (10 mins every 2 hrs)"
    )

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Vector DB & Embeddings Initialization ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def initialize_vector_db():
    embeddings = get_embeddings()
    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="nexus_hospital_docs"
    )

    if not os.path.exists(DATA_FILE):
        st.error(f"Missing `{DATA_FILE}` in project directory.")
        st.stop()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    existing_count = vector_db._collection.count()

    # Auto-ingest if DB is empty or record count changed
    if existing_count != len(raw_data):
        if existing_count > 0:
            existing_ids = vector_db._collection.get()["ids"]
            if existing_ids:
                vector_db._collection.delete(ids=existing_ids)

        docs, ids = [], []
        for item in raw_data:
            clean_meta = {
                k: (", ".join(map(str, v)) if isinstance(v, list) else v)
                for k, v in item.get("metadata", {}).items()
            }
            clean_meta["record_id"] = item["id"]
            docs.append(Document(page_content=item["text"], metadata=clean_meta))
            ids.append(item["id"])

        vector_db.add_documents(documents=docs, ids=ids)

    return vector_db


vector_db = initialize_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 4})


def invoke_with_model_fallback(prompt_template, payload: dict, groq_key: str) -> str:
    """Tries preferred Groq models and automatically discovers available models on 404."""
    last_error = None

    # 1. Try preferred production models first
    for model_id in PREFERRED_MODELS:
        try:
            llm = ChatGroq(api_key=groq_key, model_name=model_id, temperature=0.1)
            chain = prompt_template | llm | StrOutputParser()
            return chain.invoke(payload)
        except Exception as e:
            last_error = e
            if "model_not_found" in str(e) or "404" in str(e) or "decommissioned" in str(e):
                continue
            raise e

    # 2. Dynamic Fallback: Query Groq API for whatever active chat model this key has access to
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        models_response = client.models.list()
        excluded_keywords = ("whisper", "guard", "tts", "orpheus", "embed", "playai")
        for m in models_response.data:
            if not any(kw in m.id.lower() for kw in excluded_keywords):
                llm = ChatGroq(api_key=groq_key, model_name=m.id, temperature=0.1)
                chain = prompt_template | llm | StrOutputParser()
                return chain.invoke(payload)
    except Exception as fallback_err:
        last_error = fallback_err

    raise last_error


# --- Main Interface ---
st.title("🏥 Nexus Health AI Support")
st.markdown("Fast, reliable patient support grounded in **Nexus Health Medical Center (NHMC)** data.")

template = """You are a professional, empathetic hospital assistant for Nexus Health Medical Center (NHMC).
Use the retrieved hospital context below to answer the patient's question accurately.

STRICT GUIDELINES:
1. EMERGENCY PROTOCOL: If the user mentions a medical emergency (e.g., chest pain, severe bleeding, difficulty breathing, stroke symptoms, trauma), immediately instruct them to call (555) 010-9999 or go to the Ground Floor North Entrance Emergency Center.
2. GROUNDED ACCURACY: Rely strictly on the provided Context. Do not invent doctor names, floor numbers, prices, or extensions.
3. FALLBACK: If the answer is not in the context, politely state that you don't have that specific detail and refer them to the main switchboard at (555) 010-2000.
4. TONE: Be concise, clear, and well-structured (use bullet points when listing services or hours).

Recent Conversation History:
{chat_history}

Retrieved Hospital Context:
{context}

Patient Question: {question}
Assistant Answer:"""

prompt = ChatPromptTemplate.from_template(template)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Welcome to **Nexus Health Medical Center**. I can help you with department locations, doctor specialties, appointment booking, insurance coverage, lab results, or visitor hours. How can I assist you today?"
        }
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📄 View Retrieved Hospital Sources"):
                for src in m["sources"]:
                    st.caption(f"**[{src['id']}]** ({src['category']}) — {src['text']}")

if user_input := st.chat_input("Ask about departments, appointments, insurance, visiting hours..."):
    if not api_key:
        st.error("Server Configuration Error: `GROQ_API_KEY` is missing from environment variables / Streamlit Secrets.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching hospital records..."):
            try:
                retrieved_docs = retriever.invoke(user_input)
                context_text = "\n\n".join(
                    f"[{d.metadata.get('record_id', 'doc')}] {d.page_content}"
                    for d in retrieved_docs
                )

                recent_turns = st.session_state.messages[-5:-1]
                history_text = "\n".join(
                    f"{m['role'].capitalize()}: {m['content']}" for m in recent_turns
                ) or "None"

                response = invoke_with_model_fallback(
                    prompt_template=prompt,
                    payload={
                        "context": context_text,
                        "chat_history": history_text,
                        "question": user_input
                    },
                    groq_key=api_key
                )

                st.markdown(response)

                sources_list = [
                    {
                        "id": d.metadata.get("record_id", "N/A"),
                        "category": d.metadata.get("category", "general"),
                        "text": d.page_content
                    }
                    for d in retrieved_docs
                ]

                with st.expander("📄 View Retrieved Hospital Sources"):
                    for src in sources_list:
                        st.caption(f"**[{src['id']}]** ({src['category']}) — {src['text']}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources_list
                })

            except Exception as e:
                st.error(f"Error connecting to AI service: {e}")