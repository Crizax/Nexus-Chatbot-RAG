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

# Load environment variables from .env file if present
load_dotenv()

DATA_FILE = "hospital_data.json"
DB_DIR = "./nexus_health_db"

st.set_page_config(
    page_title="Nexus Health Assistant",
    page_icon="🏥",
    layout="wide"
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Resolve API Key from .env, st.secrets, or Sidebar Input
    default_key = os.getenv("GROQ_API_KEY", "")
    if not default_key:
        try:
            default_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            default_key = ""

    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        placeholder="gsk_...",
        help="Enter your free Groq API key from console.groq.com"
    )

    model_name = st.selectbox(
        "Select Groq Model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ],
        index=0,
        help="Switch models if one hits free-tier rate limits."
    )

    k_docs = st.slider("Retrieved Context Chunks (k)", min_value=2, max_value=8, value=4)

    st.divider()
    st.subheader("🚨 Quick Contacts")
    st.markdown(
        "- **Emergency:** `(555) 010-9999`\n"
        "- **Main Desk:** `(555) 010-2000`\n"
        "- **Appointments:** `(555) 010-2020`\n"
        "- **Crisis Line:** `(555) 010-9888`"
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        rebuild_db = st.button("🔄 Sync Data", use_container_width=True, help="Re-index hospital_data.json")


# --- Vector DB & Embeddings Initialization ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def initialize_vector_db(force_rebuild: bool = False):
    embeddings = get_embeddings()
    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="nexus_hospital_docs"
    )

    # Load JSON records
    if not os.path.exists(DATA_FILE):
        st.error(f"Missing `{DATA_FILE}` in project directory.")
        st.stop()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Check current vector count vs JSON count
    existing_count = vector_db._collection.count()

    if force_rebuild or existing_count != len(raw_data):
        if existing_count > 0:
            # Clear outdated records before re-indexing
            existing_ids = vector_db._collection.get()["ids"]
            if existing_ids:
                vector_db._collection.delete(ids=existing_ids)

        docs = []
        ids = []
        for item in raw_data:
            # Flatten list metadata values to strings for Chroma compatibility
            clean_meta = {
                k: (", ".join(map(str, v)) if isinstance(v, list) else v)
                for k, v in item.get("metadata", {}).items()
            }
            clean_meta["record_id"] = item["id"]
            docs.append(Document(page_content=item["text"], metadata=clean_meta))
            ids.append(item["id"])

        vector_db.add_documents(documents=docs, ids=ids)

    return vector_db


# Build or sync database
vector_db = initialize_vector_db(force_rebuild=rebuild_db)
if rebuild_db:
    st.sidebar.success("Vector database synced with hospital_data.json!")

retriever = vector_db.as_retriever(search_kwargs={"k": k_docs})

# --- Main Interface ---
st.title("🏥 Nexus Health AI Support")
st.markdown("Fast, reliable patient support grounded in **Nexus Health Medical Center (NHMC)** data.")

# Prompt Template with Chat History
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

# Initialize Session State Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Welcome to **Nexus Health Medical Center**. I can help you with department locations, doctor specialties, appointment booking, insurance coverage, lab results, or visitor hours. How can I assist you today?"
        }
    ]

# Display Chat Messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📄 View Retrieved Hospital Sources"):
                for src in m["sources"]:
                    st.caption(f"**[{src['id']}]** ({src['category']}) — {src['text']}")

# Handle User Input
if user_input := st.chat_input("Ask about departments, appointments, insurance, visiting hours..."):
    if not api_key:
        st.warning("⚠️ Please enter your Groq API Key in the left sidebar (or set `GROQ_API_KEY` in a `.env` file) to continue.")
        st.stop()

    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant documents & generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching hospital records..."):
            try:
                retrieved_docs = retriever.invoke(user_input)
                context_text = "\n\n".join(
                    f"[{d.metadata.get('record_id', 'doc')}] {d.page_content}"
                    for d in retrieved_docs
                )

                # Format last 4 turns of conversation history
                recent_turns = st.session_state.messages[-5:-1]
                history_text = "\n".join(
                    f"{m['role'].capitalize()}: {m['content']}" for m in recent_turns
                ) or "None"

                llm = ChatGroq(
                    api_key=api_key,
                    model_name=model_name,
                    temperature=0.1
                )

                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({
                    "context": context_text,
                    "chat_history": history_text,
                    "question": user_input
                })

                st.markdown(response)

                # Format source metadata for UI transparency
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
                st.error(f"Error connecting to Groq API: {e}")