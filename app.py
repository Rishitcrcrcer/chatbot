import os
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥", layout="centered")


# ---------- Helpers ----------

def extract_video_id(url_or_id: str) -> str:
    """Extract the YouTube video ID from a full URL or return as-is if already an ID."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # Fallback: assume the user pasted a raw video ID
    if len(url_or_id.strip()) == 11:
        return url_or_id.strip()
    raise ValueError("Could not extract a valid video ID from the input.")


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_vector_store(video_id: str, _embeddings):
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id)
    transcript = " ".join([entry.text for entry in transcript_list])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, _embeddings)
    return vector_store, len(chunks)


def ask_question(question, retriever, llm):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based only on the provided video transcript context.
If the answer isn't in the context, say you don't know.

Context: {context}
Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    return response.content


# ---------- Sidebar: config ----------

with st.sidebar:
    st.header("⚙️ Setup")
    groq_key = st.text_input("Groq API Key", type="password", help="Get a free key at console.groq.com")
    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
    )
    st.divider()
    st.caption("Free stack: HuggingFace embeddings (local) + FAISS (local) + Groq (free-tier LLM).")

# ---------- Main UI ----------

st.title("🎥 YouTube RAG Chatbot")
st.write("Paste a YouTube link, then ask questions about the video's content.")

video_input = st.text_input("YouTube URL or Video ID", placeholder="https://www.youtube.com/watch?v=...")
load_btn = st.button("Load Video", type="primary")

if "video_ready" not in st.session_state:
    st.session_state.video_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

if load_btn:
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar first.")
    elif not video_input:
        st.error("Please paste a YouTube URL or video ID.")
    else:
        try:
            with st.spinner("Fetching transcript and building knowledge base..."):
                video_id = extract_video_id(video_input)
                os.environ["GROQ_API_KEY"] = groq_key
                embeddings = get_embeddings()
                vector_store, num_chunks = build_vector_store(video_id, embeddings)
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                st.session_state.llm = ChatGroq(model=model_name, temperature=0.2)
                st.session_state.video_ready = True
                st.session_state.messages = []
            st.success(f"Video loaded — {num_chunks} chunks indexed. Ask away below!")
        except Exception as e:
            st.session_state.video_ready = False
            st.error(f"Something went wrong: {e}")

st.divider()

# ---------- Chat interface ----------

if st.session_state.video_ready:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask something about the video..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(question, st.session_state.retriever, st.session_state.llm)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Load a video above to start chatting.")
