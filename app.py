import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="Groq Chatbot", layout="wide")

# CSS for Styling & Animation
st.markdown("""
    <style>
        /* Page-wide font and base layout */
        html, body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f6f9fc;
        }

        /* Sidebar chat history */
        .sidebar-history {
            max-height: 80vh;
            overflow-y: auto;
        }

        .history-item {
            padding: 10px;
            margin: 5px 0;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            font-size: 15px;
        }

        /* Slide-up animation on load */
        .main-container {
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            0% { opacity: 0; transform: translateY(30px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* Response box */
        .response-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            font-size: 16px;
            line-height: 1.6;
        }

        .input-box {
            margin-top: 20px;
        }

        .title {
            text-align: center;
            font-size: 2em;
            font-weight: 600;
            color: #333333;
            margin-top: 20px;
        }

        .subtitle {
            text-align: center;
            font-size: 16px;
            color: #666666;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize model
model = init_chat_model("groq:llama-3.1-8b-instant")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide response to the user queries."),
    ("user", "Question: {question}")
])

# Chain setup
llm = ChatGroq(model="llama-3.1-8b-instant")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR: Chat History ---
with st.sidebar:
    st.markdown("### Chat History")
    st.markdown('<div class="sidebar-history">', unsafe_allow_html=True)
    for message in reversed(st.session_state.history):  # newest first
        st.markdown(f"<div class='history-item'>{message}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Clear History"):
        st.session_state.history = []

# --- MAIN INTERFACE ---
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown("<div class='title'>Groq Chatbot with LLaMA 3</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ask questions. Get intelligent answers — fast.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        input_text = st.text_input("Enter your question:", label_visibility="collapsed", placeholder="e.g., What is LangChain?", key="user_input")
    with col2:
        submitted = st.button("Generate")

    if submitted and input_text.strip() != "":
        # Add to sidebar history
        st.session_state.history.append(input_text.strip())

        with st.spinner("Generating response..."):
            response = chain.invoke({'question': input_text.strip()})

        # Display response
        st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
