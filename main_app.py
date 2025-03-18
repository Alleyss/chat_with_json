import streamlit as st
import json
from json_processor import process_json
from vector_store import init_vector_store, add_json_to_vector, retrieve_json_from_vector
from embeddings import get_embedding_model
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
def get_gemini_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

st.set_page_config(page_title="JSON Chatbot", layout="wide")

st.title("Chat with Your JSON File 📄🤖")

# Session ID Management
if "session_id" not in st.session_state:
    st.session_state.session_id = "json_chat_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

session_id = st.session_state.session_id
embedding_model = get_embedding_model()
vector_store = init_vector_store(session_id, embedding_model)
llm = get_gemini_llm()

# JSON Upload Section
uploaded_file = st.file_uploader("Upload a JSON file", type="json")

if uploaded_file:
    st.success("JSON file uploaded successfully!")
    json_data = json.load(uploaded_file)
    
    # Process and Add to Vector Store
    processed_data = process_json(json_data)
    add_json_to_vector(session_id, processed_data)
    st.session_state["json_data"] = json_data
    st.session_state["vector_ready"] = True

# Chat Interface
if "vector_ready" in st.session_state:
    st.subheader("Chat with the JSON Data 📩")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input Field
    user_input = st.chat_input("Ask me anything about the JSON data...")

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant context
        context = retrieve_json_from_vector(session_id, user_input)

        # Generate response from LLM
            
        system_prompt = """
        You are an AI that extracts faculty details from structured JSON data. 
        The JSON contains faculty details in the following structure:

        [
            {
                "name": "Dr. Alice Johnson",
                "department": "Computer Science",
                "subjects": ["Machine Learning", "Artificial Intelligence"],
                "cabin_number": "C-101",
                "email": "alice.johnson@university.edu",
                "phone": "+1-123-456-7890"
            }
        ]

        - Extract information based on the provided data.
        - If a faculty member exists, return ALL details.
        - If not found, state that it is missing.
        """

        full_prompt = f"{system_prompt}\n\nUser Question: {user_input}\nContext: {context}"
        response = llm.invoke(full_prompt)

    # st.write("🤖 **Bot:**", response)

        # Display AI response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
