from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedding_model():
    """Initializes the Gemini Embeddings model."""
    api_key = os.getenv("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
