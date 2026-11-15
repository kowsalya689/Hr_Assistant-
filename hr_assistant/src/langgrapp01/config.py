#1.API keys and configuration settings for the LangGraph project.
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

#2.Model Configuration
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

#3.RAG Configuration
from pathlib import Path
HANDBOOK_PATH = Path("langgrapp01/data/PROITBRIDGE_Employee_Handbook_2026_1.pdf")
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

#Throw run time error if any of the required environment variables are missing
def check_config() -> None:
    """Fail early with a clear message if a key or the handbook is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    if LANGSMITH_TRACING and not LANGSMITH_API_KEY:
        raise RuntimeError(
            "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is not set. "
            "Add the key to your .env file."
        )
    if not HANDBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Employee Handbook not found at: {HANDBOOK_PATH}\n"
            "Place PROITBRIDGE_Employee_Handbook_2026_1.pdf inside the data/ folder."
        )