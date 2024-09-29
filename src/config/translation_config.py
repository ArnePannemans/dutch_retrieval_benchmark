import os
from dotenv import load_dotenv

# Load environment variables (like API keys) from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_NUM_SAMPLES = 1000

DATA_DIR = os.path.join(os.getcwd(), 'data')
ENGLISH_DATA_DIR = os.path.join(DATA_DIR, 'english')
DUTCH_DATA_DIR = os.path.join(DATA_DIR, 'dutch')

TRANSLATION_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a translation assistant specialized in English to Dutch."
