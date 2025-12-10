import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# API & ENV CONFIG
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# -------------------------
# APP CONSTANTS
# -------------------------
APP_TITLE = "🤖 Lesson Agent Dashboard + Exam Manager"
PAGE_LAYOUT = "wide"

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant for teachers."

DEFAULT_START_HOUR = 9
DEFAULT_END_HOUR = 17

DEFAULT_MODEL_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_MODEL_MAVERICK = "meta-llama/llama-4-maverick-17b-128e-instruct"
