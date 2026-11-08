import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY was not loaded from .env")

print("API key loaded.")
print(f"Testing model: {model}")
print("Sending a small test request...", flush=True)

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: Gemini connection works.",
    )
    print("Gemini response:", response.text)
except Exception as exc:
    print(f"Gemini request failed: {type(exc).__name__}: {exc}")