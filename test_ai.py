from dotenv import load_dotenv
load_dotenv()

from ai_helpers import ask_ai_chat
import time

print("Empezando llamada...")
start = time.time()

result = ask_ai_chat(
    "You are a helpful assistant. Respond with ONLY a JSON object with one field: \"answer\".",
    "my dad has fever"
)

print("Tardó:", time.time() - start, "segundos")
print("Resultado:", result)