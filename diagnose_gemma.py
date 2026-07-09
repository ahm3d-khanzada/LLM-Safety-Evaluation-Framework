"""
Diagnostic: raw /v1/chat/completions call to google/gemma-4-12b, bypassing
LangChain entirely, to see exactly what LM Studio returns.

Run on YOUR machine:
    python3 diagnose_gemma.py
"""
import json
import os

import requests

BASE_URL = "http://192.168.50.174:1234/v1"
MODEL_ID = "google/gemma-4-12b"
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "")

payload = {
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": "Say hello in 5 words."}],
    "max_tokens": 512,
    "temperature": 0.0,
}

resp = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json=payload,
    timeout=60,
)

print("HTTP status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))

