"""
Tests two ways to avoid the "all tokens spent on thinking" problem:
  1. Appending "/no_think" to the prompt (Qwen3-family convention to skip
     the reasoning phase).
  2. Passing chat_template_kwargs={"enable_thinking": False} in the request
     body (supported by some llama.cpp/LM Studio backends for Qwen3 models).

Run on YOUR machine:
    python3 diagnose_lmstudio_v2.py
"""
import json
import os

import requests

BASE_URL = "http://192.168.50.174:1234/v1"
MODEL_ID = "qwen/qwen3.5-9b"
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def call(payload, label):
    print(f"\n=== {label} ===")
    resp = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=60)
    data = resp.json()
    msg = data["choices"][0]["message"]
    print("finish_reason:", data["choices"][0].get("finish_reason"))
    print("content:", repr(msg.get("content"))[:300])
    print("reasoning_content present:", bool(msg.get("reasoning_content")))
    print("usage:", data.get("usage"))


# Test 1: /no_think suffix
call({
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": "Say hello in 5 words. /no_think"}],
    "max_tokens": 512,
    "temperature": 0.0,
}, "Test 1: /no_think suffix")

# Test 2: enable_thinking=False via chat_template_kwargs
call({
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": "Say hello in 5 words."}],
    "max_tokens": 512,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}, "Test 2: chat_template_kwargs enable_thinking=False")
