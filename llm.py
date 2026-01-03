import requests
import os

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}"
}

def ask_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.4,
            "return_full_text": False
        }
    }
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()[0]["generated_text"]
