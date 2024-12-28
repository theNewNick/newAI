# llama_client.py

import requests

# Replace with your WSL Tailscale IP & port:
LLAMA_URL = "http://100.83.79.118:5000/generate"

def call_local_llama(prompt, max_tokens=200):
    """
    Sends a prompt to your home Llama server running in WSL
    and returns the generated text.
    """
    data = {"prompt": prompt, "max_tokens": max_tokens}
    
    try:
        resp = requests.post(LLAMA_URL, json=data, timeout=60)
        resp.raise_for_status()  # Raises an error if status code != 200..299
        result_json = resp.json()
        return result_json.get("text", "")
    except requests.RequestException as e:
        print(f"[llama_client] Error calling Llama server at {LLAMA_URL}:", e)
        return "Error connecting to Llama"
