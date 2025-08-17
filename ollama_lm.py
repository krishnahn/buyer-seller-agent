import requests
from typing import List, Optional

class OllamaLanguageModel:
    """
    Minimal Ollama adapter for local Llama 3.
    Uses /api/generate (non-streaming) for simplicity.
    """
    def __init__(self, model_name: str = "llama3:8b", host: str = "http://localhost:11434"):
        self.model = model_name
        self.host = host.rstrip("/")

    def generate(self, prompt: str, stop: Optional[List[str]] = None, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature}
        }
        if stop:
            payload["stop"] = stop
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Depending on Ollama version, you may need to handle streamed chunks.
        # For most recent versions, non-streaming /api/generate returns response in one JSON.
        return data.get("response", "").strip()
