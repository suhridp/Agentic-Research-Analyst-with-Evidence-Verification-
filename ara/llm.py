"""Minimal Anthropic client (stdlib only) plus a tolerant JSON parser.
Pass llm=None anywhere to run the deterministic offline heuristics instead."""
from __future__ import annotations
import hashlib, json, os, re, urllib.request
from pathlib import Path

class AnthropicLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("ARA_MODEL", "claude-sonnet-5-5")
        self.key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.in_tokens = self.out_tokens = 0
        self.cache_dir = Path(os.getenv("ARA_CACHE", ".llm_cache"))
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_hits = 0

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        key = hashlib.sha256(json.dumps([self.model, system, prompt, max_tokens]).encode()).hexdigest()
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text())["text"]
        body = json.dumps({"model": self.model, "max_tokens": max_tokens, "system": system,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
            "content-type": "application/json", "x-api-key": self.key, "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        u = data.get("usage", {})
        self.in_tokens += u.get("input_tokens", 0)
        self.out_tokens += u.get("output_tokens", 0)
        text = "".join(b.get("text", "") for b in data["content"])
        f.write_text(json.dumps({"text": text}))
        return text

def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        return None
