"""
LLM-Based Pair Verifier (Ollama)
================================

Uses a local LLM via Ollama to verify that a candidate Polymarket <->
Kalshi pair actually has IDENTICAL resolution criteria. Catches cases
like:

- Polymarket resolves YES on announcement; Kalshi only on actual departure
- Role-specific vs. administration-wide
- Same person, different dates
- "Win election" vs. "Win nomination"

This is a correctness gate, not a speed filter. It should only be run
AFTER the lexical matcher has produced a candidate list.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx


logger = logging.getLogger(__name__)


@dataclass
class VerifierDecision:
    verdict: str           # "equivalent", "different", "uncertain"
    confidence: float      # 0-1
    reasoning: str
    raw_response: str = ""

    @property
    def is_equivalent(self) -> bool:
        return self.verdict == "equivalent"


class OllamaVerifier:
    """
    Async Ollama client that asks a local LLM whether two prediction
    markets resolve identically.
    """

    DEFAULT_PROMPT = """You are a prediction-market arbitrage auditor. You are given one market from \
Polymarket and one from Kalshi. Your job is to decide whether BOTH markets resolve \
YES under EXACTLY the same real-world conditions.

Be strict. Markets are NOT equivalent if ANY of the following differ:
- The covered time window (e.g. "before 2027" vs "this year" when years line up \
matters less than whether the cutoff dates agree)
- What triggers YES (e.g. announcement vs. actual occurrence, resolution vs. \
settlement)
- Which person, role, or entity is covered (e.g. "Hegseth leaves Secretary of \
Defense" vs. "Hegseth leaves the administration")
- Resolution sources, tie-breakers, or edge-case handling that could flip the \
outcome
- Scope (single outcome vs. basket, first-event vs. ever-happens)

Output ONLY a JSON object with keys:
  verdict: "equivalent" | "different" | "uncertain"
  confidence: float 0.0-1.0
  reasoning: string (1-3 short sentences)

POLYMARKET:
  Question: {poly_question}
  Rules/Description:
  {poly_rules}

KALSHI:
  Question: {kalshi_question}
  Rules:
  {kalshi_rules}

JSON response:"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 60.0,
        max_concurrency: int = 2,
        max_rules_chars: int = 2500,
        temperature: float = 0.0,
        disable_thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_rules_chars = max_rules_chars
        self.temperature = temperature
        self.disable_thinking = disable_thinking
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "OllamaVerifier":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Return True if the Ollama server is reachable and the model is pulled."""
        try:
            assert self._client is not None
            resp = await self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            names = {t.get("name", "") for t in tags}
            if self.model not in names and not any(n.startswith(self.model) for n in names):
                logger.warning(
                    f"Ollama model {self.model!r} not found locally. "
                    f"Available: {sorted(names)}. Pull with: ollama pull {self.model}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed at {self.base_url}: {e}")
            return False

    def _truncate(self, text: str) -> str:
        text = (text or "").strip()
        if len(text) <= self.max_rules_chars:
            return text
        return text[: self.max_rules_chars] + "\n...[truncated]"

    async def verify_pair(
        self,
        poly_question: str,
        poly_rules: str,
        kalshi_question: str,
        kalshi_rules: str,
    ) -> VerifierDecision:
        """Ask the LLM whether two markets resolve identically."""
        assert self._client is not None, "Use 'async with OllamaVerifier(...)'"

        prompt = self.DEFAULT_PROMPT.format(
            poly_question=poly_question.strip(),
            poly_rules=self._truncate(poly_rules) or "(no rules provided)",
            kalshi_question=kalshi_question.strip(),
            kalshi_rules=self._truncate(kalshi_rules) or "(no rules provided)",
        )

        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        if self.disable_thinking:
            # Turn off "thinking" on reasoning models (qwen3.x, deepseek-r1,
            # gpt-oss). Ignored for non-thinking models.
            body["think"] = False

        async with self._sem:
            try:
                resp = await self._client.post(f"{self.base_url}/api/generate", json=body)
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("response", "").strip()
            except Exception as e:
                logger.warning(f"Ollama request failed: {e}")
                return VerifierDecision("uncertain", 0.0, f"LLM request failed: {e}")

        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> VerifierDecision:
        if not raw:
            return VerifierDecision("uncertain", 0.0, "empty LLM response", raw)

        # Try direct JSON parse, then try to extract a JSON object from the text
        data: Optional[dict] = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None

        if not isinstance(data, dict):
            return VerifierDecision(
                "uncertain", 0.0, "could not parse LLM JSON", raw
            )

        verdict = str(data.get("verdict", "uncertain")).lower().strip()
        if verdict not in ("equivalent", "different", "uncertain"):
            verdict = "uncertain"

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(data.get("reasoning", "")).strip()[:500]

        return VerifierDecision(verdict, confidence, reasoning, raw)
