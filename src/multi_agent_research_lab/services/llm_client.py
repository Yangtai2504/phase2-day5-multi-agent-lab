"""LLM client backed by Gemini 2.5 Flash on Vertex AI.

Uses google-auth (already installed) + httpx to call the Vertex REST API directly,
avoiding the heavy google-cloud-aiplatform dependency.

Auth: GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service-account JSON.
"""

from dataclasses import dataclass

import google.auth
import google.auth.transport.requests
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

# Gemini 2.5 Flash pricing (USD / 1M tokens, as of 2025)
_INPUT_COST_PER_M = 0.15
_OUTPUT_COST_PER_M = 0.60


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _get_access_token() -> str:
    """Refresh and return a short-lived OAuth2 access token from the service account."""
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token  # type: ignore[return-value]


class LLMClient:
    """Gemini 2.5 Flash on Vertex AI — minimal REST client."""

    def __init__(self) -> None:
        settings = get_settings()
        self._project = settings.google_cloud_project
        self._location = settings.vertex_location
        self._model = settings.vertex_model
        self._timeout = settings.timeout_seconds

    def _endpoint(self) -> str:
        return (
            f"https://{self._location}-aiplatform.googleapis.com/v1"
            f"/projects/{self._project}/locations/{self._location}"
            f"/publishers/google/models/{self._model}:generateContent"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> LLMResponse:
        """Call Gemini on Vertex and return a typed response with token counts."""
        token = _get_access_token()
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2,
            },
        }
        resp = httpx.post(
            self._endpoint(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount")
        out_tok = usage.get("candidatesTokenCount")
        cost = None
        if in_tok is not None and out_tok is not None:
            cost = (in_tok * _INPUT_COST_PER_M + out_tok * _OUTPUT_COST_PER_M) / 1_000_000

        return LLMResponse(content=content, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
