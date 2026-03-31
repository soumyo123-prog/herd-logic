"""
LLM Client — wraps OpenRouter via the OpenAI-compatible SDK.

This is the foundation of the multi-agent system. Every agent uses this client
to talk to the LLM. It provides two modes:

1. generate_structured() — Forces the LLM to return JSON matching a Pydantic schema.
   Used by agents to produce machine-readable reports that other agents can consume.

2. generate_text() — Returns free-form text. Used for reasoning summaries or
   when structured output isn't needed.

OpenRouter exposes an OpenAI-compatible API, so we use the openai SDK pointed
at OpenRouter's base URL. This also means swapping to Ollama/Groq/etc. later
is a one-line config change.
"""

import json
import os
import re
import time

from openai import OpenAI
from pydantic import BaseModel

from agents.vault import Vault

# Default retry count for transient provider errors (403, 502, 503).
# Free models are routed through multiple backend providers on OpenRouter.
# Some providers may be temporarily down or out of credits — retrying
# lets OpenRouter pick a different healthy provider.
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2


# Ordered list of free models to try. If the primary model's providers are all
# down, we fall through to the next one rather than failing the entire pipeline.
DEFAULT_FALLBACK_MODELS = [
    "minimax/minimax-m2.5:free",
    "qwen/qwen3.6-plus-preview:free",
]


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0,
        max_retries: int = MAX_RETRIES,
        fallback_models: list[str] | None = None,
    ):
        self.model = model or os.getenv("LLM_MODEL", "minimax/minimax-m2.5:free")
        self.temperature = temperature
        self.max_retries = max_retries
        self.fallback_models = fallback_models or DEFAULT_FALLBACK_MODELS

        # API key resolution order: explicit arg → env var → macOS Keychain
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            api_key = Vault().get("OPENROUTER_API_KEY")

        self.client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
        )

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: type[BaseModel],
    ) -> dict:
        """
        Ask the LLM a question and get back validated JSON.

        How it works:
        1. We send the system prompt + user message to the LLM.
        2. We tell the LLM "respond as JSON matching this schema" via response_format.
        3. The LLM returns a JSON string (possibly with <think> blocks prepended).
        4. We parse it and validate against the Pydantic schema.
        5. If any field is missing or wrong type, Pydantic raises an error
           instead of silently passing bad data downstream.

        Returns a plain dict (not a Pydantic object) so it's easy to serialize,
        store in the DB, or pass to the next agent.
        """
        # Include the schema in the system prompt so the model knows the exact
        # shape to produce. We also mention "json" explicitly because some
        # providers (e.g. Alibaba/Qwen) require it when using response_format.
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        json_system_prompt = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with valid JSON matching this exact schema (no wrapper object, no extra keys):\n"
            f"{schema_json}"
        )

        response = self._call_with_retry(
            messages=[
                {"role": "system", "content": json_system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content

        # Strip <think>...</think> blocks that MiniMax M2.5 prepends
        parsed = self._extract_json(raw_content)

        # Some models wrap the response in a single-key object like
        # {"stock_analysis": {...actual data...}}. Unwrap it.
        schema_fields = set(response_schema.model_fields.keys())
        if set(parsed.keys()) != schema_fields and len(parsed) == 1:
            inner = next(iter(parsed.values()))
            if isinstance(inner, dict):
                parsed = inner

        # Validate against the Pydantic schema
        validated = response_schema.model_validate(parsed)
        return validated.model_dump()

    def generate_text(self, system_prompt: str, user_message: str) -> str:
        """
        Ask the LLM a question and get back free-form text.
        Used for reasoning summaries or when you don't need structured output.
        """
        response = self._call_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def _call_with_retry(self, messages: list, **kwargs):
        """
        Make the API call with retries and model fallback.

        Strategy:
        1. Try the primary model, skipping failed providers on each retry.
        2. If ALL providers for a model are down (404 "All providers ignored"),
           fall through to the next model in the fallback list.
        3. If all models are exhausted, raise the last error.
        """
        from openai import APIStatusError

        # Build the list of models to try: primary first, then fallbacks
        models_to_try = [self.model]
        for m in self.fallback_models:
            if m != self.model and m not in models_to_try:
                models_to_try.append(m)

        last_error = None

        for model in models_to_try:
            ignored_providers: list[str] = []

            for attempt in range(self.max_retries):
                try:
                    extra_body: dict = {"reasoning": {"enabled": True}}

                    if ignored_providers:
                        extra_body["provider"] = {"ignore": ignored_providers}

                    return self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        extra_body=extra_body,
                        **kwargs,
                    )
                except APIStatusError as e:
                    last_error = e

                    # "All providers ignored" → this model is fully down, try next model
                    if e.status_code == 404 and "All providers" in str(e):
                        print(f"  All providers down for {model}, trying next model...")
                        break

                    if e.status_code in (403, 429, 502, 503):
                        failed_provider = self._extract_provider_name(e)
                        if failed_provider and failed_provider not in ignored_providers:
                            ignored_providers.append(failed_provider)

                        delay = RETRY_DELAY_SEC * (attempt + 1)
                        print(
                            f"  Provider error ({e.status_code})"
                            f"{f' from {failed_provider}' if failed_provider else ''}"
                            f" on {model}, retrying in {delay}s... "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise

        raise last_error

    @staticmethod
    def _extract_provider_name(error) -> str | None:
        """Extract the provider name from an OpenRouter error response."""
        try:
            body = error.response.json()
            return body.get("error", {}).get("metadata", {}).get("provider_name")
        except Exception:
            return None

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        MiniMax M2.5 has mandatory chain-of-thought: it outputs <think>...</think>
        before the actual JSON. This method strips the thinking block and parses
        the JSON portion.

        Also handles the case where the model wraps JSON in ```json ... ``` markdown.
        """
        # Strip <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        return json.loads(text)
