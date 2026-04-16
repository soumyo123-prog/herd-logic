"""
BaseAgent — template-method ABC for every analytical agent.

Mirrors BaseDataProvider: the base class owns the workflow; subclasses
plug in gather_features() (how to collect inputs) and
build_system_prompt() (the prompt that shapes LLM behavior). Everything
else — rendering, the structured LLM call, and optional post-processing
— is shared.

This is how Technical, Macro, Fundamental/Sentiment, Risk, and Strategist
will all share the same skeleton.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseAgent(ABC):
    name: str = "base"
    response_schema: type[BaseModel]

    def __init__(self, llm_client, providers: dict | None = None):
        self.llm = llm_client
        self.providers = providers or {}

    def run(self, **inputs) -> dict:
        features = self.gather_features(**inputs)
        system_prompt = self.build_system_prompt()
        user_message = self.render_user_message(features)
        raw = self.llm.generate_structured(
            system_prompt, user_message, self.response_schema
        )
        return self.post_process(raw, features)

    @abstractmethod
    def gather_features(self, **inputs) -> dict: ...

    @abstractmethod
    def build_system_prompt(self) -> str: ...

    def render_user_message(self, features: dict) -> str:
        return json.dumps(features, indent=2, default=str)

    def post_process(self, raw: dict, features: dict) -> dict:
        return raw
