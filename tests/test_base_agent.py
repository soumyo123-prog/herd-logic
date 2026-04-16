"""
Tests for BaseAgent's template-method plumbing.

We use a fake LLM client and a trivial subclass to verify the run()
skeleton invokes subclass hooks in the right order and passes their
outputs correctly. No network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from agents.base_agent import BaseAgent


class _FakeLLM:
    """Minimal LLM stand-in that records the last call and returns a preset."""

    def __init__(self, reply: dict):
        self.reply = reply
        self.last_system_prompt: str | None = None
        self.last_user_message: str | None = None
        self.last_schema: type | None = None

    def generate_structured(self, system_prompt, user_message, response_schema):
        self.last_system_prompt = system_prompt
        self.last_user_message = user_message
        self.last_schema = response_schema
        return self.reply


class _EchoSchema(BaseModel):
    value: int


class _EchoAgent(BaseAgent):
    name = "echo"
    response_schema = _EchoSchema

    def gather_features(self, **inputs):
        return {"value": inputs.get("value", 42)}

    def build_system_prompt(self) -> str:
        return "You are a test agent."


def test_base_agent_runs_template_method():
    print("TEST: BaseAgent.run invokes hooks in order and returns validated dict")
    llm = _FakeLLM(reply={"value": 42})
    agent = _EchoAgent(llm_client=llm)
    result = agent.run(value=42)

    assert result == {"value": 42}, f"got {result}"
    assert llm.last_system_prompt == "You are a test agent."
    assert llm.last_schema is _EchoSchema
    assert "42" in llm.last_user_message  # features were JSON-dumped into the message
    print("  PASSED")


def test_base_agent_post_process_can_enrich():
    print("TEST: subclass post_process can transform LLM output")
    class _DoubleAgent(_EchoAgent):
        def post_process(self, raw, features):
            return {"doubled": raw["value"] * 2}

    llm = _FakeLLM(reply={"value": 21})
    agent = _DoubleAgent(llm_client=llm)
    result = agent.run(value=21)
    assert result == {"doubled": 42}, f"got {result}"
    print("  PASSED")


if __name__ == "__main__":
    test_base_agent_runs_template_method()
    test_base_agent_post_process_can_enrich()
    print("\nAll BaseAgent tests passed.")
