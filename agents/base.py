from __future__ import annotations

from dataclasses import dataclass, field

from utils.prompt_loader import load_prompt


@dataclass(frozen=True, slots=True)
class AgentBlueprint:
    """Shared metadata for a single agent node."""

    name: str
    prompt_name: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)


def create_agent_blueprint(
    *,
    name: str,
    prompt_name: str,
    tools: list[str] | None = None,
) -> AgentBlueprint:
    """Load a prompt and return a lightweight agent blueprint."""
    return AgentBlueprint(
        name=name,
        prompt_name=prompt_name,
        system_prompt=load_prompt(prompt_name),
        tools=tools or [],
    )
def build_agent_message(agent_name: str, content: str) -> dict[str, str]:
    """Build a minimal trace message for the shared state."""
    return {
        "role": "assistant",
        "name": agent_name,
        "content": content,
    }
