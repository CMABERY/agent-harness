"""An autonomous coding-agent harness. Standard library only.

Entry points: ``agent.loop.Agent`` for embedding, ``python -m agent`` for the CLI.
"""

from agent.loop import Agent, Runtime
from agent.types import Limits, Outcome

__all__ = ["Agent", "Limits", "Outcome", "Runtime"]
