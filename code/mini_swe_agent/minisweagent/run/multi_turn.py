"""Multi-turn agent runner.

Supports external orchestrators that:
  1. Run agent on a task → save trajectory
  2. Read the trajectory, decide next instruction
  3. Inject next instruction with history inherited → run again

Usage example
-------------
from minisweagent.run.multi_turn import MultiTurnAgent, load_session, save_session
from minisweagent.models import get_model
from minisweagent.environments import get_environment

model = get_model(config={...})
env   = get_environment(config={...})
agent = MultiTurnAgent(model, env, system_template=..., instance_template=...)

SESSION = Path("runs/my_session/session.traj.json")

# ── Turn 1 ──
agent.run("Analyze this bug")
save_session(agent, SESSION)

# ── External function reads SESSION, decides next instruction, then: ──
messages = load_session(SESSION)
agent.continue_with("OK, now write the patch that fixes it", loaded_messages=messages)
save_session(agent, SESSION)
"""

import json
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow


class MultiTurnAgent(DefaultAgent):
    """DefaultAgent subclass that supports multi-turn continuation.

    The first turn is started with the standard ``run(task)`` method.
    Subsequent turns call ``continue_with(user_message)``, optionally
    reloading messages from a saved trajectory first.
    """

    def continue_with(
        self,
        user_message: str,
        loaded_messages: list[dict] | None = None,
    ) -> dict:
        """Append a new user message to existing history and keep running.

        Parameters
        ----------
        user_message:
            The next instruction from the external orchestrator.
        loaded_messages:
            If given, replace ``self.messages`` with these before appending
            the new user message.  Pass the list returned by
            :func:`load_session` when the agent object was re-created
            between turns.

        Returns
        -------
        dict
            Same as ``run()``: the ``extra`` dict of the final exit message
            (contains ``exit_status`` and ``submission``).
        """
        if loaded_messages is not None:
            self.messages = list(loaded_messages)

        self.add_messages(
            self.model.format_message(role="user", content=user_message)
        )

        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break

        return self.messages[-1].get("extra", {})


# ── Session helpers ────────────────────────────────────────────────────────────

def save_session(agent: DefaultAgent, path: Path, **extra) -> None:
    """Save full agent state (messages + meta) to *path*."""
    agent.save(path, extra)


def load_session(path: Path) -> list[dict]:
    """Load message history from a saved trajectory file.

    Returns the ``messages`` list, ready to pass to
    ``MultiTurnAgent.continue_with(loaded_messages=...)``.
    """
    data = json.loads(Path(path).read_text())
    messages = data.get("messages")
    if messages is None:
        raise ValueError(f"No 'messages' key found in {path}")
    return messages
