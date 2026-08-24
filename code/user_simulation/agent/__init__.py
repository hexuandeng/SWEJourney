"""User-sim assistant agent extensions.

This package holds the *user-sim only* additions on top of mini-swe-agent:
a `reply_to_user` tool that the assistant uses to ask the simulated user a
question or deliver its final answer, and which ends the current turn so
control returns to the simulated user.

NOTHING here modifies mini-swe-agent itself — everything is a subclass or a
local copy living under user_simulation/agent/.
"""
