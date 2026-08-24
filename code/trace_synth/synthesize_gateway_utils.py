from __future__ import annotations

import json
from typing import Any

import tool_detail_log
from synthesize_flow_utils import json_from_text, one_line, printable_text, sanitize_json_strings


def _log_tool_payload(label: str, payload: Any) -> None:
    """Main log keeps only a one-line summary; full inputs/results are written to the sidecar detail file."""
    tool_detail_log.log_payload(label, payload)


async def call_json_with_docker_cli_messages(
    *,
    messages: list[dict],
    trace: dict,
    enable_docker_cli: bool,
    event_kind: str,
    max_tool_calls: int,
    remind_every_tool_calls: int,
    remind_start_tool_calls: int = 0,
    wrap_up_reminder: bool = False,
    reminder_context: str = "",
    suppress_reminders: bool = False,
    chat_once,
    client_response_text,
    tool_call_command,
    log_step,
    trim_tool_output,
    docker_cli_tool,
    docker_cli_session_cls,
    skills_manager=None,
) -> Any:
    """Run one round of a JSON-generation session that may use `docker_cli`.

    This is the shared tool-calling main loop for PLAN / ATTEMPT / AUGMENT:
    - First send the model a request carrying the tool schema
    - If the model finishes its answer, directly parse the final JSON
    - If the model requests `docker_cli`, execute it in the shared worktree, then
      feed the observation back to the model as a `tool_result`
    - When a reminder threshold is reached, append a host reminder to nudge the
      model toward converging its output as soon as possible

    The reason this is factored into utils is that it is essentially a
    "multi-round JSON session protocol with tools"; it does not belong to any
    specific flow, and should not keep piling up inside the `LLMGateway` class body.
    """
    messages = [dict(m) for m in messages]
    cli = docker_cli_session_cls(trace, trim_tool_output)
    tool_call_count = 0
    try:
        while True:
            body = await chat_once(
                messages,
                tools=docker_cli_tool,
                usage_stage=event_kind,
            )
            if body.get("stop_reason") == "end_turn":
                return json_from_text(client_response_text(body))

            messages.append(body)
            content = body.get("content")
            if not isinstance(content, list):
                raise RuntimeError(
                    "Claude returned stop_reason=tool_use but content is not a list: "
                    f"{printable_text(json.dumps(body, ensure_ascii=False)[:500])}"
                )

            tool_result_blocks: list[dict[str, Any]] = []
            reminder_texts: list[str] = []
            for tool_call in content:
                if not isinstance(tool_call, dict) or tool_call.get("type") != "tool_use":
                    continue
                tool_call_count += 1
                tool_name = str(tool_call.get("name") or "").strip()
                call_id = str(tool_call.get("id") or "").strip()
                _log_tool_payload(
                    f"call event={event_kind} index={tool_call_count} tool={tool_name or '<missing>'} id={call_id}",
                    tool_call,
                )
                if tool_name == "read_skills":
                    log_step(
                        f"{event_kind.lower()}-tool",
                        detail=f"event={event_kind} index={tool_call_count} tool=read_skills call_id={call_id}",
                    )
                    if skills_manager is None:
                        obs = {"tool": "read_skills", "error": "skills library is not available for this stage"}
                    else:
                        obs = skills_manager.read(event_kind)
                elif tool_name == "patch_skills":
                    args = tool_call.get("input") or {}
                    patch_text = args.get("patch") if isinstance(args, dict) else None
                    log_step(
                        f"{event_kind.lower()}-tool",
                        detail=f"event={event_kind} index={tool_call_count} tool=patch_skills call_id={call_id}",
                    )
                    if skills_manager is None:
                        obs = {"tool": "patch_skills", "error": "skills library is not available for this stage"}
                    else:
                        obs = skills_manager.patch(event_kind, patch_text)
                elif tool_name == "reset_worktree":
                    log_step(
                        f"{event_kind.lower()}-tool",
                        detail=f"event={event_kind} index={tool_call_count} tool=reset_worktree call_id={call_id}",
                    )
                    if not enable_docker_cli:
                        obs = {
                            "tool": "reset_worktree",
                            "error": "docker_cli is disabled because --no-execute is set",
                        }
                    else:
                        obs = cli.reset()
                else:
                    call_id, command, error = tool_call_command(tool_call)
                    command_head = one_line(command, limit=160) if command else ""
                    log_step(
                        f"{event_kind.lower()}-tool",
                        detail=f"event={event_kind} index={tool_call_count} tool=docker_cli call_id={call_id} cmd={command_head}",
                    )
                    if error is not None:
                        obs = {"tool": "docker_cli", "error": error}
                        log_step(
                            f"{event_kind.lower()}-tool",
                            ok=False,
                            detail=f"event={event_kind} index={tool_call_count} tool=docker_cli call_id={call_id} cmd={command_head} parse_error={str(error)[:200]}",
                        )
                    elif not enable_docker_cli:
                        obs = {
                            "tool": "docker_cli",
                            "command": command,
                            "error": "docker_cli is disabled because --no-execute is set",
                        }
                        log_step(
                            f"{event_kind.lower()}-tool",
                            ok=False,
                            detail=f"event={event_kind} index={tool_call_count} tool=docker_cli call_id={call_id} cmd={command_head} error=disabled(no-execute)",
                        )
                    else:
                        obs = cli.run(command)
                _log_tool_payload(
                    f"result event={event_kind} index={tool_call_count} tool={tool_name or '<missing>'} id={call_id}",
                    obs,
                )
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": json.dumps(sanitize_json_strings(obs), ensure_ascii=False),
                })
                # Trigger timing: by default, start at integer multiples of
                # remind_every_tool_calls (backward compatible); if
                # remind_start_tool_calls is specified, trigger once at that count
                # first, then every remind_every_tool_calls afterwards (i.e. start,
                # start+every, start+2*every ...).
                if remind_every_tool_calls > 0:
                    if remind_start_tool_calls > 0:
                        should_remind = (
                            tool_call_count >= remind_start_tool_calls
                            and (tool_call_count - remind_start_tool_calls) % remind_every_tool_calls == 0
                        )
                    else:
                        should_remind = tool_call_count % remind_every_tool_calls == 0
                else:
                    should_remind = False
                # deepseek/sglang's OpenAI Chat Completions protocol requires that
                # assistant(tool_calls) be immediately followed by the corresponding
                # tool messages, with no text-bearing user message inserted in between.
                # Our reminder text is fed back concatenated into the same user message
                # as the tool_result; when translated into OpenAI messages the text is
                # split off into a standalone {role:user} inserted between the assistant
                # and the tool responses, triggering a 400 "must be followed by tool
                # messages". So for these backends we simply do not inject reminders (the
                # tool cap is a soft reminder anyway, skipping it does not affect correctness).
                if suppress_reminders:
                    should_remind = False
                if should_remind:
                    if wrap_up_reminder:
                        # Wrap-up reminder: do not expose remaining count/cap, only suggest converging the output as soon as possible.
                        reminder = (
                            f"Wrap-up reminder: you have explored enough; it is recommended to enter the wrap-up phase now. "
                            f"Prefer outputting the final {event_kind} JSON based on existing observations; "
                            f"only continue calling tools if critical information is genuinely missing."
                        )
                    else:
                        remaining = max(0, max_tool_calls - tool_call_count)
                        reminder = (
                            f"Tool call progress reminder: you have called tools {tool_call_count} times (docker_cli / reset_worktree),"
                            f"recommended maximum is {max_tool_calls}, {remaining} calls remaining."
                            f"Prefer outputting the final {event_kind} JSON based on existing observations; only continue calling tools if critical information is genuinely missing."
                        )
                    if event_kind == "PLAN":
                        reminder += "When writing PLAN, do not include pure-test or pure-documentation subtasks in subtasks_blueprint; only include observable behavior changes."
                    elif event_kind == "ATTEMPT":
                        reminder += (
                            f"When writing ATTEMPT, you must directly implement the current subtask_text: {reminder_context};"
                            "unless subtask_text explicitly requires adding a public API, do not add public APIs or wrapper/helper functions to bypass the actual task."
                        )
                    elif event_kind == "AUGMENT_TESTS":
                        reminder += (
                            "When writing AUGMENT_TESTS, prioritize semantically valid, verifiable, non-duplicate own_p2p entries;"
                            "only output newly added regression node-ids and necessary test_patch; do not modify other event fields or fabricate node-ids to pad the count."
                        )
                    reminder_texts.append(reminder)
            if not tool_result_blocks:
                raise RuntimeError(
                    "Claude returned stop_reason=tool_use but no tool_use blocks: "
                    f"{printable_text(json.dumps(body, ensure_ascii=False)[:500])}"
                )
            messages.append({
                "role": "user",
                "content": tool_result_blocks + [{"type": "text", "text": text} for text in reminder_texts],
            })
    finally:
        try:
            cli.close()
        except Exception as exc:
            print(
                f"[synthesize] warning: docker_cli final cleanup failed: {printable_text(exc)}",
                flush=True,
            )
