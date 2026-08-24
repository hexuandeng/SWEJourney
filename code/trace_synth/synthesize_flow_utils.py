from __future__ import annotations

import json
import re
from inspect import isawaitable
from typing import Any


class GeneratedJSONParseError(ValueError):
    """Indicates the model output could not be directly parsed into a complete JSON value."""

    def __init__(self, text: str, cause: json.JSONDecodeError):
        self.raw_text = text
        super().__init__(
            f"model output is not valid JSON: {cause.msg} "
            f"(line {cause.lineno}, column {cause.colno}, char {cause.pos})"
        )


def json_from_text(text: str) -> Any:
    text = text.strip()
    if text.startswith("[LLM Error]"):
        raise RuntimeError(text)
    if text.startswith("```"):
        m = re.search(r"```(?:json|jsonc)?\s*(.*?)```", text, re.S)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        start = min([i for i in [text.find("{"), text.find("[")] if i >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError as nested_error:
                raise GeneratedJSONParseError(text, nested_error) from nested_error
        raise GeneratedJSONParseError(text, first_error) from first_error


def printable_text(text: Any) -> str:
    """Escape malformed Unicode so gateway payloads remain safe to log."""
    return str(text or "").encode("utf-8", errors="backslashreplace").decode("utf-8")


def find_invalid_surrogate(value: Any, *, path: str = "$") -> str | None:
    """Return the first JSON path containing a Unicode surrogate code unit."""
    if isinstance(value, str):
        for idx, char in enumerate(value):
            codepoint = ord(char)
            if 0xD800 <= codepoint <= 0xDFFF:
                return f"{path}[{idx}] contains surrogate code unit U+{codepoint:04X}"
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            key_error = find_invalid_surrogate(key, path=f"{path}.<key>")
            if key_error is not None:
                return key_error
            item_error = find_invalid_surrogate(item, path=f"{path}.{key}")
            if item_error is not None:
                return item_error
        return None
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            item_error = find_invalid_surrogate(item, path=f"{path}[{idx}]")
            if item_error is not None:
                return item_error
    return None


def sanitize_json_strings(value: Any) -> Any:
    """Escape invalid surrogates in host-generated JSON-compatible values."""
    if isinstance(value, str):
        return printable_text(value)
    if isinstance(value, dict):
        return {
            printable_text(key): sanitize_json_strings(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json_strings(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_strings(item) for item in value]
    return value


def one_line(text: Any, *, limit: int = 220) -> str:
    for raw in printable_text(text).splitlines():
        line = raw.strip()
        if line:
            return line[:limit]
    return ""


class BaseFlow:
    """Shared generate -> quality_validate -> retry skeleton.

    All flows reuse the same outer protocol layer:
    1. Call `generate(...)` to obtain the raw model output
    2. Run host-side `quality_validate(...)`
    3. If it fails, feed the normalized previous output back into the next retry
    4. After exceeding `max_attempts`, return the final error message

    Additionally, in v2 the "automatic repair after JSON parse failure" is unified
    down here as well: if the subclass/caller provides a `json_repair` callback,
    then on encountering `GeneratedJSONParseError` it will first attempt repair,
    then continue with the original quality validation.
    """

    def __init__(self, *, max_attempts: int = 3, json_repair=None):
        """Record the max number of retries, plus an optional JSON-repair callback."""
        self.max_attempts = max_attempts
        self.json_repair = json_repair

    async def generate(self, attempt_no: int, last_obj: Any, last_error: Exception | None) -> Any:
        raise NotImplementedError

    def quality_validate(self, obj: Any) -> Any:
        return obj

    async def try_json_repair(self, error: GeneratedJSONParseError) -> Any:
        """Attempt the unified JSON-repair flow when JSON parsing fails.

        By default, if `json_repair` is not configured, the original exception is
        re-raised directly. If a repair callback is configured, the raw text and the
        parse exception are handed to it. The repairer must be responsible for
        returning a result that "can already be parsed into a Python object", rather
        than returning another chunk of text still awaiting parsing.
        """
        if self.json_repair is None:
            raise error
        return await self.json_repair(error.raw_text, error)

    def normalize_retry_obj(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return obj
        return {"_raw": obj}

    async def run(self) -> tuple[Any | None, str | None]:
        """Run the generic retry loop for one flow round."""
        last_obj: Any = None
        last_error: Exception | None = None
        for attempt_no in range(1, self.max_attempts + 1):
            raw_obj: Any = None
            try:
                print(
                    f"[synthesize] step flow-loop: starting  flow={type(self).__name__} "
                    f"attempt={attempt_no}/{self.max_attempts} "
                    f"last_error={type(last_error).__name__ if last_error else 'none'}",
                    flush=True,
                )
                raw_obj = await self.generate(attempt_no, last_obj, last_error)
                validated = self.quality_validate(raw_obj)
                if isawaitable(validated):
                    validated = await validated
                print(
                    f"[synthesize] step flow-loop: ok  flow={type(self).__name__} "
                    f"attempt={attempt_no}/{self.max_attempts}",
                    flush=True,
                )
                return validated, None
            except GeneratedJSONParseError as e:
                print(
                    f"[synthesize] step flow-json-parse: FAIL  flow={type(self).__name__} "
                    f"attempt={attempt_no}/{self.max_attempts} "
                    f"error={type(e).__name__}: {str(e)[:300]}",
                    flush=True,
                )
                repaired_obj: Any = None
                try:
                    repaired_obj = await self.try_json_repair(e)
                    print(
                        f"[synthesize] step flow-json-repair: ok  flow={type(self).__name__} "
                        f"attempt={attempt_no}/{self.max_attempts}",
                        flush=True,
                    )
                    validated = self.quality_validate(repaired_obj)
                    if isawaitable(validated):
                        validated = await validated
                    print(
                        f"[synthesize] step flow-loop: ok  flow={type(self).__name__} "
                        f"attempt={attempt_no}/{self.max_attempts} source=json-repair",
                        flush=True,
                    )
                    return validated, None
                except GeneratedJSONParseError as repair_parse_error:
                    print(
                        f"[synthesize] step flow-json-repair: FAIL  flow={type(self).__name__} "
                        f"attempt={attempt_no}/{self.max_attempts} phase=parse "
                        f"error={type(repair_parse_error).__name__}: {str(repair_parse_error)[:300]}",
                        flush=True,
                    )
                    last_obj = self.normalize_retry_obj({"_raw": repair_parse_error.raw_text})
                    last_error = repair_parse_error
                except Exception as repair_error:
                    if repaired_obj is None:
                        print(
                            f"[synthesize] step flow-json-repair: FAIL  flow={type(self).__name__} "
                            f"attempt={attempt_no}/{self.max_attempts} phase=call "
                            f"error={type(repair_error).__name__}: {str(repair_error)[:300]}",
                            flush=True,
                        )
                        last_obj = self.normalize_retry_obj({"_raw": e.raw_text})
                        last_error = repair_error
                        continue
                    print(
                        f"[synthesize] step flow-json-repair-validate: FAIL  flow={type(self).__name__} "
                        f"attempt={attempt_no}/{self.max_attempts} "
                        f"error={type(repair_error).__name__}: {str(repair_error)[:300]}",
                        flush=True,
                    )
                    # repair already produced parseable JSON, but host validation did not pass;
                    # in this case entering the regular schema retry is still meaningful.
                    last_obj = self.normalize_retry_obj(repaired_obj)
                    last_error = repair_error
            except Exception as e:
                print(
                    f"[synthesize] step flow-loop: FAIL  flow={type(self).__name__} "
                    f"attempt={attempt_no}/{self.max_attempts} "
                    f"error={type(e).__name__}: {str(e)[:300]}",
                    flush=True,
                )
                last_obj = self.normalize_retry_obj(raw_obj)
                last_error = e
        assert last_error is not None
        print(
            f"[synthesize] step flow-loop: exhausted  flow={type(self).__name__} "
            f"attempts={self.max_attempts} final_error={type(last_error).__name__}: {str(last_error)[:500]}",
            flush=True,
        )
        return None, f"{type(last_error).__name__}: {last_error}"
