"""
assistant_client.py
====================

Thin HTTP client for the mini-swe-agent assistant server
(minisweagent.run.assistant_server). Zero external deps (urllib).

See USER_SIM_STANDALONE_PLAN.md §1.7 for the protocol.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional


class AssistantClient:
    def __init__(self, base_url: str, timeout: float = 86400.0):  # 24h: a single assistant turn can be very slow
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str, timeout: Optional[float] = None) -> dict:
        req = urllib.request.Request(self.base_url + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- API -----------------------------------------------------------------

    def health(self) -> dict:
        return self._get("/health", timeout=10.0)

    def container(self) -> dict:
        return self._get("/container", timeout=10.0)

    def message(self, text: str, *, first: bool = False,
                big_task_id: Optional[str] = None) -> dict:
        return self._post("/message", {"text": text, "first": first, "big_task_id": big_task_id})

    def reset(self, big_task_id: Optional[str] = None) -> dict:
        return self._post("/reset", {"big_task_id": big_task_id})

    def shutdown(self) -> dict:
        try:
            return self._post("/shutdown", {})
        except Exception:
            return {"ok": False}

    # -- startup helper ------------------------------------------------------

    def wait_ready(self, max_wait: float = 600.0, poll: float = 2.0,
                   expect_instance: Optional[str] = None) -> dict:
        """Poll /health until the server (and its container) is up.

        expect_instance: if given, /health's returned instance_id must match it to be
        considered ready. This is the fallback guard against port crosstalk — if
        base_url happens to connect to another cell's server (a different instance),
        the instance_id will not match -> treated as not ready, keep polling until the
        timeout errors out, and never mistakenly connect to someone else's container
        and pollute the data.
        """
        deadline = time.monotonic() + max_wait
        last_err: Any = None
        while time.monotonic() < deadline:
            try:
                h = self.health()
                if h.get("ok") and h.get("container"):
                    if expect_instance is not None and h.get("instance_id") != expect_instance:
                        last_err = (f"instance mismatch: health={h.get('instance_id')!r} "
                                    f"expected={expect_instance!r} (likely port crosstalk to another cell)")
                    else:
                        return h
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
            time.sleep(poll)
        raise TimeoutError(f"assistant server not ready within {max_wait}s (last={last_err!r})")
