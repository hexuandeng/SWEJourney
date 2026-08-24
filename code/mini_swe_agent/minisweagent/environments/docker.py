import logging
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge

# Optional on-demand image loader, supplied via the SWEJOURNEY_ENSURE_IMAGE env var (hit -> use; miss -> from
# a local .tar via `docker load`; never pull from the network). That script is not on sys.path, so use
# importlib to lazily load it once by absolute path, cached to the module-level _ensure_image_fn. On load
# failure, degrade to a no-op (preserving the original behavior).
_ENSURE_IMAGE_PATH = os.environ.get("SWEJOURNEY_ENSURE_IMAGE", "")
_ensure_image_fn = None
_ensure_image_loaded = False


def _ensure_image_local(image_ref: str, logger: logging.Logger) -> None:
    """Ensure the image is available locally before starting the container (loaded on demand from the CFS tar, never pulled). On failure, only warn, do not block."""
    global _ensure_image_fn, _ensure_image_loaded
    if not _ENSURE_IMAGE_PATH:
        return
    if not _ensure_image_loaded:
        _ensure_image_loaded = True
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_swejourney_ensure_image", _ENSURE_IMAGE_PATH)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _ensure_image_fn = mod.ensure_image
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ensure_image] Unable to load the on-demand loader ({e}), skipping local tar preloading")
            _ensure_image_fn = None
    if _ensure_image_fn is None:
        return
    try:
        ok = _ensure_image_fn(image_ref, verbose=False)
        if not ok:
            logger.warning(f"[ensure_image] No local image and no matching tar: {image_ref} (docker run will handle it itself)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ensure_image] Exception preloading {image_ref}: {e} (continuing to try docker run)")


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = ["--rm"]
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """
    attach_container_id: str = ""
    """If set, ATTACH to this already-running container instead of starting a new one.
    Used by resume/checkpoint-resume: the container was started by a previous `docker run -d`
    process that has since been killed; its filesystem state (the agent's edits) and
    remaining wall-clock are inherited. In attach mode the container is NOT owned by
    this object — cleanup()/__del__ will never stop or remove it.
    """


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        # _owns_container: True only when this object started the container itself via `docker run`. In attach
        # mode (resume borrowing an already-alive container) it is False — cleanup()/__del__ will never
        # stop/rm it, preventing the accidental deletion of a live container owned by another process and still in use.
        self._owns_container = not self.config.attach_container_id
        if self.config.attach_container_id:
            self._attach_container()
        else:
            self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        _ensure_image_local(self.config.image, self.logger)
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        # Must use --entrypoint sleep to override the image's built-in ENTRYPOINT. SWE-bench Pro's sweap-images
        # images have ENTRYPOINT=[/bin/bash]; without the override, `<image> sleep 48h` gets parsed as
        # `/bin/bash /usr/bin/sleep 48h` → bash tries to run the sleep binary as a script →
        # "cannot execute binary file" → ExitCode=126, the container exits instantly → --rm removes it immediately →
        # all the agent's subsequent docker exec calls get "No such container" and spin until step_limit.
        # (trace_synth/swebench_dockerlib.py likewise uses --entrypoint sleep ... infinity and avoids this pitfall.)
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            "--entrypoint",
            "sleep",
            *self.config.run_args,
            self.config.image,
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def _attach_container(self):
        """Attach to an already-running container (resume/checkpoint-resume) instead of starting one.

        Validates that the container exists and is running; raises a clear error
        otherwise so the caller can fall back to a fresh run (delete-and-rerun).
        Does NOT take ownership: cleanup()/__del__ will never stop/rm it.
        """
        cid = self.config.attach_container_id
        try:
            result = subprocess.run(
                [self.config.executable, "inspect", "-f", "{{.State.Running}}", cid],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot attach to container {cid!r}: docker inspect failed ({e}). "
                "The container may have been deleted/stopped; cannot resume."
            ) from e
        if result.stdout.strip() != "true":
            raise RuntimeError(
                f"Cannot attach to container {cid!r}: State.Running={result.stdout.strip()!r} "
                "(the container is not running; cannot resume)."
            )
        self.container_id = cid
        self.logger.info(f"Attached to existing running container {cid}")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        """Stop and remove the Docker container.

        No-op in attach mode (``_owns_container=False``): a borrowed/resumed
        container is owned by another process and must never be stopped/removed here.
        """
        if not getattr(self, "_owns_container", True):
            return
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
