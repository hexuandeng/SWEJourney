"""
infra_failure_classify.py
=========================

Language-agnostic classifier for *infrastructure* (non-test) failures in a
test-run's raw output, used by the user-simulator tooling (live_subtask_test_tool).

Why this lives in user_simulation/ (not trace_synth/):
  This is purely a user-sim concern — it turns a raw runner failure into a
  structured verdict the simulated user can ground its feedback on. It must NOT
  touch the synthesize/ATTEMPT execution path in trace_synth/local_test_executor.py.
  Everything needed is self-contained here (including a private copy of the
  build-failure sniff), so the trace-synthesis pipeline is left untouched.

The repos under test span Go / Python / JS / TS (pytest / go test / jest /
mocha), so the classifier is table-driven and does NOT special-case any one
toolchain.

Output of classify_infra_failure(output) -> dict | None:

    {
      "kind": "MISSING_DEPENDENCY" | "COMPILE_ERROR" | "SUITE_CRASH",
      "language": "go" | "python" | "js" | "ts" | "",
      "package": "<failing module/package, best-effort>",
      "actionable": "<concrete fix command/instruction, or ''>",
      "causes": ["<plain-language candidate cause>", ...],
      "is_test_layer": bool,   # True -> cites a hidden-test symbol; user-sim MUST paraphrase
      "evidence": "<short root-cause excerpt>",
    }

`is_test_layer` is the privacy switch: True means the failure cites a symbol
that lives ONLY in the hidden test patch (the agent cannot see it; the user-sim
must paraphrase the symptom). False means the failure is build/dependency/
runtime info equally visible inside the agent's own container, so the user-sim
MAY relay a concrete, actionable hint.

Returns None when the output shows no infra-level failure (i.e. it's an ordinary
assertion failure, or the suite passed).
"""

from __future__ import annotations

import re


# --- file-name heuristics (language-agnostic) -------------------------------
_TESTFILE_PATH_RE = re.compile(
    r"\b([\w./-]*?(?:_test\.go"           # Go:      foo_test.go
    r"|[._-]test\.[jt]sx?"                 # JS/TS:   foo.test.ts / foo-test.tsx / foo_test.js
    r"|[._-]spec\.[jt]sx?"                 # JS/TS:   foo.spec.js / foo-spec.ts
    r"|__tests__/[\w./-]+\.[jt]sx?"        # JS/TS:   __tests__/foo.tsx
    r"|test_[\w.-]+\.py"                   # Python:  test_foo.py
    r"|[\w.-]*_test\.py"                   # Python:  foo_test.py
    r"|tests?/[\w./-]+\.py))"              # Python:  tests/foo.py
    , re.IGNORECASE,
)


def _cites_test_file(line: str) -> bool:
    return bool(_TESTFILE_PATH_RE.search(line or ""))


# Symbols that signal "X is referenced but not defined" across languages. When
# such a symbol error is tied to a TEST file, the missing name lives in the
# hidden test patch and the agent cannot see it (is_test_layer=True).
_MISSING_SYMBOL_RE = re.compile(
    r"undefined"                                   # Go: X undefined
    r"|has no field or method"                     # Go: type has no field/method X
    r"|is not defined"                             # JS: X is not defined
    r"|has no attribute"                           # Python: 'T' object has no attribute 'x'
    r"|name '[^']+' is not defined"                # Python: NameError
    r"|cannot find name"                           # TS: cannot find name 'X'
    r"|is not exported"                            # TS/Go: X is not exported
    , re.IGNORECASE,
)


def _is_test_layer_symbol(text: str, cause_line: str) -> bool:
    """True iff the failure is a missing symbol that ONLY the hidden test names.

    Language-agnostic. The privacy risk is: the *test* references a symbol the
    production code does not define yet (e.g. ``cache_test.go: setJSON
    undefined``, ``test_x.py: 'Store' object has no attribute 'set_json'``).
    Relaying that raw error would leak a hidden-test symbol the agent cannot
    see. We treat it as test-layer when BOTH hold:
      - a "missing symbol / undefined name / no attribute" diagnostic appears,
      - and it is attributed to a TEST file (same line, or — for multi-line
        tracebacks like Python — anywhere in the output while no production
        file carries the same diagnostic).
    Everything else (syntax error, type mismatch, production-file symbol error)
    is NOT test-layer.
    """
    # Direct: the diagnostic line itself cites a test file AND a missing symbol.
    if cause_line and _cites_test_file(cause_line) and _MISSING_SYMBOL_RE.search(cause_line):
        return True
    # Multi-line (e.g. Python traceback): the missing-symbol diagnostic and the
    # test-file frame are on different lines. Require a missing-symbol error to
    # exist, AND the cited source frames to be test-only.
    if not _MISSING_SYMBOL_RE.search(text or ""):
        return False
    cited_files = re.findall(r"([\w./-]+\.(?:py|go|[jt]sx?))", text or "")
    if not cited_files:
        return False
    # If ANY non-test source file is cited, treat as production code -> not
    # test-layer. Only when exclusively test files are cited is it test-layer.
    has_test = any(_cites_test_file(f) for f in cited_files)
    has_nontest = any(not _cites_test_file(f) for f in cited_files)
    return has_test and not has_nontest


# --- missing-dependency patterns: (regex, install-cmd builder) --------------
# Each builder takes the regex match + full output and returns
# (package_name, install_cmd). An empty install_cmd means "this is a missing
# SOURCE FILE (relative import), not an installable dependency".
def _go_install_cmd(pkg: str, out: str) -> str:
    m = re.search(r"to add it:\s*\n?\s*(go get [^\s\\\"]+)", out or "", re.I)
    base = m.group(1).strip() if m else f"go get {pkg}"
    return base + " && go mod tidy"


def _npm_install_cmd(mod: str) -> str:
    mod = mod.strip()
    # relative imports are NOT installable deps — they're missing source files
    if mod.startswith(".") or mod.startswith("/"):
        return ""
    if mod.startswith("@"):
        pkg = "/".join(mod.split("/")[:2])  # keep scoped '@scope/pkg'
    else:
        pkg = mod.split("/")[0]             # strip subpath 'lodash/merge' -> 'lodash'
    return f"npm install {pkg}"


_MISSING_DEP_PATTERNS = [
    # Go: "no required module provides package X; to add it: go get X"
    (re.compile(r"no required module provides package\s+([^\s;]+)", re.I),
     lambda m, out: (m.group(1), _go_install_cmd(m.group(1), out))),
    # Python: "ModuleNotFoundError: No module named 'x'" / "ImportError: No module named x"
    (re.compile(r"(?:ModuleNotFoundError|ImportError):\s*No module named\s*['\"]?([\w.]+)", re.I),
     lambda m, out: (m.group(1), f"pip install {m.group(1).split('.')[0]}")),
    # JS/TS (node): "Cannot find module 'x'"
    (re.compile(r"Cannot find module\s*['\"]([^'\"]+)['\"]", re.I),
     lambda m, out: (m.group(1), _npm_install_cmd(m.group(1)))),
    # TS (compiler): "error TS2307: Cannot find module 'x' ..."
    (re.compile(r"error TS2307:\s*Cannot find module\s*['\"]([^'\"]+)", re.I),
     lambda m, out: (m.group(1), _npm_install_cmd(m.group(1)))),
    # JS/TS bundler: "Can't resolve 'x'"
    (re.compile(r"Can't resolve\s*['\"]([^'\"]+)['\"]", re.I),
     lambda m, out: (m.group(1), _npm_install_cmd(m.group(1)))),
]


# --- generic compile/build-error markers (any language) ---------------------
_COMPILE_MARKERS = (
    # Go
    "[build failed]", "has no field or method", "undefined:", "undefined (type ",
    "not enough arguments", "too many arguments",
    # Python (collection/import-time — a test file failing to import counts)
    "syntaxerror", "indentationerror", "nameerror", "attributeerror",
    # JS/TS (parse-time)
    "referenceerror:", "is not defined",
    # generic
    "compilation error", "compilation failed", "cannot find symbol",
    "build failed", "cannot use ",
)

_COMPILE_CAUSE_RE = re.compile(
    r"(undefined|has no field|cannot use|not enough arguments|too many arguments"
    r"|syntaxerror|indentationerror|nameerror|attributeerror|referenceerror"
    r"|is not defined|error TS\d{4}|cannot find symbol)", re.I
)

# --- runtime crash markers (built & started, then aborted) ------------------
_PANIC_MARKERS = (
    "panic:",              # Go
    "fatal error:",        # Go runtime
    "uncaught exception",  # JS/mocha
    "unhandledpromiserejection",  # JS
    "segmentation fault",  # native
)


def _guess_language(text: str) -> str:
    low = (text or "").lower()
    if ("go.mod" in low or re.search(r"\.go[:\s]", low) or "no required module" in low
            or "goroutine " in low or re.search(r"\bgo\.[\w.]+\.\w+/", low)):
        return "go"
    if "error ts" in low or re.search(r"\.tsx?[:\s]", low):
        return "ts"
    if (re.search(r"\.jsx?[:\s]", low) or "node_modules" in low or "npm" in low
            or "_resolvefilename" in low or "uncaught exception" in low):
        return "js"
    if ("traceback (most recent call last)" in low or re.search(r"\.py[:\"]", low)
            or "modulenotfounderror" in low or "attributeerror" in low):
        return "python"
    return ""


def _failing_unit(text: str) -> str:
    """Best-effort name of the failing package/module/file across languages."""
    # Go: "FAIL\t<pkg>" or "# <pkg>"
    m = re.search(r"^(?:FAIL|#)\s+(\S+)", text or "", re.MULTILINE)
    if m:
        return m.group(1).strip()
    # JS/TS/Python: first cited source file in a diagnostic line
    m = re.search(r"([\w./-]+\.(?:py|[jt]sx?|go))[:\s]", text or "")
    return m.group(1).strip() if m else ""


def _looks_like_compile_failure(output: str) -> bool:
    text = output or ""
    low = text.lower()
    if not low.strip():
        return False
    if any(mk in low for mk in _COMPILE_MARKERS):
        return True
    return re.search(r"\berror TS\d{4}:", text) is not None


def _looks_like_suite_crash(output: str) -> bool:
    low = (output or "").lower()
    if not low.strip():
        return False
    return any(mk in low for mk in _PANIC_MARKERS)


def classify_infra_failure(output: str) -> dict | None:
    """Classify a non-test (infra) failure from raw runner output.

    Ordered most-specific-first: a missing dependency (we can name the exact
    install command) beats a generic compile error, which beats a runtime crash.
    Returns None when no infra-level failure is present.
    """
    text = output or ""
    if not text.strip():
        return None

    lang = _guess_language(text)

    # 1) Missing dependency — recognizable per-language signature; we can
    #    synthesize the exact install command.
    for rx, builder in _MISSING_DEP_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        pkg, cmd = builder(m, text)
        # A relative-path "module not found" ('./x', '../x') is a missing SOURCE
        # FILE, not an installable dep.
        if cmd == "":
            return {
                "kind": "COMPILE_ERROR",
                "language": lang,
                "package": _failing_unit(text),
                "actionable": (
                    f"the code imports '{pkg}', but that source file/module does "
                    f"not exist in the tree; create it or fix the import path, then re-run"
                ),
                "causes": [
                    f"an import of '{pkg}' resolves to a file that is not present "
                    f"— it was never created, or the path/name is wrong",
                ],
                "is_test_layer": False,
                "evidence": (m.group(0) or "").strip()[:240],
            }
        return {
            "kind": "MISSING_DEPENDENCY",
            "language": lang,
            "package": pkg,
            "actionable": cmd,
            "causes": [
                f"the project's dependency manifest does not declare '{pkg}', "
                f"so the package under test cannot resolve it at build time",
            ],
            "is_test_layer": False,  # manifests (go.mod/package.json/requirements) are agent-owned
            "evidence": (m.group(0) or "").strip()[:240],
        }

    # 2) Runtime crash — built & started, then aborted (panic / uncaught / fatal).
    #    Checked BEFORE generic compile so a runtime "TypeError"/"Uncaught" is
    #    not misread as a parse error.
    if _looks_like_suite_crash(text):
        evidence = ""
        for line in text.splitlines():
            low = line.strip().lower()
            if any(low.startswith(mk) or mk in low for mk in _PANIC_MARKERS):
                evidence = line.strip()[:300]
                break
        return {
            "kind": "SUITE_CRASH",
            "language": lang,
            "package": _failing_unit(text),
            "actionable": (
                "the code crashes at runtime (panic/uncaught exception) and "
                "aborts the rest of the suite; fix the crashing code path, then re-run"
            ),
            "causes": [
                "a nil/undefined value or unexpected mock call is hit on a code "
                "path the new code introduced",
                "the crash kills the whole test process, so any later checks in "
                "the same module never get to run",
            ],
            "is_test_layer": False,
            "evidence": evidence or "the test process crashed at runtime",
        }

    # 3) Compile / build / collection failure (no installable dep, not a crash).
    if _looks_like_compile_failure(text):
        cause_line = ""
        mm = _COMPILE_CAUSE_RE.search(text)
        if mm:
            for line in text.splitlines():
                if mm.group(0).lower() in line.lower() and "build failed" not in line.lower():
                    cause_line = line.strip()[:300]
                    break
        is_test_layer = _is_test_layer_symbol(text, cause_line)
        return {
            "kind": "COMPILE_ERROR",
            "language": lang,
            "package": _failing_unit(text),
            "actionable": (
                "" if is_test_layer
                else "the code under test does not compile/parse; fix the "
                     "reported symbol/type error in the production source, then re-run"
            ),
            "causes": (
                ["the test references a symbol/name that the production code "
                 "does not yet define — the implementation is incomplete"]
                if is_test_layer else
                ["a syntax/type/symbol error in the production code prevents the "
                 "module from building",
                 "a renamed/removed public symbol broke another file that imports it"]
            ),
            "is_test_layer": is_test_layer,
            "evidence": cause_line or "code under test failed to compile/parse",
        }

    return None
