#!/usr/bin/env python3
"""PreToolUse hook: blocks Write/Edit/NotebookEdit under immutable raw-data paths.

Enforces the "raw data no modificado" rule from the project spec by denying
the tool call before it runs, rather than relying on memory/discipline.
Reads the Claude Code PreToolUse JSON payload from stdin and, if the target
path falls under a protected prefix, prints a permissionDecision=deny
response. Prints nothing (allows) otherwise. Fails open on malformed input.
"""
import json
import sys

# One prefix per protected directory, relative to the repo root, POSIX-style.
PROTECTED_PREFIXES = [
    "data/raw",
    "data/external",
]


def _normalized(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _is_protected(path: str):
    rel = _normalized(path)
    for prefix in PROTECTED_PREFIXES:
        if rel == prefix or rel.startswith(prefix + "/"):
            return prefix
    return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # malformed input: fail open, never block on a parse error

    tool_input = payload.get("tool_input", {}) or {}

    candidate_paths = []
    for key in ("file_path", "notebook_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str):
            candidate_paths.append(value)

    for path in candidate_paths:
        prefix = _is_protected(path)
        if prefix:
            result = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Bloqueado por regla de proyecto: '{path}' esta bajo "
                        f"'{prefix}/', una carpeta de datos crudos inmutables. "
                        "Si el cambio es realmente necesario, hazlo fuera de "
                        "Claude Code y documenta el motivo en el README/ADR."
                    ),
                }
            }
            print(json.dumps(result))
            sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
