"""Sandbox utilities for executing lightweight Python code snippets."""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from typing import List

FORBIDDEN_KEYWORDS = {"import", "open", "exec", "eval", "__", "os", "sys", "subprocess"}


@dataclass
class SandboxResult:
    code: str
    success: bool
    stdout: str
    stderr: str
    error: str | None = None


class PythonSandbox:
    """Very small helper around running python code via subprocess with guard rails."""

    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout

    def validate(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                raise ValueError("Imports are not allowed in sandboxed code")
        lowered = code.lower()
        if any(keyword in lowered for keyword in FORBIDDEN_KEYWORDS):
            raise ValueError("Forbidden keyword detected in sandboxed code")

    def run(self, code: str) -> SandboxResult:
        try:
            self.validate(code)
        except Exception as err:
            return SandboxResult(code=code, success=False, stdout="", stderr="", error=str(err))

        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as err:
            return SandboxResult(code=code, success=False, stdout=err.stdout or "", stderr=err.stderr or "", error="timeout")

        success = proc.returncode == 0
        error = None if success else f"exit_code={proc.returncode}"
        return SandboxResult(code=code, success=success, stdout=proc.stdout, stderr=proc.stderr, error=error)


def extract_python_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []
    inside = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if inside:
                blocks.append("\n".join(current).strip())
                current = []
                inside = False
            else:
                inside = True
            continue
        if inside:
            current.append(line)
    if inside and current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]
