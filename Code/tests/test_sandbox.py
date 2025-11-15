"""Tests for sandbox execution."""

from mathlm.rewards import PythonSandbox, extract_python_blocks


def test_extract_python_blocks_parses_fenced_code():
    text = """
Here is code:
```python
x = 2 + 2
print(x)
```
"""
    blocks = extract_python_blocks(text)
    assert len(blocks) == 1
    assert "print(x)" in blocks[0]


def test_sandbox_allows_simple_math():
    sandbox = PythonSandbox(timeout=1.0)
    result = sandbox.run("print(2 + 2)")
    assert result.success
    assert result.stdout.strip() == "4"


def test_sandbox_blocks_imports():
    sandbox = PythonSandbox()
    result = sandbox.run("import os")
    assert not result.success
    assert "not allowed" in result.error.lower()
