"""CodeExecutionTool for safely running Python snippets and validating outputs."""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Type
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:
        name: str = ""
        description: str = ""
        args_schema: Optional[Type[BaseModel]] = None

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def _run(self, *args, **kwargs):
            raise NotImplementedError


class CodeExecInput(BaseModel):
    """Input schema for CodeExecutionTool."""
    code: str = Field(..., description="Python source code to execute and test.")


class CodeExecutionTool(BaseTool):
    """CrewAI tool for executing Python code in an isolated subprocess."""
    name: str = "code_execution_tool"
    description: str = (
        "Execute Python code snippets in a subprocess to test functions, "
        "verify syntax, or inspect output. Returns stdout, stderr, and return code."
    )
    args_schema: Type[BaseModel] = CodeExecInput

    timeout_seconds: int = 10

    def _run(self, code: str) -> str:
        """Run code in isolated temporary file and capture results."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", encoding="utf-8", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(code)

        try:
            res = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            stdout = res.stdout.strip()
            stderr = res.stderr.strip()

            output = [f"Return Code: {res.returncode}"]
            if stdout:
                output.append(f"STDOUT:\n{stdout}")
            if stderr:
                output.append(f"STDERR:\n{stderr}")
            if not stdout and not stderr:
                output.append("(Execution produced no output)")

            return "\n".join(output)
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {self.timeout_seconds} seconds."
        except Exception as e:
            return f"Execution failed: {str(e)}"
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass


class FileWriteInput(BaseModel):
    """Input schema for FileWriteTool."""
    file_path: str = Field(..., description="Relative or absolute path of the file to write.")
    content: str = Field(..., description="Text or code content to write to the file.")


class FileWriteTool(BaseTool):
    """CrewAI tool for writing code and documentation files to disk."""
    name: str = "file_write_tool"
    description: str = "Write code, tests, or documentation files to local storage."
    args_schema: Type[BaseModel] = FileWriteInput

    def _run(self, file_path: str, content: str) -> str:
        """Write content to file."""
        try:
            target = Path(file_path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File successfully written to: {target}"
        except Exception as e:
            return f"Failed to write file {file_path}: {str(e)}"
