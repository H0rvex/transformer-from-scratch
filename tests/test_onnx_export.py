from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_export_onnx_script(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "export_onnx.py"
    python = shutil.which("python3") or shutil.which("python") or sys.executable
    subprocess.check_call([python, str(script), "--out-dir", str(tmp_path / "onnx")], cwd=str(repo))
