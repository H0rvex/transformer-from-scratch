from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_export_onnx_script(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "export_onnx.py"
    subprocess.check_call([sys.executable, str(script), "--out-dir", str(tmp_path / "onnx")], cwd=str(repo))
