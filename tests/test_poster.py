"""Smoke tests for ``scripts/build_poster.py``.

These tests run the actual build script and assert that the resulting
``poster.pptx`` exists, is a valid OOXML archive, and is byte-identical
when the build is run twice in a row. They do NOT inspect the rendered
preview PNG, since that path delegates to ``soffice`` (LibreOffice) which
isn't guaranteed to be available in every CI image.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_SCRIPT = os.path.join(REPO_ROOT, "scripts", "build_poster.py")
POSTER_PPTX = os.path.join(REPO_ROOT, "poster", "poster.pptx")


def _run_build():
    """Invoke ``scripts/build_poster.py`` as a subprocess from the repo root."""
    result = subprocess.run(
        [sys.executable, BUILD_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"build_poster.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def test_build_poster_produces_file():
    """First-pass build produces a non-empty, well-formed .pptx archive."""
    # Clean any pre-existing artifact so the test sees the script writing it.
    if os.path.exists(POSTER_PPTX):
        os.remove(POSTER_PPTX)
    _run_build()
    assert os.path.exists(POSTER_PPTX), "poster.pptx was not produced"
    size = os.path.getsize(POSTER_PPTX)
    assert size > 50_000, f"poster.pptx looks suspiciously small: {size} bytes"
    # A valid pptx is a ZIP with a Content_Types.xml entry at the root.
    with zipfile.ZipFile(POSTER_PPTX) as zf:
        names = set(zf.namelist())
    assert "[Content_Types].xml" in names, "missing [Content_Types].xml entry"
    assert any(n.startswith("ppt/slides/") for n in names), (
        "no slide xml entries found in poster.pptx"
    )


def test_build_poster_idempotent(tmp_path):
    """Running the build twice in a row produces a byte-identical .pptx."""
    _run_build()
    first_sha = _sha256(POSTER_PPTX)
    snapshot = tmp_path / "poster_run1.pptx"
    shutil.copyfile(POSTER_PPTX, snapshot)

    _run_build()
    second_sha = _sha256(POSTER_PPTX)

    assert first_sha == second_sha, (
        "poster.pptx hash changed between two consecutive builds; "
        f"first={first_sha} second={second_sha}"
    )
