import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_generation.ingest import ingest_paths, notebook_to_text


class TestIngestNotebook(unittest.TestCase):
    def _write_notebook(self, payload: dict) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "sample.ipynb"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_notebook_strips_outputs(self) -> None:
        payload = {
            "metadata": {"language_info": {"name": "python"}},
            "cells": [
                {"cell_type": "markdown", "source": "# Title"},
                {
                    "cell_type": "code",
                    "source": "print('hello')\n",
                    "outputs": [{"output_type": "stream", "text": ["OUTPUT_ONLY\n"]}],
                },
            ],
        }
        path = self._write_notebook(payload)
        text = notebook_to_text(path)
        self.assertIn("print('hello')", text)
        self.assertNotIn("OUTPUT_ONLY", text)
        self.assertNotIn("# Title", text)

    def test_notebook_non_python_rejected(self) -> None:
        payload = {
            "metadata": {"language_info": {"name": "r"}},
            "cells": [],
        }
        path = self._write_notebook(payload)
        with self.assertRaises(ValueError):
            notebook_to_text(path)


class TestIngestPaths(unittest.TestCase):
    def test_plain_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "script.py"
            path.write_text("print('ok')\n", encoding="utf-8")
            text = ingest_paths([path])
            self.assertIn("# File: script.py", text)
            self.assertIn("print('ok')", text)
