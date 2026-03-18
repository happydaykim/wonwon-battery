from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImportSafetyTests(unittest.TestCase):
    def test_validator_module_can_be_imported_directly(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agents.validator import validator_node; print(callable(validator_node))",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIn("True", result.stdout)

    def test_supervisor_module_can_be_imported_directly(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agents.supervisor import supervisor_node; print(callable(supervisor_node))",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIn("True", result.stdout)


if __name__ == "__main__":
    unittest.main()
