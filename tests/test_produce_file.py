import importlib.util
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# produce-file.py has a hyphen in its name, so it can't be imported normally.
_spec = importlib.util.spec_from_file_location(
    "produce_file", PROJECT_ROOT / "script" / "produce-file.py"
)
produce_file = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(produce_file)


class TestCorruptData(unittest.TestCase):
    def test_same_seed_produces_identical_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.csv"
            second = Path(tmp) / "second.csv"

            produce_file.corrupt_data(output_path=first, seed=42)
            produce_file.corrupt_data(output_path=second, seed=42)

            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
