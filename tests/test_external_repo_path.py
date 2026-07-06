import tempfile
import unittest
from pathlib import Path

from eegworkbench.models.external import _resolve_repo_path, _resolve_repo_paths


class ExternalRepoPathTest(unittest.TestCase):
    def test_accepts_bendr_parent_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bendr = root / "BENDR"
            bendr.mkdir()
            (bendr / "dn3_ext.py").write_text("# test marker\n", encoding="utf-8")

            self.assertEqual(_resolve_repo_path(str(root)), bendr)

    def test_adds_sibling_dn3_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bendr = root / "BENDR"
            dn3 = root / "dn3"
            bendr.mkdir()
            (dn3 / "dn3").mkdir(parents=True)
            (bendr / "dn3_ext.py").write_text("# test marker\n", encoding="utf-8")

            self.assertEqual(_resolve_repo_paths(str(root)), [bendr, dn3])

    def test_accepts_bendr_repo_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bendr = Path(temp_dir)
            (bendr / "dn3_ext.py").write_text("# test marker\n", encoding="utf-8")

            self.assertEqual(_resolve_repo_path(str(bendr)), bendr)


if __name__ == "__main__":
    unittest.main()
