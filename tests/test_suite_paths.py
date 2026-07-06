import os
import unittest
from pathlib import Path

from eegworkbench.suites import _missing_required_paths, _resolve_path


class SuitePathTest(unittest.TestCase):
    def test_resolve_path_from_nested_suite_to_repo_root(self):
        resolved = _resolve_path(
            Path("configs/suites/p300_bendr_compare_windows.yml"),
            "configs/bendr_p300_bnci2014_008.yml",
        )

        self.assertTrue(str(resolved).endswith("configs/bendr_p300_bnci2014_008.yml"))

    def test_missing_required_paths_expands_environment(self):
        os.environ["EEGWORKBENCH_TEST_PATH"] = str(Path.cwd())
        self.assertEqual(_missing_required_paths(["${EEGWORKBENCH_TEST_PATH}"]), [])
        self.assertEqual(_missing_required_paths(["${EEGWORKBENCH_DOES_NOT_EXIST}"]), ["${EEGWORKBENCH_DOES_NOT_EXIST}"])


if __name__ == "__main__":
    unittest.main()
