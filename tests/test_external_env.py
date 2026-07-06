import unittest

from eegworkbench.models.external import ExternalModelError, _raise_if_unresolved_env_vars


class ExternalEnvTest(unittest.TestCase):
    def test_unresolved_env_var_is_actionable(self):
        with self.assertRaises(ExternalModelError) as context:
            _raise_if_unresolved_env_vars("${BENDR_REPO}", "model.external.repo_path")

        self.assertIn("--bendr-repo", str(context.exception))


if __name__ == "__main__":
    unittest.main()
