import unittest

from eegworkbench.config import experiment_config_from_mapping, to_plain_data


class ConfigTest(unittest.TestCase):
    def test_config_defaults_and_nested_values(self):
        config = experiment_config_from_mapping(
            {
                "experiment_name": "smoke",
                "dataset": {"source": "synthetic", "synthetic_trials": 12},
                "model": {"name": "eegnet"},
                "training": {"epochs": 1},
            }
        )

        self.assertEqual(config.experiment_name, "smoke")
        self.assertEqual(config.dataset.source, "synthetic")
        self.assertEqual(config.dataset.synthetic_trials, 12)
        self.assertEqual(config.model.name, "eegnet")
        self.assertEqual(config.training.epochs, 1)
        self.assertEqual(to_plain_data(config)["dataset"]["source"], "synthetic")


if __name__ == "__main__":
    unittest.main()
