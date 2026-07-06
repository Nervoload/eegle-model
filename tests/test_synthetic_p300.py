import unittest

from eegworkbench.config import DatasetConfig
from eegworkbench.data import load_eeg_data


class SyntheticP300Test(unittest.TestCase):
    def test_synthetic_p300_bundle_shape_and_labels(self):
        bundle = load_eeg_data(
            DatasetConfig(
                source="synthetic",
                paradigm="p300",
                synthetic_trials=32,
                synthetic_channels=8,
                synthetic_times=128,
                synthetic_classes=2,
            )
        )

        self.assertEqual(bundle.X.shape, (32, 8, 128))
        self.assertEqual(bundle.label_names, ["NonTarget", "Target"])
        self.assertEqual(bundle.n_outputs, 2)
        self.assertGreater(bundle.y.sum(), 0)


if __name__ == "__main__":
    unittest.main()
