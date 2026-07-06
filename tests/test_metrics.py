import unittest

import numpy as np

from eegworkbench.evaluation.metrics import classification_metrics


class MetricsTest(unittest.TestCase):
    def test_binary_metrics_include_erp_score_metrics(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        y_score = np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.6, 0.4],
            ]
        )

        metrics = classification_metrics(
            y_true,
            y_pred,
            label_names=["NonTarget", "Target"],
            y_score=y_score,
        )

        self.assertEqual(metrics["positive_label"], "Target")
        self.assertIn("roc_auc", metrics)
        self.assertIn("average_precision", metrics)
        self.assertAlmostEqual(metrics["positive_prevalence"], 0.5)


if __name__ == "__main__":
    unittest.main()
