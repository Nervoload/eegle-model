import unittest
from pathlib import Path

from eegworkbench.attention8 import (
    ATTENTION8_CHANNELS,
    Attention8BridgeError,
    Attention8ExportOptions,
    attention8_input_contract,
    attention8_model_config,
    kind_for_profile,
    normalize_profile,
)


class Attention8BridgeTest(unittest.TestCase):
    def test_profile_aliases_resolve_to_closedloop_kinds(self):
        self.assertEqual(normalize_profile("foundation_head_logreg"), "lora")
        self.assertEqual(kind_for_profile("lora"), "foundation_head_logreg")
        self.assertEqual(kind_for_profile("film"), "foundation_prototype")
        self.assertEqual(kind_for_profile("eegnet"), "torch_eegnet")
        self.assertEqual(kind_for_profile("bendr"), "foundation_bendr")

    def test_unknown_profile_is_actionable(self):
        with self.assertRaises(Attention8BridgeError) as context:
            normalize_profile("not-a-model")

        self.assertIn("unknown attention8 model profile", str(context.exception))

    def test_input_contract_matches_attention8_montage(self):
        contract = attention8_input_contract(sample_rate_hz=500.0, sample_count=1000)

        self.assertEqual(contract["channel_order"], list(ATTENTION8_CHANNELS))
        self.assertEqual(contract["required_channels"], list(ATTENTION8_CHANNELS))
        self.assertEqual(contract["epoch_window_seconds"], [-2.0, 0.0])
        self.assertEqual(contract["sample_rate_hz"], 500.0)
        self.assertEqual(contract["sample_count"], 1000)
        self.assertEqual(contract["tensor_layout"], "batch_1_channels_samples")

    def test_lora_config_is_closedloop_trainable(self):
        options = Attention8ExportOptions(
            session_dirs=(Path("/tmp/session-a"),),
            output_dir=Path("/tmp/models"),
            support_trials=25,
        )
        config = attention8_model_config("lora", options)

        self.assertEqual(config["kind"], "foundation_head_logreg")
        self.assertEqual(config["target"], "attention_lapse_binary")
        self.assertEqual(config["support_trials"], 25)
        self.assertEqual(config["calibration"]["support_trials"], 25)
        self.assertEqual(config["session_dirs"], [str(Path("/tmp/session-a").resolve())])
        self.assertEqual(config["input_layout"], "samples_x_channels")
        self.assertEqual(config["input_contract"]["channel_order"], list(ATTENTION8_CHANNELS))
        self.assertEqual(config["embedding"]["profile"], "lora")


if __name__ == "__main__":
    unittest.main()
