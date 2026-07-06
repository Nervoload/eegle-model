#!/usr/bin/env python
"""Run an EEG workbench experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eegworkbench.config import apply_cli_overrides, load_config
from eegworkbench.training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eegnet_bnci2014_001.yml")
    parser.add_argument("--dataset", help="Override the MOABB dataset class name.")
    parser.add_argument("--model", help="Override model name, e.g. eegnet, shallow, riemann_tangent.")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to load.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", help="auto, cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic data and a tiny run.")
    parser.add_argument("--bendr-repo", help="Set BENDR_REPO for BENDR configs.")
    parser.add_argument("--bendr-encoder", help="Set BENDR_ENCODER_WEIGHTS.")
    parser.add_argument("--bendr-context", help="Set BENDR_CONTEXT_WEIGHTS.")
    args = parser.parse_args()

    _set_env("BENDR_REPO", args.bendr_repo)
    _set_env("BENDR_ENCODER_WEIGHTS", args.bendr_encoder)
    _set_env("BENDR_CONTEXT_WEIGHTS", args.bendr_context)

    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        dataset=args.dataset,
        model=args.model,
        subjects=args.subjects,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _set_env(name: str, value: str | None) -> None:
    if value:
        os.environ[name] = value


if __name__ == "__main__":
    main()
