#!/usr/bin/env python
"""Run an EEG workbench experiment suite."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eegworkbench.suites import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="configs/suites/p300_eegnet_windows.yml")
    parser.add_argument("--subjects", nargs="+", type=int, help="Override all experiment subjects.")
    parser.add_argument("--epochs", type=int, help="Override all experiment epochs.")
    parser.add_argument("--batch-size", type=int, help="Override all experiment batch sizes.")
    parser.add_argument("--device", help="auto, cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--output-dir", help="Override suite output directory.")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bendr-repo", help="Set BENDR_REPO for BENDR suite configs.")
    parser.add_argument("--bendr-encoder", help="Set BENDR_ENCODER_WEIGHTS.")
    parser.add_argument("--bendr-context", help="Set BENDR_CONTEXT_WEIGHTS.")
    args = parser.parse_args()

    _set_env("BENDR_REPO", args.bendr_repo)
    _set_env("BENDR_ENCODER_WEIGHTS", args.bendr_encoder)
    _set_env("BENDR_CONTEXT_WEIGHTS", args.bendr_context)

    summary = run_suite(
        args.suite,
        subjects=args.subjects,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        include_disabled=args.include_disabled,
        continue_on_error=not args.stop_on_error,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _set_env(name: str, value: str | None) -> None:
    if value:
        os.environ[name] = value


if __name__ == "__main__":
    main()
