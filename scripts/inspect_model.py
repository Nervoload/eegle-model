#!/usr/bin/env python
"""Inspect model shape, parameter count, and a synthetic forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eegworkbench.config import ModelConfig, load_config
from eegworkbench.models import build_model, list_model_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional experiment config to read model settings from.")
    parser.add_argument("--model", default="eegnet", help=f"One of: {', '.join(list_model_names())}")
    parser.add_argument("--n-chans", type=int, default=22)
    parser.add_argument("--n-times", type=int, default=512)
    parser.add_argument("--n-outputs", type=int, default=2)
    args = parser.parse_args()

    model_config = load_config(args.config).model if args.config else ModelConfig(name=args.model)
    if args.model and not args.config:
        model_config.name = args.model

    model = build_model(
        model_config,
        n_chans=args.n_chans,
        n_times=args.n_times,
        n_outputs=args.n_outputs,
    )
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(model)
    print(f"\nparameters: total={total:,} trainable={trainable:,}")

    try:
        import torch

        with torch.no_grad():
            output = model(torch.zeros(2, args.n_chans, args.n_times))
        shape = tuple(output[0].shape if isinstance(output, (tuple, list)) else output.shape)
        print(f"forward output shape: {shape}")
    except Exception as exc:
        print(f"forward pass skipped/failed: {exc}")


if __name__ == "__main__":
    main()

