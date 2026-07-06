#!/usr/bin/env python
"""Diagnose whether BENDR/DN3 can be imported by the workbench."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eegworkbench.adapters.bendr import _patch_collections_abc_aliases
from eegworkbench.models.external import _resolve_repo_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bendr-repo", default=os.environ.get("BENDR_REPO"))
    args = parser.parse_args()

    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"cwd: {Path.cwd()}")
    print(f"bendr_repo_arg: {args.bendr_repo}")

    if not args.bendr_repo:
        print("ERROR: pass --bendr-repo or set BENDR_REPO")
        return 2

    resolved_paths = _resolve_repo_paths(args.bendr_repo)
    print("resolved_repo_paths:")
    for path in resolved_paths:
        print(f"  - {path.resolve()} exists={path.exists()}")

    for path in reversed(resolved_paths):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    _patch_collections_abc_aliases()
    _print_package_version("pyyaml-include")
    _print_package_version("PyYAML")
    _print_spec("yamlinclude")
    _print_spec("dn3")
    _print_spec("dn3_ext")

    try:
        importlib.import_module("dn3_ext")
    except Exception:
        print("dn3_ext import: FAILED")
        traceback.print_exc()
        return 1

    print("dn3_ext import: OK")
    return 0


def _print_spec(module_name: str) -> None:
    spec = importlib.util.find_spec(module_name)
    print(f"{module_name}_spec: {spec.origin if spec else None}")


def _print_package_version(package_name: str) -> None:
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    print(f"{package_name}_version: {version}")


if __name__ == "__main__":
    raise SystemExit(main())
