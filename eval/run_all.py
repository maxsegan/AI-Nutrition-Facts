#!/usr/bin/env python3
"""Run the complete AI Nutrition Facts evaluation battery.

Usage:
    python -m eval.run_all                    # all evaluations, all targets
    python -m eval.run_all --targets gpt-4o   # single target
    python -m eval.run_all --skip-political   # skip political evaluation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from eval.config import DEFAULT_TARGETS, RESULTS_DIR


async def main():
    parser = argparse.ArgumentParser(description="Run complete AI Nutrition Facts evaluation")
    parser.add_argument("--targets", nargs="+",
                        help="Target model names to evaluate (default: all)")
    parser.add_argument("--skip-sycophancy", action="store_true")
    parser.add_argument("--skip-political", action="store_true")
    parser.add_argument("--skip-supplementary", action="store_true")
    args = parser.parse_args()

    target_names = args.targets
    total_start = time.time()

    # Run sycophancy
    if not args.skip_sycophancy:
        print("\n" + "=" * 70)
        print("PHASE 1: SYCOPHANCY EVALUATION")
        print("=" * 70)
        from eval.run_sycophancy import main as syco_main
        sys.argv = ["run_sycophancy"]
        if target_names:
            sys.argv += ["--targets"] + target_names
        await syco_main()

    # Run political
    if not args.skip_political:
        print("\n" + "=" * 70)
        print("PHASE 2: POLITICAL & GEOPOLITICAL EVALUATION")
        print("=" * 70)
        from eval.run_political import main as pol_main
        sys.argv = ["run_political"]
        if target_names:
            sys.argv += ["--targets"] + target_names
        await pol_main()

    # Run supplementary
    if not args.skip_supplementary:
        print("\n" + "=" * 70)
        print("PHASE 3: SUPPLEMENTARY METRICS")
        print("=" * 70)
        from eval.run_supplementary import main as supp_main
        sys.argv = ["run_supplementary"]
        if target_names:
            sys.argv += ["--targets"] + target_names
        await supp_main()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}")
    print(f"View results: python -m bench.app")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
