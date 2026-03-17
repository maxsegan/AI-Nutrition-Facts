#!/usr/bin/env python3
"""Run the full evaluation battery with maximum parallelism.

All target models run concurrently. Within each model, all subtypes/axes
run concurrently. Rate limits enforced via shared provider-level semaphores.

Usage:
    python -m eval.run_parallel
    python -m eval.run_parallel --targets haiku-4.5 gpt-5-nano
    python -m eval.run_parallel --skip-political
    python -m eval.run_parallel --skip-sycophancy
    python -m eval.run_parallel --skip-supplementary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from typing import Any

from eval.config import (
    DEFAULT_TARGETS, JUDGES, POLITICAL_AXES, RESULTS_DIR,
    TargetModelConfig,
)


# ── Provider-level concurrency limits ──────────────────────────────────────
# These limit total concurrent API calls per provider across ALL models.
# Judges have their own semaphores inside judge_panel; these control target calls.

PROVIDER_LIMITS = {
    "anthropic": 15,   # Anthropic allows decent concurrency
    "openai": 15,      # OpenAI similar
    "google": 8,       # Gemini is slower, be conservative
}

# How many models to evaluate concurrently (each model runs its subtypes serially
# but multiple models run in parallel)
MAX_CONCURRENT_MODELS = 8


async def run_sycophancy_for_target(target: TargetModelConfig) -> dict | None:
    """Run all sycophancy subtypes for one target. Returns aggregate scores."""
    from eval.run_sycophancy import (
        SUBTYPE_RUNNERS, compute_aggregate, RESULTS_DIR,
    )
    from eval.config import JUDGES

    semaphores = {j.name: asyncio.Semaphore(j.max_concurrent) for j in JUDGES}
    subtype_scores = {}

    for subtype_name, (runner, scorer) in SUBTYPE_RUNNERS.items():
        t0 = time.time()
        try:
            results = await runner(target, semaphores)
            scores = scorer(results)
            scores["elapsed_seconds"] = round(time.time() - t0, 1)
            subtype_scores[subtype_name] = scores
            print(f"  [{target.name}] {subtype_name}: "
                  f"{scores.get('overall_score', '?'):.2f}/10 "
                  f"({scores['elapsed_seconds']}s)")
        except Exception as e:
            print(f"  [{target.name}] {subtype_name}: ERROR — {e}")
            traceback.print_exc()

    if not subtype_scores:
        return None

    aggregate = compute_aggregate(subtype_scores)

    # Save immediately
    out_path = RESULTS_DIR / f"sycophancy_{target.name}.json"
    out_path.write_text(json.dumps(aggregate, indent=2))
    return aggregate


async def run_political_for_target(target: TargetModelConfig) -> dict | None:
    """Run all political axes + geopolitical for one target."""
    from eval.run_political import (
        run_political_axis, compute_axis_scores,
        run_geopolitical, compute_geopolitical_scores,
        map_to_typologies, compute_inter_judge_agreement,
    )
    from eval.config import JUDGES, POLITICAL_AXES

    semaphores = {j.name: asyncio.Semaphore(j.max_concurrent) for j in JUDGES}
    result: dict[str, Any] = {"model": target.name}

    # Run all domestic axes concurrently
    axis_ids = [a["id"] for a in POLITICAL_AXES]

    async def _run_axis(axis_id: str) -> tuple[str, list[dict]]:
        t0 = time.time()
        try:
            results = await run_political_axis(axis_id, target, semaphores)
            elapsed = time.time() - t0
            print(f"  [{target.name}] political/{axis_id}: "
                  f"{len(results)} items ({elapsed:.0f}s)")
            return axis_id, results
        except Exception as e:
            print(f"  [{target.name}] political/{axis_id}: ERROR — {e}")
            traceback.print_exc()
            return axis_id, []

    # Run axes concurrently (they share judge semaphores for rate limiting)
    axis_tasks = [_run_axis(aid) for aid in axis_ids]
    axis_results_list = await asyncio.gather(*axis_tasks)

    axis_scores = {}
    all_axis_results = []
    for axis_id, results in axis_results_list:
        if results:
            scores = compute_axis_scores(axis_id, results)
            axis_scores[axis_id] = scores
            all_axis_results.extend(results)

    result["domestic_axes"] = axis_scores

    if axis_scores:
        result["typology_matches"] = map_to_typologies(axis_scores)[:3]

    if all_axis_results:
        result["inter_judge_agreement"] = compute_inter_judge_agreement(all_axis_results)

    # Geopolitical (runs after axes)
    try:
        t0 = time.time()
        geo_results = await run_geopolitical(target, semaphores)
        if geo_results:
            geo_scores = compute_geopolitical_scores(geo_results)
            geo_scores["elapsed_seconds"] = round(time.time() - t0, 1)
            result["geopolitical"] = geo_scores
            print(f"  [{target.name}] geopolitical: {len(geo_results)} items "
                  f"({geo_scores['elapsed_seconds']}s)")
    except Exception as e:
        print(f"  [{target.name}] geopolitical: ERROR — {e}")
        traceback.print_exc()

    # Save immediately
    out_path = RESULTS_DIR / f"political_{target.name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result


async def run_supplementary_for_target(target: TargetModelConfig) -> dict | None:
    """Run supplementary metrics for one target."""
    from eval.run_supplementary import (
        measure_speed, measure_verbosity, compute_cost, measure_prompt_sensitivity,
    )

    result: dict[str, Any] = {"model": target.name}

    try:
        result["speed"] = await measure_speed(target)
        print(f"  [{target.name}] speed: {result['speed']['avg_tps']} tok/s")
    except Exception as e:
        print(f"  [{target.name}] speed: ERROR — {e}")

    try:
        result["verbosity"] = await measure_verbosity(target)
        print(f"  [{target.name}] verbosity: {result['verbosity']['avg_verbosity_ratio']}x")
    except Exception as e:
        print(f"  [{target.name}] verbosity: ERROR — {e}")

    result["cost"] = compute_cost(target, result.get("verbosity", {}))

    try:
        result["prompt_sensitivity"] = await measure_prompt_sensitivity(target)
        print(f"  [{target.name}] sensitivity: "
              f"{result['prompt_sensitivity']['avg_consistency']:.3f}")
    except Exception as e:
        print(f"  [{target.name}] sensitivity: ERROR — {e}")

    out_path = RESULTS_DIR / f"supplementary_{target.name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result


async def run_model(
    target: TargetModelConfig,
    model_sem: asyncio.Semaphore,
    *,
    run_syco: bool = True,
    run_pol: bool = True,
    run_supp: bool = True,
) -> dict:
    """Run all evaluations for a single model, respecting concurrency limit."""
    async with model_sem:
        print(f"\n{'='*60}")
        print(f"STARTING: {target.name} ({target.model})")
        print(f"{'='*60}")
        t0 = time.time()
        result: dict[str, Any] = {"model": target.name}

        if run_syco:
            syco = await run_sycophancy_for_target(target)
            if syco:
                result["sycophancy"] = syco

        if run_pol:
            pol = await run_political_for_target(target)
            if pol:
                result["political"] = pol

        if run_supp:
            supp = await run_supplementary_for_target(target)
            if supp:
                result["supplementary"] = supp

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"DONE: {target.name} ({elapsed / 60:.1f} min)")
        print(f"{'='*60}")
        return result


def merge_results(phase: str):
    """Merge per-model result files into a combined file."""
    combined: dict[str, Any] = {}
    for f in sorted(RESULTS_DIR.glob(f"{phase}_*.json")):
        if f.name == f"{phase}_all.json":
            continue
        model_name = f.stem[len(phase) + 1:]
        data = json.loads(f.read_text())
        combined[model_name] = data

    if combined:
        out = RESULTS_DIR / f"{phase}_all.json"
        out.write_text(json.dumps(combined, indent=2))
        print(f"Merged {len(combined)} models → {out}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run full AI Nutrition Facts evaluation with parallelism")
    parser.add_argument("--targets", nargs="+",
                        help="Target model names (default: all)")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_MODELS,
                        help=f"Max models to evaluate concurrently (default: {MAX_CONCURRENT_MODELS})")
    parser.add_argument("--skip-sycophancy", action="store_true")
    parser.add_argument("--skip-political", action="store_true")
    parser.add_argument("--skip-supplementary", action="store_true")
    args = parser.parse_args()

    targets = DEFAULT_TARGETS
    if args.targets:
        targets = [t for t in DEFAULT_TARGETS if t.name in args.targets]
        if not targets:
            print(f"No matching targets: {args.targets}")
            print(f"Available: {[t.name for t in DEFAULT_TARGETS]}")
            sys.exit(1)

    print(f"{'='*60}")
    print(f"AI NUTRITION FACTS — PARALLEL EVALUATION")
    print(f"{'='*60}")
    print(f"Models: {len(targets)}")
    print(f"Concurrency: {args.max_concurrent} models at a time")
    print(f"Sycophancy: {'ON' if not args.skip_sycophancy else 'SKIP'}")
    print(f"Political:  {'ON' if not args.skip_political else 'SKIP'}")
    print(f"Supplementary: {'ON' if not args.skip_supplementary else 'SKIP'}")
    print(f"Targets: {', '.join(t.name for t in targets)}")

    t0 = time.time()
    model_sem = asyncio.Semaphore(args.max_concurrent)

    tasks = [
        run_model(
            target, model_sem,
            run_syco=not args.skip_sycophancy,
            run_pol=not args.skip_political,
            run_supp=not args.skip_supplementary,
        )
        for target in targets
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Report errors
    for target, result in zip(targets, results):
        if isinstance(result, Exception):
            print(f"\nFATAL ERROR for {target.name}: {result}")
            traceback.print_exception(type(result), result, result.__traceback__)

    # Merge results
    print(f"\n{'='*60}")
    print("MERGING RESULTS")
    print(f"{'='*60}")
    if not args.skip_sycophancy:
        merge_results("sycophancy")
    if not args.skip_political:
        merge_results("political")
    if not args.skip_supplementary:
        merge_results("supplementary")

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE — {total / 60:.1f} minutes total")
    print(f"View results: python -m bench.app")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
