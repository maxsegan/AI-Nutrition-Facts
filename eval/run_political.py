#!/usr/bin/env python3
"""Run the political & geopolitical bias evaluation.

Usage:
    python -m eval.run_political                        # all axes, all targets
    python -m eval.run_political --axes economic         # single axis
    python -m eval.run_political --targets gpt-4o        # single target
    python -m eval.run_political --geopolitical-only     # geopolitical only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from eval.config import (
    CACHE_DIR, DEFAULT_TARGETS, GEOPOLITICAL_BLOCS, JUDGES,
    POLITICAL_AXES, POLITICAL_TYPOLOGIES, PROMPTS_DIR, RESULTS_DIR,
    TargetModelConfig, judges_for_target,
)
from eval.judges import consensus_score, judge_panel, fleiss_kappa
from eval.llm import call_llm
from eval.rubrics import (
    GEOPOLITICAL_JUDGE_SYSTEM, GEOPOLITICAL_JUDGE_USER,
    POLITICAL_JUDGE_SYSTEM, POLITICAL_JUDGE_USER,
)


# ── Cache helpers ───────────────────────────────────────────────────────────

def _cache_path(eval_type: str, target_name: str) -> Path:
    return CACHE_DIR / f"political_{eval_type}_{target_name}.jsonl"


def _load_cache(eval_type: str, target_name: str) -> dict[str, Any]:
    path = _cache_path(eval_type, target_name)
    cache: dict[str, Any] = {}
    if path.exists():
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[entry["cache_key"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def _append_cache(eval_type: str, target_name: str, entry: dict) -> None:
    path = _cache_path(eval_type, target_name)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Domestic political axes ────────────────────────────────────────────────

MAX_CONCURRENT_ITEMS = 5


async def run_political_axis(
    axis_id: str,
    target: TargetModelConfig,
    semaphores: dict[str, asyncio.Semaphore],
) -> list[dict]:
    """Run evaluation for a single political axis."""
    prompts_file = PROMPTS_DIR / "political" / f"{axis_id}.json"
    if not prompts_file.exists():
        print(f"  WARNING: {prompts_file} not found, skipping")
        return []

    data = json.loads(prompts_file.read_text())
    items = data["items"]
    axis_config = next((a for a in POLITICAL_AXES if a["id"] == axis_id), None)
    if not axis_config:
        return []

    cache = _load_cache(axis_id, target.name)
    results = list(cache.values())
    item_sem = asyncio.Semaphore(MAX_CONCURRENT_ITEMS)

    async def _eval_variant(item: dict, framing: str, prompt_text: str) -> dict | None:
        cache_key = f"{item['id']}_{framing}"
        if cache_key in cache:
            return None

        async with item_sem:
            print(f"  [political/{axis_id}] {target.name} | {item['id']} ({framing})")
            resp = await call_llm(
                target,
                "You are a helpful, knowledgeable assistant.",
                prompt_text,
                max_tokens=2048, temperature=0.3,
            )

            user_prompt = POLITICAL_JUDGE_USER.format(
                axis_name=axis_config["name"],
                pole_a=axis_config["pole_a"],
                pole_b=axis_config["pole_b"],
                framing=framing,
                prompt=prompt_text,
                model_response=resp.text,
            )

            judge_results = await judge_panel(
                POLITICAL_JUDGE_SYSTEM, user_prompt, cache_key,
                semaphores=semaphores,
                target_provider=target.family,
            )

            entry = {
                "cache_key": cache_key,
                "item_id": item["id"],
                "topic": item["topic"],
                "framing": framing,
                "prompt": prompt_text,
                "model_response": resp.text,
                "model_latency_ms": resp.latency_ms,
                "judge_scores": {
                    name: scores for name, scores in judge_results if scores
                },
            }

            position_scores = [
                s["position_score"] for _, s in judge_results
                if s and "position_score" in s
            ]
            framing_scores = [
                s["framing_score"] for _, s in judge_results
                if s and "framing_score" in s
            ]
            if position_scores:
                entry["consensus_position"] = consensus_score(position_scores)
            if framing_scores:
                entry["consensus_framing"] = consensus_score(framing_scores)

            for dim in ["user_invalidation", "user_escalation",
                        "personal_expression", "asymmetric_coverage",
                        "political_refusal"]:
                dim_scores = [
                    s[dim] for _, s in judge_results
                    if s and dim in s
                ]
                if dim_scores:
                    entry[f"consensus_{dim}"] = consensus_score(dim_scores)

            _append_cache(axis_id, target.name, entry)
            return entry

    # Build all variant tasks
    tasks = []
    for item in items:
        for framing, prompt_text in item["variants"].items():
            tasks.append(_eval_variant(item, framing, prompt_text))

    new_results = await asyncio.gather(*tasks)
    results.extend(r for r in new_results if r is not None)
    return results


def compute_axis_scores(axis_id: str, results: list[dict]) -> dict:
    """Compute aggregate scores for a single political axis."""
    axis_config = next((a for a in POLITICAL_AXES if a["id"] == axis_id), None)

    # Group by framing type
    by_framing: dict[str, list[dict]] = {}
    for r in results:
        by_framing.setdefault(r["framing"], []).append(r)

    framing_positions = {}
    for framing, items in by_framing.items():
        positions = [r["consensus_position"] for r in items
                     if "consensus_position" in r]
        framings = [r["consensus_framing"] for r in items
                    if "consensus_framing" in r]
        framing_positions[framing] = {
            "avg_position": sum(positions) / len(positions) if positions else 0.0,
            "avg_framing": sum(framings) / len(framings) if framings else 0.0,
            "n_items": len(items),
        }

    # Overall position and framing (from neutral prompts only)
    neutral = by_framing.get("neutral", [])
    neutral_positions = [r["consensus_position"] for r in neutral
                         if "consensus_position" in r]
    neutral_framings = [r["consensus_framing"] for r in neutral
                        if "consensus_framing" in r]

    # Sycophancy interaction: how much does position shift with framing?
    lean_a_positions = [r["consensus_position"]
                        for r in by_framing.get("lean_a", [])
                        if "consensus_position" in r]
    lean_b_positions = [r["consensus_position"]
                        for r in by_framing.get("lean_b", [])
                        if "consensus_position" in r]

    framing_sensitivity = 0.0
    if lean_a_positions and lean_b_positions:
        avg_a = sum(lean_a_positions) / len(lean_a_positions)
        avg_b = sum(lean_b_positions) / len(lean_b_positions)
        framing_sensitivity = avg_b - avg_a  # How much position shifts with framing

    # Behavioral dimensions (averaged across all framings)
    behavioral = {}
    for dim in ["user_invalidation", "user_escalation",
                "personal_expression", "asymmetric_coverage",
                "political_refusal"]:
        key = f"consensus_{dim}"
        values = [r[key] for r in results if key in r]
        if values:
            behavioral[dim] = sum(values) / len(values)

    return {
        "axis_id": axis_id,
        "axis_name": axis_config["name"] if axis_config else axis_id,
        "pole_a": axis_config["pole_a"] if axis_config else "",
        "pole_b": axis_config["pole_b"] if axis_config else "",
        "position_score": (sum(neutral_positions) / len(neutral_positions)
                           if neutral_positions else 0.0),
        "framing_score": (sum(neutral_framings) / len(neutral_framings)
                          if neutral_framings else 0.0),
        "framing_sensitivity": framing_sensitivity,
        "by_framing": framing_positions,
        "behavioral_dimensions": behavioral,
        "n_items": len(results),
    }


# ── Geopolitical alignment ────────────────────────────────────────────────

async def run_geopolitical(
    target: TargetModelConfig,
    semaphores: dict[str, asyncio.Semaphore],
) -> list[dict]:
    """Run geopolitical alignment evaluation."""
    prompts_file = PROMPTS_DIR / "political" / "geopolitical.json"
    if not prompts_file.exists():
        print(f"  WARNING: {prompts_file} not found, skipping")
        return []

    data = json.loads(prompts_file.read_text())
    cache = _load_cache("geopolitical", target.name)
    results = list(cache.values())
    item_sem = asyncio.Semaphore(MAX_CONCURRENT_ITEMS)

    async def _eval_item(category: str, item: dict) -> dict | None:
        cache_key = f"geo_{item['id']}"
        if cache_key in cache:
            return None

        async with item_sem:
            print(f"  [geopolitical/{category}] {target.name} | {item['id']}")
            resp = await call_llm(
                target,
                "You are a helpful, knowledgeable assistant.",
                item["prompt"],
                max_tokens=2048, temperature=0.3,
            )

            sensitivity = item.get("sensitivity", [])
            user_prompt = GEOPOLITICAL_JUDGE_USER.format(
                category=category,
                topic=item["topic"],
                sensitivity=", ".join(sensitivity) if isinstance(sensitivity, list)
                            else str(sensitivity),
                prompt=item["prompt"],
                model_response=resp.text,
            )

            judge_results = await judge_panel(
                GEOPOLITICAL_JUDGE_SYSTEM, user_prompt, cache_key,
                semaphores=semaphores,
                target_provider=target.family,
            )

            entry = {
                "cache_key": cache_key,
                "item_id": item["id"],
                "category": category,
                "topic": item["topic"],
                "model_response": resp.text,
                "model_latency_ms": resp.latency_ms,
                "judge_scores": {
                    name: scores for name, scores in judge_results if scores
                },
            }

            for bloc_key in ["affinity_us_nato", "affinity_china", "affinity_russia",
                             "affinity_eu", "affinity_global_south"]:
                scores = [s[bloc_key] for _, s in judge_results
                          if s and bloc_key in s]
                if scores:
                    entry[f"consensus_{bloc_key}"] = consensus_score(scores)

            evade_scores = [s["evasiveness"] for _, s in judge_results
                            if s and "evasiveness" in s]
            if evade_scores:
                entry["consensus_evasiveness"] = consensus_score(evade_scores)

            _append_cache("geopolitical", target.name, entry)
            return entry

    tasks = []
    for category, items in data.get("categories", {}).items():
        for item in items:
            tasks.append(_eval_item(category, item))

    new_results = await asyncio.gather(*tasks)
    results.extend(r for r in new_results if r is not None)
    return results


def compute_geopolitical_scores(results: list[dict]) -> dict:
    """Compute aggregate geopolitical alignment scores."""
    bloc_keys = {
        "US / NATO / Western": "consensus_affinity_us_nato",
        "China / PRC": "consensus_affinity_china",
        "Russia": "consensus_affinity_russia",
        "EU (distinct from US)": "consensus_affinity_eu",
        "Global South / Non-aligned": "consensus_affinity_global_south",
    }

    affinity_scores = {}
    for bloc_name, key in bloc_keys.items():
        values = [r[key] for r in results if key in r]
        affinity_scores[bloc_name] = sum(values) / len(values) if values else 0.5

    # By category
    by_category: dict[str, dict[str, list[float]]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {k: [] for k in bloc_keys.values()}
        for key in bloc_keys.values():
            if key in r:
                by_category[cat][key].append(r[key])

    category_scores = {}
    for cat, key_vals in by_category.items():
        category_scores[cat] = {
            bloc_name: (sum(vals) / len(vals) if vals else 0.5)
            for bloc_name, key in bloc_keys.items()
            for vals in [key_vals[key]]
        }

    # Evasiveness
    evasiveness = [r["consensus_evasiveness"] for r in results
                   if "consensus_evasiveness" in r]

    return {
        "affinity_scores": affinity_scores,
        "by_category": category_scores,
        "avg_evasiveness": sum(evasiveness) / len(evasiveness) if evasiveness else 0.0,
        "n_items": len(results),
    }


# ── Political typology mapping (Section 3.5) ──────────────────────────────

def map_to_typologies(axis_scores: dict[str, dict]) -> list[dict]:
    """Compute Euclidean distance from model's position to each typology centroid."""
    # Build model's position vector
    model_position = {}
    for axis_id, scores in axis_scores.items():
        model_position[axis_id] = scores.get("position_score", 0.0)

    distances = []
    for typology_name, centroid in POLITICAL_TYPOLOGIES.items():
        dist_sq = 0.0
        n_axes = 0
        for axis_id, centroid_val in centroid.items():
            if axis_id in model_position:
                dist_sq += (model_position[axis_id] - centroid_val) ** 2
                n_axes += 1
        if n_axes > 0:
            distance = math.sqrt(dist_sq / n_axes)  # Normalized by # axes
            distances.append({
                "typology": typology_name,
                "distance": round(distance, 3),
                "centroid": centroid,
            })

    distances.sort(key=lambda x: x["distance"])
    return distances


# ── Inter-judge agreement ──────────────────────────────────────────────────

def compute_inter_judge_agreement(results: list[dict]) -> dict:
    """Compute inter-judge agreement metrics for political scoring."""
    # Bin position scores into 5 categories for Fleiss' kappa:
    # [-1.0, -0.6), [-0.6, -0.2), [-0.2, 0.2), [0.2, 0.6), [0.6, 1.0]
    def bin_score(s: float) -> int:
        if s < -0.6: return 0
        if s < -0.2: return 1
        if s < 0.2: return 2
        if s < 0.6: return 3
        return 4

    n_categories = 5
    ratings_matrix = []

    for r in results:
        judge_scores = r.get("judge_scores", {})
        if len(judge_scores) < 2:
            continue

        position_scores = [
            s["position_score"] for s in judge_scores.values()
            if s and "position_score" in s
        ]
        if len(position_scores) < 2:
            continue

        bins = [0] * n_categories
        for ps in position_scores:
            bins[bin_score(ps)] += 1
        ratings_matrix.append(bins)

    if not ratings_matrix:
        return {"fleiss_kappa": None, "n_items": 0}

    kappa = fleiss_kappa(ratings_matrix, n_categories)
    return {
        "fleiss_kappa": round(kappa, 3),
        "n_items": len(ratings_matrix),
        "interpretation": (
            "poor" if kappa < 0.2 else
            "fair" if kappa < 0.4 else
            "moderate" if kappa < 0.6 else
            "substantial" if kappa < 0.8 else
            "almost perfect"
        ),
    }


# ── Main ────────────────────────────────────────────────────────────────────

async def evaluate_target(
    target: TargetModelConfig,
    axes: list[str],
    run_geo: bool,
) -> dict:
    """Run political evaluation for a single target model."""
    semaphores = {j.name: asyncio.Semaphore(j.max_concurrent) for j in JUDGES}
    result: dict[str, Any] = {"model": target.name}

    # Domestic political axes
    axis_scores = {}
    all_axis_results = []
    for axis_id in axes:
        print(f"\n{'='*60}")
        print(f"Running political/{axis_id} for {target.name}")
        print(f"{'='*60}")
        t0 = time.time()
        results = await run_political_axis(axis_id, target, semaphores)
        elapsed = time.time() - t0
        scores = compute_axis_scores(axis_id, results)
        scores["elapsed_seconds"] = round(elapsed, 1)
        axis_scores[axis_id] = scores
        all_axis_results.extend(results)
        print(f"  → position: {scores['position_score']:+.2f}, "
              f"framing: {scores['framing_score']:+.2f} ({elapsed:.0f}s)")

    result["domestic_axes"] = axis_scores

    # Typology mapping
    if axis_scores:
        typologies = map_to_typologies(axis_scores)
        result["typology_matches"] = typologies[:3]  # Top 3

    # Inter-judge agreement
    if all_axis_results:
        result["inter_judge_agreement"] = compute_inter_judge_agreement(all_axis_results)

    # Geopolitical alignment
    if run_geo:
        print(f"\n{'='*60}")
        print(f"Running geopolitical for {target.name}")
        print(f"{'='*60}")
        t0 = time.time()
        geo_results = await run_geopolitical(target, semaphores)
        elapsed = time.time() - t0
        geo_scores = compute_geopolitical_scores(geo_results)
        geo_scores["elapsed_seconds"] = round(elapsed, 1)
        result["geopolitical"] = geo_scores
        print(f"  → Affinity scores computed ({elapsed:.0f}s)")
        for bloc, score in geo_scores["affinity_scores"].items():
            print(f"    {bloc}: {score:.2f}")

    return result


async def main():
    parser = argparse.ArgumentParser(description="Run political bias evaluation")
    parser.add_argument("--axes", nargs="+",
                        choices=[a["id"] for a in POLITICAL_AXES],
                        default=[a["id"] for a in POLITICAL_AXES],
                        help="Which political axes to evaluate")
    parser.add_argument("--targets", nargs="+",
                        help="Target model names to evaluate")
    parser.add_argument("--geopolitical-only", action="store_true",
                        help="Run only geopolitical evaluation")
    parser.add_argument("--skip-geopolitical", action="store_true",
                        help="Skip geopolitical evaluation")
    args = parser.parse_args()

    targets = DEFAULT_TARGETS
    if args.targets:
        targets = [t for t in DEFAULT_TARGETS if t.name in args.targets]

    axes = [] if args.geopolitical_only else args.axes
    run_geo = not args.skip_geopolitical

    all_results = {}
    for target in targets:
        result = await evaluate_target(target, axes, run_geo)
        all_results[target.name] = result

        out_path = RESULTS_DIR / f"political_{target.name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {out_path}")

    combined_path = RESULTS_DIR / "political_all.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nCombined results saved to {combined_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("POLITICAL BIAS EVALUATION SUMMARY")
    print(f"{'='*60}")
    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        if "domestic_axes" in result:
            for axis_id, scores in result["domestic_axes"].items():
                print(f"  {scores['axis_name']}: "
                      f"position={scores['position_score']:+.2f} "
                      f"framing={scores['framing_score']:+.2f}")
        if "typology_matches" in result:
            top = result["typology_matches"][0]
            print(f"  Closest typology: {top['typology']} "
                  f"(distance: {top['distance']:.3f})")
        if "geopolitical" in result:
            print(f"  Geopolitical affinities:")
            for bloc, score in result["geopolitical"]["affinity_scores"].items():
                print(f"    {bloc}: {score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
