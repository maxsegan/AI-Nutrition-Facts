#!/usr/bin/env python3
"""Run supplementary evaluation metrics (speed, cost, verbosity, prompt sensitivity).

These are the "additional nutrition facts" from Section 4 of the spec.

Usage:
    python -m eval.run_supplementary
    python -m eval.run_supplementary --metrics speed verbosity
    python -m eval.run_supplementary --targets gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from eval.config import CACHE_DIR, DEFAULT_TARGETS, RESULTS_DIR, TargetModelConfig
from eval.llm import call_llm


# ── Standardized test prompts for supplementary metrics ────────────────────

SPEED_PROMPTS = [
    {"id": "speed-short", "prompt": "What is the capital of France?",
     "expected_length": "short"},
    {"id": "speed-medium", "prompt": "Explain the difference between TCP and UDP protocols.",
     "expected_length": "medium"},
    {"id": "speed-long", "prompt": "Write a comprehensive overview of the causes, major events, and consequences of World War I.",
     "expected_length": "long"},
]

VERBOSITY_PROMPTS = [
    {"id": "verb-01", "prompt": "What is 2+2?", "ideal_tokens": 5},
    {"id": "verb-02", "prompt": "Is the Earth round or flat?", "ideal_tokens": 10},
    {"id": "verb-03", "prompt": "What is the capital of Japan?", "ideal_tokens": 5},
    {"id": "verb-04", "prompt": "True or false: water boils at 100°C at sea level.",
     "ideal_tokens": 8},
    {"id": "verb-05", "prompt": "Name the three primary colors.", "ideal_tokens": 10},
    {"id": "verb-06", "prompt": "Summarize photosynthesis in one sentence.",
     "ideal_tokens": 25},
    {"id": "verb-07", "prompt": "What year did the Titanic sink?", "ideal_tokens": 8},
    {"id": "verb-08", "prompt": "Define 'entropy' in one sentence.", "ideal_tokens": 20},
    {"id": "verb-09", "prompt": "How many continents are there?", "ideal_tokens": 5},
    {"id": "verb-10", "prompt": "What does DNA stand for?", "ideal_tokens": 10},
]

# Prompt sensitivity: same question rephrased in 3 ways
SENSITIVITY_PROMPTS = [
    {
        "id": "sens-01",
        "topic": "Climate change cause",
        "variants": [
            "What causes climate change?",
            "What is the primary driver of climate change?",
            "Can you explain why the climate is changing?",
        ],
    },
    {
        "id": "sens-02",
        "topic": "Python vs Java",
        "variants": [
            "Should I learn Python or Java first?",
            "Which is better for a beginner: Python or Java?",
            "I want to start programming. Python or Java?",
        ],
    },
    {
        "id": "sens-03",
        "topic": "Exercise benefits",
        "variants": [
            "What are the benefits of regular exercise?",
            "Why is exercise good for you?",
            "How does regular physical activity improve health?",
        ],
    },
    {
        "id": "sens-04",
        "topic": "AI risk",
        "variants": [
            "What are the risks of artificial intelligence?",
            "Is AI dangerous?",
            "What are the potential downsides of AI development?",
        ],
    },
    {
        "id": "sens-05",
        "topic": "Healthy diet",
        "variants": [
            "What does a healthy diet look like?",
            "What should I eat to be healthy?",
            "Can you describe the components of a nutritious diet?",
        ],
    },
]

# Per-token cost data (input / output per million tokens)
# Updated as of early 2026 — add entries as pricing is confirmed
MODEL_COSTS = {
    "opus-4.6": {"input": 15.00, "output": 75.00},
    "sonnet-4.6": {"input": 3.00, "output": 15.00},
    "haiku-4.5": {"input": 0.80, "output": 4.00},
    "gpt-5.4-pro": {"input": 10.00, "output": 40.00},
    "gpt-5.4": {"input": 2.50, "output": 10.00},
    "gpt-5-mini": {"input": 0.40, "output": 1.60},
    "gemini-3.1-pro": {"input": 1.25, "output": 5.00},
    "gemini-3.1-lite": {"input": 0.04, "output": 0.15},
    "gemini-3-flash": {"input": 0.15, "output": 0.60},
    # OpenRouter-routed models — costs vary; these are estimates
    "qwen3.5-397b": {"input": 1.20, "output": 1.20},
    "qwen3.5-27b": {"input": 0.20, "output": 0.20},
    "qwen3.5-9b": {"input": 0.08, "output": 0.08},
    "deepseek-v3.2": {"input": 0.27, "output": 1.10},
    "deepseek-v3.2-speciale": {"input": 0.55, "output": 2.19},
    "llama-4-maverick": {"input": 0.50, "output": 0.50},
    "llama-4-scout": {"input": 0.15, "output": 0.15},
    "grok-4.1": {"input": 3.00, "output": 15.00},
    "grok-4.20": {"input": 5.00, "output": 25.00},
    "grok-4.1-fast": {"input": 1.00, "output": 5.00},
    "mistral-large-3": {"input": 2.00, "output": 6.00},
    "ministral-3-14b": {"input": 0.10, "output": 0.10},
    "kimi-k2.5": {"input": 0.60, "output": 0.60},
}


# ── Speed measurement ──────────────────────────────────────────────────────

async def measure_speed(target: TargetModelConfig, n_runs: int = 3) -> dict:
    """Measure TTFT and TPS across test prompts."""
    all_measurements = []

    for prompt_item in SPEED_PROMPTS:
        measurements = []
        for run_idx in range(n_runs):
            t0 = time.perf_counter()
            resp = await call_llm(
                target,
                "You are a helpful assistant.",
                prompt_item["prompt"],
                max_tokens=2048,
                temperature=0.3,
            )
            total_ms = int((time.perf_counter() - t0) * 1000)

            output_tokens = resp.usage.get("output_tokens", 0)
            # Estimate TPS (total time / output tokens is approximate since
            # we don't have streaming TTFT separately)
            tps = (output_tokens / (total_ms / 1000)) if total_ms > 0 else 0

            measurements.append({
                "run": run_idx,
                "total_ms": total_ms,
                "output_tokens": output_tokens,
                "estimated_tps": round(tps, 1),
            })

        avg_ms = sum(m["total_ms"] for m in measurements) / len(measurements)
        avg_tps = sum(m["estimated_tps"] for m in measurements) / len(measurements)

        all_measurements.append({
            "prompt_id": prompt_item["id"],
            "expected_length": prompt_item["expected_length"],
            "avg_total_ms": round(avg_ms),
            "avg_tps": round(avg_tps, 1),
            "runs": measurements,
        })

    # Overall averages
    total_avg_ms = sum(m["avg_total_ms"] for m in all_measurements) / len(all_measurements)
    total_avg_tps = sum(m["avg_tps"] for m in all_measurements) / len(all_measurements)

    return {
        "avg_response_ms": round(total_avg_ms),
        "avg_tps": round(total_avg_tps, 1),
        "by_length": all_measurements,
    }


# ── Verbosity measurement ─────────────────────────────────────────────────

async def measure_verbosity(target: TargetModelConfig) -> dict:
    """Measure verbosity ratio (actual tokens vs. ideal response length)."""
    measurements = []

    for prompt_item in VERBOSITY_PROMPTS:
        resp = await call_llm(
            target,
            "You are a helpful assistant. Be concise.",
            prompt_item["prompt"],
            max_tokens=1024,
            temperature=0.3,
        )

        output_tokens = resp.usage.get("output_tokens", 0)
        ratio = output_tokens / prompt_item["ideal_tokens"] if prompt_item["ideal_tokens"] > 0 else 0

        measurements.append({
            "prompt_id": prompt_item["id"],
            "ideal_tokens": prompt_item["ideal_tokens"],
            "actual_tokens": output_tokens,
            "verbosity_ratio": round(ratio, 2),
        })

    ratios = [m["verbosity_ratio"] for m in measurements]
    return {
        "avg_verbosity_ratio": round(sum(ratios) / len(ratios), 2) if ratios else 0,
        "median_verbosity_ratio": round(sorted(ratios)[len(ratios) // 2], 2) if ratios else 0,
        "measurements": measurements,
    }


# ── Cost calculation ───────────────────────────────────────────────────────

def compute_cost(target: TargetModelConfig, verbosity: dict) -> dict:
    """Compute per-token and effective cost (adjusted for verbosity)."""
    costs = MODEL_COSTS.get(target.name, {})
    if not costs:
        return {"per_token_cost": "unknown", "effective_cost": "unknown"}

    input_cost = costs["input"]
    output_cost = costs["output"]
    verbosity_ratio = verbosity.get("avg_verbosity_ratio", 1.0)

    return {
        "input_cost_per_mtok": input_cost,
        "output_cost_per_mtok": output_cost,
        "verbosity_ratio": verbosity_ratio,
        "effective_output_cost_per_mtok": round(output_cost * verbosity_ratio, 2),
    }


# ── Prompt sensitivity ────────────────────────────────────────────────────

async def measure_prompt_sensitivity(target: TargetModelConfig) -> dict:
    """Measure how much responses vary with trivial rephrasing."""
    from eval.judges import judge_panel
    from eval.config import JUDGES

    semaphores = {j.name: asyncio.Semaphore(j.max_concurrent) for j in JUDGES}
    measurements = []

    for item in SENSITIVITY_PROMPTS:
        responses = []
        for variant in item["variants"]:
            resp = await call_llm(
                target,
                "You are a helpful assistant.",
                variant,
                max_tokens=1024,
                temperature=0.0,  # Temperature 0 to isolate prompt sensitivity
            )
            responses.append(resp.text)

        # Use judge panel to score consistency between responses
        judge_system = """\
You are evaluating whether three AI responses to rephrased versions of the same \
question are substantively consistent.

Score the consistency:
- "consistency_score": float 0.0-1.0 where 1.0 = all three responses convey the \
  same substantive information and conclusions, 0.0 = completely different answers
- "factual_agreement": boolean — do all responses agree on facts?
- "recommendation_agreement": boolean — do all responses give the same recommendation?
- "reasoning": brief explanation

Respond with JSON only.
"""

        judge_user = f"""\
## Topic: {item["topic"]}

**Variant 1:** "{item["variants"][0]}"
**Response 1:** {responses[0]}

**Variant 2:** "{item["variants"][1]}"
**Response 2:** {responses[1]}

**Variant 3:** "{item["variants"][2]}"
**Response 3:** {responses[2]}

Score consistency. Respond with JSON only.
"""

        judge_results = await judge_panel(
            judge_system, judge_user, f"sens_{item['id']}",
            semaphores=semaphores,
        )

        consistency_scores = [
            s["consistency_score"] for _, s in judge_results
            if s and "consistency_score" in s
        ]
        from eval.judges import consensus_score as cs
        avg_consistency = cs(consistency_scores) if consistency_scores else 0.0

        measurements.append({
            "topic": item["topic"],
            "consistency_score": avg_consistency,
            "judge_scores": {
                name: scores for name, scores in judge_results if scores
            },
        })

    consistency_scores = [m["consistency_score"] for m in measurements]
    return {
        "avg_consistency": (round(sum(consistency_scores) / len(consistency_scores), 3)
                            if consistency_scores else 0.0),
        "measurements": measurements,
    }


# ── Main ────────────────────────────────────────────────────────────────────

AVAILABLE_METRICS = ["speed", "verbosity", "cost", "prompt_sensitivity"]


async def evaluate_target(
    target: TargetModelConfig,
    metrics: list[str],
) -> dict:
    """Run supplementary metrics for a single target model."""
    result: dict[str, Any] = {"model": target.name}

    if "speed" in metrics:
        print(f"\n  Measuring speed for {target.name}...")
        result["speed"] = await measure_speed(target)
        print(f"    → avg response: {result['speed']['avg_response_ms']}ms, "
              f"avg TPS: {result['speed']['avg_tps']}")

    if "verbosity" in metrics:
        print(f"\n  Measuring verbosity for {target.name}...")
        result["verbosity"] = await measure_verbosity(target)
        print(f"    → avg verbosity ratio: {result['verbosity']['avg_verbosity_ratio']}x")

    if "cost" in metrics:
        verbosity = result.get("verbosity", {})
        if not verbosity and "verbosity" not in metrics:
            # Need verbosity for cost calculation
            print(f"\n  Measuring verbosity for cost calculation...")
            verbosity = await measure_verbosity(target)
        result["cost"] = compute_cost(target, verbosity)
        print(f"    → effective output cost: "
              f"${result['cost'].get('effective_output_cost_per_mtok', '?')}/Mtok")

    if "prompt_sensitivity" in metrics:
        print(f"\n  Measuring prompt sensitivity for {target.name}...")
        result["prompt_sensitivity"] = await measure_prompt_sensitivity(target)
        print(f"    → avg consistency: "
              f"{result['prompt_sensitivity']['avg_consistency']:.3f}")

    return result


async def main():
    parser = argparse.ArgumentParser(description="Run supplementary evaluations")
    parser.add_argument("--metrics", nargs="+", choices=AVAILABLE_METRICS,
                        default=AVAILABLE_METRICS,
                        help="Which metrics to evaluate")
    parser.add_argument("--targets", nargs="+",
                        help="Target model names to evaluate")
    args = parser.parse_args()

    targets = DEFAULT_TARGETS
    if args.targets:
        targets = [t for t in DEFAULT_TARGETS if t.name in args.targets]

    all_results = {}
    for target in targets:
        print(f"\n{'='*60}")
        print(f"Supplementary metrics for {target.name}")
        print(f"{'='*60}")
        result = await evaluate_target(target, args.metrics)
        all_results[target.name] = result

        out_path = RESULTS_DIR / f"supplementary_{target.name}.json"
        out_path.write_text(json.dumps(result, indent=2))

    combined_path = RESULTS_DIR / "supplementary_all.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {combined_path}")


if __name__ == "__main__":
    asyncio.run(main())
