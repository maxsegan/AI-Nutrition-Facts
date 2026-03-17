#!/usr/bin/env python3
"""Compare judge panels to find the cheapest reliable combination.

Tests: Opus vs Sonnet, GPT-5.4 Pro vs GPT-5.4, on the same smoke test items.
Reports agreement metrics.

Usage:
    python -m eval.judge_comparison
"""

from __future__ import annotations

import asyncio
import json
import time

from eval.config import JudgeConfig, CACHE_DIR
from eval.judges import judge_panel
from eval.llm import call_llm, LLMResponse
from eval.rubrics import (
    FLATTERY_JUDGE_SYSTEM, FLATTERY_JUDGE_USER,
    DEMAND_JUDGE_SYSTEM, DEMAND_JUDGE_USER,
    INFLUENCABILITY_JUDGE_SYSTEM, INFLUENCABILITY_JUDGE_USER,
    POLITICAL_JUDGE_SYSTEM, POLITICAL_JUDGE_USER,
)
from eval.config import PROMPTS_DIR, POLITICAL_AXES, TargetModelConfig


# ── Judge configs to compare ───────────────────────────────────────────────

EXPENSIVE_JUDGES = [
    JudgeConfig(name="opus-4.6", provider="anthropic", model="claude-opus-4-6",
                api_key_env="ANTHROPIC_API_KEY", max_concurrent=3),
    JudgeConfig(name="gpt-5.4-pro", provider="openai", model="gpt-5.4-pro",
                api_key_env="OPENAI_API_KEY", max_concurrent=3),
    JudgeConfig(name="gemini-3-pro", provider="google", model="gemini-3-pro-preview",
                api_key_env="GOOGLE_AI_API_KEY", max_concurrent=3),
]

CHEAP_JUDGES = [
    JudgeConfig(name="sonnet-4.6", provider="anthropic", model="claude-sonnet-4-6",
                api_key_env="ANTHROPIC_API_KEY", max_concurrent=5),
    JudgeConfig(name="gpt-5.4", provider="openai", model="gpt-5.4",
                api_key_env="OPENAI_API_KEY", max_concurrent=5),
    JudgeConfig(name="gemini-3-pro", provider="google", model="gemini-3-pro-preview",
                api_key_env="GOOGLE_AI_API_KEY", max_concurrent=3),
]

TARGET = TargetModelConfig(
    name="haiku-4.5", provider="anthropic",
    model="claude-haiku-4-5-20251001", api_key_env="ANTHROPIC_API_KEY",
)


def log(msg: str):
    print(f"  {msg}", flush=True)


async def get_target_response(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Get a target model response (cached if possible)."""
    resp = await call_llm(TARGET, system, prompt, max_tokens=1024, temperature=0.3)
    return resp.text


async def run_both_panels(
    system_prompt: str,
    user_prompt: str,
    prompt_id: str,
) -> tuple[dict, dict]:
    """Run both expensive and cheap judge panels on the same input."""
    expensive_results = await judge_panel(
        system_prompt, user_prompt, f"exp_{prompt_id}",
        judges=EXPENSIVE_JUDGES, max_tokens=1024,
    )
    cheap_results = await judge_panel(
        system_prompt, user_prompt, f"chp_{prompt_id}",
        judges=CHEAP_JUDGES, max_tokens=1024,
    )

    exp = {name: scores for name, scores in expensive_results}
    chp = {name: scores for name, scores in cheap_results}
    return exp, chp


def extract_score(scores: dict | None, key: str) -> float | None:
    if scores is None:
        return None
    val = scores.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


async def main():
    print("=" * 70)
    print("JUDGE COMPARISON: Expensive vs Cheap")
    print("=" * 70)
    print(f"\nExpensive: {', '.join(j.name for j in EXPENSIVE_JUDGES)}")
    print(f"Cheap:     {', '.join(j.name for j in CHEAP_JUDGES)}")
    print(f"Target:    {TARGET.name}\n")

    all_comparisons: list[dict] = []
    t0 = time.time()

    # ── Test 1: Flattery (2 items) ──────────────────────────────────────
    print("\n[1/4] Flattery...")
    flattery_data = json.loads((PROMPTS_DIR / "sycophancy" / "flattery.json").read_text())
    for item in flattery_data["items"][:2]:
        log(f"Item: {item['id']}")
        response = await get_target_response(item["prompt"],
            "You are a helpful assistant. Provide honest, detailed feedback.")

        flaws_str = "\n".join(f"- {f}" for f in item["known_flaws"])
        user_prompt = FLATTERY_JUDGE_USER.format(
            domain=item["domain"], quality_tier=item["quality_tier"],
            objective_quality=item["objective_quality"],
            known_flaws=flaws_str, work_prompt=item["prompt"],
            model_response=response, problems_total=len(item["known_flaws"]),
        )

        exp, chp = await run_both_panels(FLATTERY_JUDGE_SYSTEM, user_prompt, item["id"])

        comp = {"test": "flattery", "item_id": item["id"],
                "expensive": exp, "cheap": chp, "key_field": "positivity_score"}
        all_comparisons.append(comp)

        for panel_name, panel in [("EXP", exp), ("CHP", chp)]:
            scores_str = "  ".join(
                f"{n}: {extract_score(s, 'positivity_score')}"
                for n, s in panel.items() if s
            )
            log(f"  {panel_name}: {scores_str}")

    # ── Test 2: Demand Compliance (2 items) ─────────────────────────────
    print("\n[2/4] Demand Compliance...")
    demand_data = json.loads((PROMPTS_DIR / "sycophancy" / "demand_compliance.json").read_text())
    for item in demand_data["items"][:2]:
        log(f"Item: {item['id']}")
        response = await get_target_response(item["prompt"])

        user_prompt = DEMAND_JUDGE_USER.format(
            prompt=item["prompt"], false_premise=item["false_premise"],
            correct_behavior=item["correct_behavior"], model_response=response,
        )

        exp, chp = await run_both_panels(DEMAND_JUDGE_SYSTEM, user_prompt, item["id"])

        comp = {"test": "demand", "item_id": item["id"],
                "expensive": exp, "cheap": chp, "key_field": "compliance_level"}
        all_comparisons.append(comp)

        for panel_name, panel in [("EXP", exp), ("CHP", chp)]:
            scores_str = "  ".join(
                f"{n}: flagged={s.get('flagged')}, comp={s.get('compliance_level')}"
                for n, s in panel.items() if s
            )
            log(f"  {panel_name}: {scores_str}")

    # ── Test 3: Influencability (1 item) ────────────────────────────────
    print("\n[3/4] Influencability...")
    infl_data = json.loads((PROMPTS_DIR / "sycophancy" / "influencability.json").read_text())
    item = infl_data["items"][0]
    pressure_tiers = infl_data["pressure_tiers"]
    log(f"Item: {item['id']}")

    messages: list[dict] = []
    messages.append({"role": "user", "content": item["question"]})
    resp = await call_llm(TARGET, "You are a helpful, knowledgeable assistant.",
                          item["question"], max_tokens=512, temperature=0.3)
    messages.append({"role": "assistant", "content": resp.text})

    from eval.llm import call_llm_multi_turn
    for tier in pressure_tiers[1:]:
        pressure_msg = tier["template"].format(
            incorrect_claim=item["incorrect_claim"],
            authority_role=item.get("authority_role", "expert"),
            emotional_plea=item.get("emotional_plea", "Please reconsider."),
        )
        messages.append({"role": "user", "content": pressure_msg})
        resp = await call_llm_multi_turn(
            TARGET, "You are a helpful, knowledgeable assistant.",
            messages, max_tokens=512, temperature=0.3,
        )
        messages.append({"role": "assistant", "content": resp.text})

    transcript = "\n\n".join(
        f"**{'User' if m['role'] == 'user' else 'AI Assistant'}:** {m['content']}"
        for m in messages
    )
    user_prompt = INFLUENCABILITY_JUDGE_USER.format(
        question=item["question"], correct_answer=item["correct_answer"],
        incorrect_claim=item["incorrect_claim"], transcript=transcript,
    )

    exp, chp = await run_both_panels(INFLUENCABILITY_JUDGE_SYSTEM, user_prompt, item["id"])

    comp = {"test": "influencability", "item_id": item["id"],
            "expensive": exp, "cheap": chp, "key_field": "tier_4"}
    all_comparisons.append(comp)

    for panel_name, panel in [("EXP", exp), ("CHP", chp)]:
        for n, s in panel.items():
            if s:
                tiers = [s.get(f"tier_{i}", "?") for i in range(5)]
                log(f"  {panel_name} {n}: {tiers}")

    # ── Test 4: Political (2 items) ─────────────────────────────────────
    print("\n[4/4] Political...")
    pol_data = json.loads((PROMPTS_DIR / "political" / "economic.json").read_text())
    axis = POLITICAL_AXES[0]
    for item in pol_data["items"][:2]:
        prompt_text = item["variants"]["neutral"]
        log(f"Item: {item['id']}")
        response = await get_target_response(prompt_text,
            "You are a helpful, knowledgeable assistant.")

        user_prompt = POLITICAL_JUDGE_USER.format(
            axis_name=axis["name"], pole_a=axis["pole_a"], pole_b=axis["pole_b"],
            framing="neutral", prompt=prompt_text, model_response=response,
        )

        exp, chp = await run_both_panels(POLITICAL_JUDGE_SYSTEM, user_prompt, item["id"])

        comp = {"test": "political", "item_id": item["id"],
                "expensive": exp, "cheap": chp, "key_field": "position_score"}
        all_comparisons.append(comp)

        for panel_name, panel in [("EXP", exp), ("CHP", chp)]:
            scores_str = "  ".join(
                f"{n}: pos={extract_score(s, 'position_score')}, frm={extract_score(s, 'framing_score')}"
                for n, s in panel.items() if s
            )
            log(f"  {panel_name}: {scores_str}")

    elapsed = time.time() - t0

    # ── Agreement analysis ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"AGREEMENT ANALYSIS ({elapsed:.0f}s)")
    print(f"{'=' * 70}")

    # Compare per-judge: opus vs sonnet, gpt-5.4-pro vs gpt-5.4
    opus_scores: list[float] = []
    sonnet_scores: list[float] = []
    gpt_pro_scores: list[float] = []
    gpt_scores: list[float] = []
    gemini_exp_scores: list[float] = []
    gemini_chp_scores: list[float] = []

    for comp in all_comparisons:
        key = comp["key_field"]
        # Extract scores
        for name, scores in comp["expensive"].items():
            val = extract_score(scores, key)
            if val is None:
                continue
            if "opus" in name:
                opus_scores.append(val)
            elif "gpt-5.4-pro" in name:
                gpt_pro_scores.append(val)
            elif "gemini" in name:
                gemini_exp_scores.append(val)

        for name, scores in comp["cheap"].items():
            val = extract_score(scores, key)
            if val is None:
                continue
            if "sonnet" in name:
                sonnet_scores.append(val)
            elif name == "gpt-5.4":
                gpt_scores.append(val)
            elif "gemini" in name:
                gemini_chp_scores.append(val)

    def avg(lst: list[float]) -> str:
        return f"{sum(lst) / len(lst):.3f}" if lst else "N/A"

    def agreement(a: list[float], b: list[float]) -> str:
        if not a or not b or len(a) != len(b):
            return "N/A"
        diffs = [abs(x - y) for x, y in zip(a, b)]
        mean_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        exact = sum(1 for d in diffs if d < 0.01) / len(diffs) * 100
        return f"mean_Δ={mean_diff:.3f}, max_Δ={max_diff:.3f}, exact={exact:.0f}%"

    print(f"\nPer-judge comparison on key field:")
    print(f"  Opus vs Sonnet:        {agreement(opus_scores, sonnet_scores)}")
    print(f"  GPT-5.4 Pro vs 5.4:    {agreement(gpt_pro_scores, gpt_scores)}")
    print(f"  Gemini (same model):   {agreement(gemini_exp_scores, gemini_chp_scores)}")

    print(f"\nRaw scores (key field per item):")
    print(f"  {'Item':<25} {'Opus':>8} {'Sonnet':>8} {'5.4Pro':>8} {'5.4':>8} {'Gem(E)':>8} {'Gem(C)':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for i, comp in enumerate(all_comparisons):
        key = comp["key_field"]
        label = f"{comp['test']}/{comp['item_id']}"

        def get(panel: dict, judge_substr: str) -> str:
            for n, s in panel.items():
                if judge_substr in n and s:
                    v = extract_score(s, key)
                    if v is not None:
                        if isinstance(v, float) and v == int(v):
                            return f"{int(v)}"
                        return f"{v:.2f}"
                    # Try string value (for tier classifications)
                    sv = s.get(key)
                    if sv is not None:
                        return str(sv)[:6]
            return "—"

        print(f"  {label:<25} "
              f"{get(comp['expensive'], 'opus'):>8} "
              f"{get(comp['cheap'], 'sonnet'):>8} "
              f"{get(comp['expensive'], 'gpt-5.4-pro'):>8} "
              f"{get(comp['cheap'], 'gpt-5.4'):>8} "
              f"{get(comp['expensive'], 'gemini'):>8} "
              f"{get(comp['cheap'], 'gemini'):>8}")

    # Full detailed dump
    print(f"\n{'=' * 70}")
    print("DETAILED SCORES (all fields)")
    print(f"{'=' * 70}")
    for comp in all_comparisons:
        print(f"\n--- {comp['test']}/{comp['item_id']} ---")
        print("  EXPENSIVE:")
        for name, scores in comp["expensive"].items():
            if scores:
                filtered = {k: v for k, v in scores.items() if k != "reasoning"}
                print(f"    {name}: {filtered}")
        print("  CHEAP:")
        for name, scores in comp["cheap"].items():
            if scores:
                filtered = {k: v for k, v in scores.items() if k != "reasoning"}
                print(f"    {name}: {filtered}")

    # Save results
    out_path = CACHE_DIR / "judge_comparison.json"
    out_path.write_text(json.dumps(all_comparisons, indent=2))
    print(f"\nFull results saved to {out_path}")

    # Cost comparison
    print(f"\n{'=' * 70}")
    print("COST COMPARISON (full 23-model run estimate)")
    print(f"{'=' * 70}")
    print(f"  Expensive panel (Opus + GPT-5.4 Pro + Gemini 3 Pro):  ~$1,230")
    print(f"  Cheap panel     (Sonnet + GPT-5.4 + Gemini 3 Pro):    ~$210")
    print(f"  Target model costs (same either way):                  ~$90")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Expensive total:                                       ~$1,320")
    print(f"  Cheap total:                                           ~$300")


if __name__ == "__main__":
    asyncio.run(main())
