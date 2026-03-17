#!/usr/bin/env python3
"""Smoke test: run 2 items per sycophancy subtype on a single model.

Validates the full pipeline: target call → judge panel → scoring → cache.

Usage:
    python -m eval.smoke_test
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback

from eval.config import JUDGES, PROMPTS_DIR, CACHE_DIR, RESULTS_DIR, TargetModelConfig
from eval.llm import call_llm, call_llm_multi_turn, parse_json
from eval.judges import judge_panel, consensus_score

# Use Haiku as the cheap target for smoke testing
SMOKE_TARGET = TargetModelConfig(
    name="haiku-4.5",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    api_key_env="ANTHROPIC_API_KEY",
)

SMOKE_CACHE = CACHE_DIR / "smoke_test.jsonl"


def log(msg: str):
    print(f"  {msg}", flush=True)


def save_result(entry: dict):
    with open(SMOKE_CACHE, "a") as f:
        f.write(json.dumps(entry) + "\n")


async def test_target_call():
    """Test basic LLM call to target model."""
    print("\n[1/6] Testing target model call...")
    try:
        resp = await call_llm(
            SMOKE_TARGET,
            "You are a helpful assistant.",
            "What is 2+2? Reply with just the number.",
            max_tokens=64,
            temperature=0.0,
        )
        log(f"Response: {resp.text[:100]!r}")
        log(f"Usage: {resp.usage}")
        log(f"Latency: {resp.latency_ms}ms")
        assert resp.text.strip(), "Empty response"
        log("PASS")
        return True
    except Exception as e:
        log(f"FAIL: {e}")
        traceback.print_exc()
        return False


async def test_judge_call():
    """Test a single judge panel call."""
    print("\n[2/6] Testing judge panel call...")
    try:
        system = "You are scoring a test. Respond with JSON: {\"score\": 5, \"reasoning\": \"test\"}"
        user = "Score this: 'Hello world'. Respond with JSON only."

        results = await judge_panel(
            system, user, "smoke-judge-test",
            max_tokens=256, temperature=0.0,
        )
        for name, scores in results:
            if scores:
                log(f"  {name}: {scores}")
            else:
                log(f"  {name}: FAILED TO PARSE")

        success = sum(1 for _, s in results if s is not None)
        log(f"{success}/{len(results)} judges returned valid JSON")
        assert success >= 2, f"Only {success} judges succeeded"
        log("PASS")
        return True
    except Exception as e:
        log(f"FAIL: {e}")
        traceback.print_exc()
        return False


async def test_flattery():
    """Smoke test flattery: 2 items."""
    print("\n[3/6] Testing flattery (2 items)...")
    from eval.rubrics import FLATTERY_JUDGE_SYSTEM, FLATTERY_JUDGE_USER
    prompts = json.loads((PROMPTS_DIR / "sycophancy" / "flattery.json").read_text())
    items = prompts["items"][:2]
    semaphores = {j.name: asyncio.Semaphore(j.max_concurrent) for j in JUDGES}

    for item in items:
        log(f"Item: {item['id']} ({item['quality_tier']})")
        try:
            resp = await call_llm(
                SMOKE_TARGET,
                "You are a helpful assistant. Provide honest, detailed feedback.",
                item["prompt"],
                max_tokens=1024, temperature=0.3,
            )
            log(f"  Target response: {len(resp.text)} chars, {resp.latency_ms}ms")

            flaws_str = "\n".join(f"- {f}" for f in item["known_flaws"])
            user_prompt = FLATTERY_JUDGE_USER.format(
                domain=item["domain"],
                quality_tier=item["quality_tier"],
                objective_quality=item["objective_quality"],
                known_flaws=flaws_str,
                work_prompt=item["prompt"],
                model_response=resp.text,
                problems_total=len(item["known_flaws"]),
            )

            judge_results = await judge_panel(
                FLATTERY_JUDGE_SYSTEM, user_prompt, item["id"],
                semaphores=semaphores, max_tokens=1024,
            )

            for name, scores in judge_results:
                if scores:
                    log(f"  Judge {name}: positivity={scores.get('positivity_score','?')}, "
                        f"problems={scores.get('problems_identified','?')}/{len(item['known_flaws'])}")
                else:
                    log(f"  Judge {name}: PARSE FAILED")

            save_result({
                "test": "flattery", "item_id": item["id"],
                "response_len": len(resp.text),
                "judges": {n: s for n, s in judge_results},
            })
        except Exception as e:
            log(f"  FAIL: {e}")
            traceback.print_exc()
            return False
    log("PASS")
    return True


async def test_influencability():
    """Smoke test influencability: 1 item (multi-turn)."""
    print("\n[4/6] Testing influencability (1 item, multi-turn)...")
    from eval.rubrics import INFLUENCABILITY_JUDGE_SYSTEM, INFLUENCABILITY_JUDGE_USER
    prompts = json.loads((PROMPTS_DIR / "sycophancy" / "influencability.json").read_text())
    item = prompts["items"][0]
    pressure_tiers = prompts["pressure_tiers"]
    semaphores = {j.name: asyncio.Semaphore(j.max_concurrent) for j in JUDGES}

    log(f"Item: {item['id']} — {item['question']}")
    try:
        messages: list[dict] = []
        tier_responses: list[str] = []

        # Tier 0
        messages.append({"role": "user", "content": item["question"]})
        resp = await call_llm_multi_turn(
            SMOKE_TARGET, "You are a helpful, knowledgeable assistant.",
            messages, max_tokens=512, temperature=0.3,
        )
        tier_responses.append(resp.text)
        messages.append({"role": "assistant", "content": resp.text})
        log(f"  Tier 0: {resp.text[:80]}...")

        # Tiers 1-4
        for tier in pressure_tiers[1:]:
            pressure_msg = tier["template"].format(
                incorrect_claim=item["incorrect_claim"],
                authority_role=item.get("authority_role", "expert"),
                emotional_plea=item.get("emotional_plea", "Please reconsider."),
            )
            messages.append({"role": "user", "content": pressure_msg})
            resp = await call_llm_multi_turn(
                SMOKE_TARGET, "You are a helpful, knowledgeable assistant.",
                messages, max_tokens=512, temperature=0.3,
            )
            tier_responses.append(resp.text)
            messages.append({"role": "assistant", "content": resp.text})
            log(f"  Tier {tier['tier']}: {resp.text[:80]}...")

        # Judge
        transcript_parts = []
        for i, msg in enumerate(messages):
            role = "User" if msg["role"] == "user" else "AI Assistant"
            transcript_parts.append(f"**{role}:** {msg['content']}")
        transcript = "\n\n".join(transcript_parts)

        user_prompt = INFLUENCABILITY_JUDGE_USER.format(
            question=item["question"],
            correct_answer=item["correct_answer"],
            incorrect_claim=item["incorrect_claim"],
            transcript=transcript,
        )

        judge_results = await judge_panel(
            INFLUENCABILITY_JUDGE_SYSTEM, user_prompt, item["id"],
            semaphores=semaphores, max_tokens=1024,
        )

        for name, scores in judge_results:
            if scores:
                tiers = [scores.get(f"tier_{i}", "?") for i in range(5)]
                log(f"  Judge {name}: {tiers}")
            else:
                log(f"  Judge {name}: PARSE FAILED")

        save_result({
            "test": "influencability", "item_id": item["id"],
            "tier_responses_len": [len(r) for r in tier_responses],
            "judges": {n: s for n, s in judge_results},
        })
    except Exception as e:
        log(f"  FAIL: {e}")
        traceback.print_exc()
        return False
    log("PASS")
    return True


async def test_demand_compliance():
    """Smoke test demand compliance: 2 items."""
    print("\n[5/6] Testing demand compliance (2 items)...")
    from eval.rubrics import DEMAND_JUDGE_SYSTEM, DEMAND_JUDGE_USER
    prompts = json.loads((PROMPTS_DIR / "sycophancy" / "demand_compliance.json").read_text())
    items = prompts["items"][:2]
    semaphores = {j.name: asyncio.Semaphore(j.max_concurrent) for j in JUDGES}

    for item in items:
        log(f"Item: {item['id']} ({item['category']})")
        try:
            resp = await call_llm(
                SMOKE_TARGET, "You are a helpful assistant.",
                item["prompt"], max_tokens=1024, temperature=0.3,
            )
            log(f"  Target response: {len(resp.text)} chars")

            user_prompt = DEMAND_JUDGE_USER.format(
                prompt=item["prompt"],
                false_premise=item["false_premise"],
                correct_behavior=item["correct_behavior"],
                model_response=resp.text,
            )

            judge_results = await judge_panel(
                DEMAND_JUDGE_SYSTEM, user_prompt, item["id"],
                semaphores=semaphores, max_tokens=1024,
            )

            for name, scores in judge_results:
                if scores:
                    log(f"  Judge {name}: flagged={scores.get('flagged','?')}, "
                        f"compliance={scores.get('compliance_level','?')}")
                else:
                    log(f"  Judge {name}: PARSE FAILED")

            save_result({
                "test": "demand_compliance", "item_id": item["id"],
                "judges": {n: s for n, s in judge_results},
            })
        except Exception as e:
            log(f"  FAIL: {e}")
            traceback.print_exc()
            return False
    log("PASS")
    return True


async def test_political():
    """Smoke test political: 1 item, neutral framing."""
    print("\n[6/6] Testing political (1 item)...")
    from eval.rubrics import POLITICAL_JUDGE_SYSTEM, POLITICAL_JUDGE_USER
    from eval.config import POLITICAL_AXES
    prompts = json.loads((PROMPTS_DIR / "political" / "economic.json").read_text())
    item = prompts["items"][0]
    axis = POLITICAL_AXES[0]
    semaphores = {j.name: asyncio.Semaphore(j.max_concurrent) for j in JUDGES}

    prompt_text = item["variants"]["neutral"]
    log(f"Item: {item['id']} — {prompt_text[:60]}...")
    try:
        resp = await call_llm(
            SMOKE_TARGET, "You are a helpful, knowledgeable assistant.",
            prompt_text, max_tokens=1024, temperature=0.3,
        )
        log(f"  Target response: {len(resp.text)} chars")

        user_prompt = POLITICAL_JUDGE_USER.format(
            axis_name=axis["name"],
            pole_a=axis["pole_a"],
            pole_b=axis["pole_b"],
            framing="neutral",
            prompt=prompt_text,
            model_response=resp.text,
        )

        judge_results = await judge_panel(
            POLITICAL_JUDGE_SYSTEM, user_prompt, f"{item['id']}_neutral",
            semaphores=semaphores, max_tokens=1024,
        )

        for name, scores in judge_results:
            if scores:
                log(f"  Judge {name}: position={scores.get('position_score','?')}, "
                    f"framing={scores.get('framing_score','?')}")
            else:
                log(f"  Judge {name}: PARSE FAILED")

        save_result({
            "test": "political", "item_id": item["id"],
            "judges": {n: s for n, s in judge_results},
        })
    except Exception as e:
        log(f"  FAIL: {e}")
        traceback.print_exc()
        return False
    log("PASS")
    return True


async def main():
    print("=" * 60)
    print("AI NUTRITION FACTS — SMOKE TEST")
    print("=" * 60)
    print(f"Target: {SMOKE_TARGET.name} ({SMOKE_TARGET.model})")
    print(f"Judges: {', '.join(j.name for j in JUDGES)}")

    # Clear old smoke cache
    if SMOKE_CACHE.exists():
        SMOKE_CACHE.unlink()

    t0 = time.time()
    results = {}

    tests = [
        ("target_call", test_target_call),
        ("judge_call", test_judge_call),
        ("flattery", test_flattery),
        ("influencability", test_influencability),
        ("demand_compliance", test_demand_compliance),
        ("political", test_political),
    ]

    for name, test_fn in tests:
        try:
            results[name] = await test_fn()
        except Exception as e:
            print(f"\n  UNEXPECTED ERROR in {name}: {e}")
            traceback.print_exc()
            results[name] = False

        # Bail early on fundamental failures
        if name in ("target_call", "judge_call") and not results[name]:
            print(f"\nFundamental failure in {name} — aborting remaining tests.")
            break

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST RESULTS ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\nAll tests passed. Pipeline is operational.")
    else:
        print(f"\nSome tests failed. Check errors above.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
