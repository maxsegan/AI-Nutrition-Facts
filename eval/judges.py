"""Multi-judge consensus engine.

All subjective evaluations use a panel of 3 judges from different providers.
Scores are aggregated via median. Disagreements are flagged.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.config import ERRORS_DIR, JudgeConfig, JUDGES
from eval.llm import LLMResponse, call_llm, parse_json


# ── Error logging ───────────────────────────────────────────────────────────

def log_error(
    judge_name: str,
    prompt_id: str,
    error_type: str,
    attempts: list[str],
    extra: dict | None = None,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    entry: dict[str, Any] = {
        "prompt_id": prompt_id,
        "judge": judge_name,
        "error_type": error_type,
        "attempts": attempts,
        "timestamp": ts,
    }
    if extra:
        entry.update(extra)

    path = ERRORS_DIR / f"{judge_name}_{ts[:8]}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Consensus scoring ──────────────────────────────────────────────────────

def consensus_score(scores: list[float]) -> float:
    """Median of valid (non-zero) scores, or average if only 2 valid."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    if len(valid) == 1:
        return valid[0]
    if len(valid) == 2:
        return (valid[0] + valid[1]) / 2
    valid.sort()
    return valid[len(valid) // 2]


def check_disagreement(scores: list[float], threshold: float = 3.0) -> bool:
    """Flag if any two judges disagree by more than threshold on 0-10 scale."""
    valid = [s for s in scores if s is not None]
    if len(valid) < 2:
        return False
    return (max(valid) - min(valid)) > threshold


def fleiss_kappa(ratings_matrix: list[list[int]], n_categories: int) -> float:
    """Compute Fleiss' kappa for inter-judge agreement.

    ratings_matrix: list of items, each item is a list of category counts.
    n_categories: number of possible categories.
    """
    n_items = len(ratings_matrix)
    if n_items == 0:
        return 0.0
    n_raters = sum(ratings_matrix[0])
    if n_raters <= 1:
        return 0.0

    # P_i for each item
    p_items = []
    for row in ratings_matrix:
        total = sum(row)
        if total <= 1:
            p_items.append(0.0)
            continue
        p_i = (sum(r * r for r in row) - total) / (total * (total - 1))
        p_items.append(p_i)

    p_bar = sum(p_items) / n_items

    # P_j for each category
    col_sums = [0] * n_categories
    total_ratings = 0
    for row in ratings_matrix:
        for j, r in enumerate(row):
            col_sums[j] += r
            total_ratings += r

    p_e = sum((c / total_ratings) ** 2 for c in col_sums)

    if abs(1 - p_e) < 1e-10:
        return 1.0 if abs(p_bar - 1.0) < 1e-10 else 0.0
    return (p_bar - p_e) / (1 - p_e)


# ── Judge orchestration ────────────────────────────────────────────────────

async def judge_single(
    judge_cfg: JudgeConfig,
    system_prompt: str,
    user_prompt: str,
    prompt_id: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_parse_retries: int = 2,
) -> dict | None:
    """Call a single judge and parse JSON response. Returns parsed dict or None."""
    attempts: list[str] = []

    for attempt in range(max_parse_retries + 1):
        try:
            prefix = ""
            if attempt > 0:
                prefix = (
                    "Your previous response was not valid JSON. "
                    "Please respond with ONLY a valid JSON object.\n\n"
                )
            resp = await call_llm(
                judge_cfg,
                system_prompt,
                prefix + user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            attempts.append(resp.text)
            parsed = parse_json(resp.text)
            if parsed is not None:
                return parsed
        except Exception as e:
            attempts.append(f"ERROR: {e}")

    log_error(judge_cfg.name, prompt_id, "malformed_json", attempts)
    return None


async def judge_panel(
    system_prompt: str,
    user_prompt: str,
    prompt_id: str,
    judges: list[JudgeConfig] | None = None,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    semaphores: dict[str, asyncio.Semaphore] | None = None,
    target_provider: str | None = None,
) -> list[tuple[str, dict | None]]:
    """Run all judges in parallel, return list of (judge_name, parsed_result).

    If target_provider is set, judges from the same provider are excluded
    (recusal rule: no provider judges its own models).
    """
    if judges is None:
        judges = JUDGES

    if target_provider:
        judges = [j for j in judges if j.provider != target_provider]

    async def _run_one(cfg: JudgeConfig) -> tuple[str, dict | None]:
        if semaphores and cfg.name in semaphores:
            async with semaphores[cfg.name]:
                result = await judge_single(
                    cfg, system_prompt, user_prompt, prompt_id,
                    max_tokens=max_tokens, temperature=temperature,
                )
        else:
            result = await judge_single(
                cfg, system_prompt, user_prompt, prompt_id,
                max_tokens=max_tokens, temperature=temperature,
            )
        return cfg.name, result

    tasks = [_run_one(cfg) for cfg in judges]
    return await asyncio.gather(*tasks)
