#!/usr/bin/env python3
"""AI Nutrition Facts — Benchmark Viewer.

Hosted via Tailscale serve on port 5010.

Usage:
    python -m bench.app
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, render_template, jsonify, request

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval" / "results"
CACHE_DIR = REPO_ROOT / "eval" / "cache"
PROMPTS_DIR = REPO_ROOT / "eval" / "prompts"

app = Flask(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _load_result(filename: str) -> dict | None:
    return _load_json(RESULTS_DIR / filename)


def _load_prompts(category: str, name: str) -> dict | None:
    return _load_json(PROMPTS_DIR / category / f"{name}.json")


def _load_cache(subtype: str, target: str) -> list[dict]:
    """Load per-item results from JSONL cache."""
    path = CACHE_DIR / f"syco_{subtype}_{target}.jsonl"
    if not path.exists():
        path = CACHE_DIR / f"political_{subtype}_{target}.jsonl"
    if not path.exists():
        return []
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def _all_models() -> list[str]:
    """Get all model names from any available results."""
    models = set()
    for pattern in ["sycophancy_*.json", "political_*.json", "supplementary_*.json"]:
        for f in RESULTS_DIR.glob(pattern):
            name = f.stem
            for prefix in ["sycophancy_", "political_", "supplementary_"]:
                if name.startswith(prefix) and name != prefix + "all":
                    models.add(name[len(prefix):])
    return sorted(models)


def _sycophancy_badge(score: float) -> str:
    if score < 3.0: return "badge-good"
    if score < 6.0: return "badge-ok"
    return "badge-bad"


def _political_badge(score: float) -> str:
    """Badge for position scores (-1 to +1). Neutral near 0."""
    a = abs(score)
    if a < 0.2: return "badge-good"
    if a < 0.5: return "badge-ok"
    return "badge-bad"


app.jinja_env.globals.update(
    sycophancy_badge=_sycophancy_badge,
    political_badge=_political_badge,
    abs=abs,
    min=min,
    max=max,
    round=round,
    len=len,
    isinstance=isinstance,
    enumerate=enumerate,
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Dashboard: overview of all models and stats."""
    sycophancy = _load_result("sycophancy_all.json") or {}
    political = _load_result("political_all.json") or {}
    supplementary = _load_result("supplementary_all.json") or {}
    models = _all_models()

    # Count test items
    prompt_counts = {}
    for cat in ["sycophancy", "political"]:
        for f in (PROMPTS_DIR / cat).glob("*.json"):
            data = _load_json(f)
            if data and "items" in data:
                prompt_counts[f.stem] = len(data["items"])
            elif data and "categories" in data:
                prompt_counts[f.stem] = sum(len(v) for v in data["categories"].values())

    return render_template("index.html",
                           models=models,
                           sycophancy=sycophancy,
                           political=political,
                           supplementary=supplementary,
                           prompt_counts=prompt_counts)


@app.route("/label/<model_name>")
def nutrition_label(model_name: str):
    """The nutrition facts label for a single model."""
    sycophancy = _load_result(f"sycophancy_{model_name}.json")
    political = _load_result(f"political_{model_name}.json")
    supplementary = _load_result(f"supplementary_{model_name}.json")

    return render_template("label.html",
                           model_name=model_name,
                           sycophancy=sycophancy,
                           political=political,
                           supplementary=supplementary)


@app.route("/sycophancy")
def sycophancy_overview():
    """Sycophancy evaluation: summary across all models."""
    data = _load_result("sycophancy_all.json") or {}
    prompts = {}
    for name in ["flattery", "opinion_mirroring", "influencability",
                  "conflict_avoidance", "demand_compliance"]:
        prompts[name] = _load_prompts("sycophancy", name)

    return render_template("sycophancy.html", data=data, prompts=prompts)


@app.route("/sycophancy/<subtype>/<model_name>")
def sycophancy_detail(subtype: str, model_name: str):
    """Per-prompt results for a sycophancy subtype and model."""
    items = _load_cache(subtype, model_name)
    prompts_data = _load_prompts("sycophancy", subtype)
    aggregate = _load_result(f"sycophancy_{model_name}.json")
    subtype_score = None
    if aggregate and "subtypes" in aggregate:
        subtype_score = aggregate["subtypes"].get(subtype)

    return render_template("sycophancy_detail.html",
                           subtype=subtype,
                           model_name=model_name,
                           items=items,
                           prompts_data=prompts_data,
                           subtype_score=subtype_score)


@app.route("/political")
def political_overview():
    """Political bias evaluation: summary across all models."""
    data = _load_result("political_all.json") or {}
    return render_template("political.html", data=data)


@app.route("/political/<axis_id>/<model_name>")
def political_detail(axis_id: str, model_name: str):
    """Per-prompt results for a political axis and model."""
    items = _load_cache(axis_id, model_name)
    prompts_data = _load_prompts("political", axis_id)
    aggregate = _load_result(f"political_{model_name}.json")
    axis_score = None
    if aggregate and "domestic_axes" in aggregate:
        axis_score = aggregate["domestic_axes"].get(axis_id)

    return render_template("political_detail.html",
                           axis_id=axis_id,
                           model_name=model_name,
                           items=items,
                           prompts_data=prompts_data,
                           axis_score=axis_score)


@app.route("/rubrics")
def rubrics():
    """Display all scoring rubrics."""
    return render_template("rubrics.html")


@app.route("/compare")
def compare():
    """Side-by-side model comparison."""
    sycophancy = _load_result("sycophancy_all.json") or {}
    political = _load_result("political_all.json") or {}
    supplementary = _load_result("supplementary_all.json") or {}

    return render_template("compare.html",
                           sycophancy=sycophancy,
                           political=political,
                           supplementary=supplementary)


# ── API routes (for AJAX) ────────────────────────────────────────────────────

@app.route("/api/cache/<category>/<subtype>/<model_name>")
def api_cache(category: str, subtype: str, model_name: str):
    """Return cached per-item results as JSON."""
    items = _load_cache(subtype, model_name)
    return jsonify(items)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5010))
    print(f"AI Nutrition Facts viewer at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
