#!/usr/bin/env python3
"""AI Nutrition Facts — Benchmark Viewer.

Reads results from both completed JSON files and in-progress JSONL cache
files, so results appear live as the evaluation runs.

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


# ── Model family / logo mapping ─────────────────────────────────────────────

MODEL_FAMILIES = {
    "opus-4.6": "anthropic",
    "sonnet-4.6": "anthropic",
    "haiku-4.5": "anthropic",
    "gpt-5.4-pro": "openai",
    "gpt-5.4": "openai",
    "gpt-5-mini": "openai",
    "gpt-5-nano": "openai",
    "gemini-3-pro": "google",
    "gemini-3.1-pro": "google",
    "gemini-3-flash": "google",
    "qwen3.5-397b": "qwen",
    "qwen3.5-27b": "qwen",
    "qwen3.5-9b": "qwen",
    "deepseek-v3.2": "deepseek",
    "deepseek-v3.2-speciale": "deepseek",
    "llama-4-maverick": "meta",
    "llama-4-scout": "meta",
    "grok-4.1": "xai",
    "grok-4.20": "xai",
    "grok-4.1-fast": "xai",
    "mistral-large-3": "mistral",
    "ministral-3-14b": "mistral",
    "kimi-k2.5": "moonshot",
}

FAMILY_COLORS = {
    "anthropic": "#D4A27F",
    "openai": "#10A37F",
    "google": "#4285F4",
    "qwen": "#6F3CE8",
    "deepseek": "#4D6BFE",
    "meta": "#0668E1",
    "xai": "#111111",
    "mistral": "#F54E42",
    "moonshot": "#1A1A2E",
}

FAMILY_DISPLAY = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "qwen": "Alibaba / Qwen",
    "deepseek": "DeepSeek",
    "meta": "Meta",
    "xai": "xAI",
    "mistral": "Mistral",
    "moonshot": "Moonshot",
}

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
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def _live_progress() -> dict:
    """Scan cache files to build a live progress report."""
    progress: dict[str, dict[str, int]] = {}

    for f in CACHE_DIR.glob("syco_*.jsonl"):
        parts = f.stem.split("_", 1)  # syco_<subtype>_<model>
        rest = parts[1] if len(parts) > 1 else ""
        # Find the model name: everything after the subtype
        for subtype in ["flattery", "opinion_mirroring", "influencability",
                        "conflict_avoidance", "demand_compliance"]:
            if rest.startswith(subtype + "_"):
                model = rest[len(subtype) + 1:]
                if model not in progress:
                    progress[model] = {}
                count = sum(1 for line in f.read_text().splitlines()
                            if line.strip())
                progress[model][subtype] = count
                break

    for f in CACHE_DIR.glob("political_*.jsonl"):
        rest = f.stem[len("political_"):]
        # Axes: economic, governance, etc. or geopolitical
        from eval.config import POLITICAL_AXES
        axis_ids = [a["id"] for a in POLITICAL_AXES] + ["geopolitical"]
        for axis_id in axis_ids:
            if rest.startswith(axis_id + "_"):
                model = rest[len(axis_id) + 1:]
                if model not in progress:
                    progress[model] = {}
                count = sum(1 for line in f.read_text().splitlines()
                            if line.strip())
                progress[model][f"pol_{axis_id}"] = count
                break

    return progress


def _all_models() -> list[str]:
    """Get all model names from any available results or cache."""
    models = set()
    for pattern in ["sycophancy_*.json", "political_*.json", "supplementary_*.json"]:
        for f in RESULTS_DIR.glob(pattern):
            name = f.stem
            for prefix in ["sycophancy_", "political_", "supplementary_"]:
                if name.startswith(prefix) and name != prefix + "all":
                    models.add(name[len(prefix):])
    # Also scan cache for in-progress models
    progress = _live_progress()
    models.update(progress.keys())
    return sorted(models)


def _sycophancy_badge(score: float) -> str:
    if score < 3.0: return "badge-good"
    if score < 6.0: return "badge-ok"
    return "badge-bad"


def _political_badge(score: float) -> str:
    a = abs(score)
    if a < 0.2: return "badge-good"
    if a < 0.5: return "badge-ok"
    return "badge-bad"


def _model_family(model_name: str) -> str:
    return MODEL_FAMILIES.get(model_name, "")


def _model_logo(model_name: str) -> str:
    family = MODEL_FAMILIES.get(model_name, "")
    if family:
        return f"/static/logos/{family}.svg"
    return ""


def _model_color(model_name: str) -> str:
    family = MODEL_FAMILIES.get(model_name, "")
    return FAMILY_COLORS.get(family, "#888")


def _family_display(model_name: str) -> str:
    family = MODEL_FAMILIES.get(model_name, "")
    return FAMILY_DISPLAY.get(family, family)


app.jinja_env.globals.update(
    sycophancy_badge=_sycophancy_badge,
    political_badge=_political_badge,
    model_family=_model_family,
    model_logo=_model_logo,
    model_color=_model_color,
    family_display=_family_display,
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
    sycophancy = _load_result("sycophancy_all.json") or {}
    political = _load_result("political_all.json") or {}
    supplementary = _load_result("supplementary_all.json") or {}
    models = _all_models()
    progress = _live_progress()

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
                           prompt_counts=prompt_counts,
                           progress=progress)


@app.route("/label/<model_name>")
def nutrition_label(model_name: str):
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
    data = _load_result("sycophancy_all.json") or {}
    prompts = {}
    for name in ["flattery", "opinion_mirroring", "influencability",
                  "conflict_avoidance", "demand_compliance"]:
        prompts[name] = _load_prompts("sycophancy", name)

    return render_template("sycophancy.html", data=data, prompts=prompts)


@app.route("/sycophancy/<subtype>/<model_name>")
def sycophancy_detail(subtype: str, model_name: str):
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
    data = _load_result("political_all.json") or {}
    return render_template("political.html", data=data)


@app.route("/political/<axis_id>/<model_name>")
def political_detail(axis_id: str, model_name: str):
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
    return render_template("rubrics.html")


@app.route("/compare")
def compare():
    sycophancy = _load_result("sycophancy_all.json") or {}
    political = _load_result("political_all.json") or {}
    supplementary = _load_result("supplementary_all.json") or {}

    return render_template("compare.html",
                           sycophancy=sycophancy,
                           political=political,
                           supplementary=supplementary)


# ── API routes ───────────────────────────────────────────────────────────────

@app.route("/api/progress")
def api_progress():
    """Live progress data for auto-refresh."""
    return jsonify(_live_progress())


@app.route("/api/cache/<category>/<subtype>/<model_name>")
def api_cache(category: str, subtype: str, model_name: str):
    items = _load_cache(subtype, model_name)
    return jsonify(items)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5010))
    print(f"AI Nutrition Facts viewer at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
