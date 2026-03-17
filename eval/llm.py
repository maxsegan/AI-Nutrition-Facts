"""Unified async LLM client for calling target models and judges.

Follows patterns from Reason_AI/eval/judges.py that are proven to work.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load .env file before importing API clients
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import anthropic
import openai
from google import genai

from eval.config import JudgeConfig, TargetModelConfig


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int]
    latency_ms: int
    model: str


# ── Client factories ────────────────────────────────────────────────────────

_clients: dict[str, Any] = {}


def _get_client(provider: str, api_key_env: str, base_url: str = ""):
    key = f"{provider}:{api_key_env}:{base_url}"
    if key not in _clients:
        api_key = os.environ.get(api_key_env, "")
        if provider == "anthropic":
            _clients[key] = anthropic.AsyncAnthropic(api_key=api_key)
        elif provider == "openai":
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            _clients[key] = openai.AsyncOpenAI(**kwargs)
        elif provider == "google":
            _clients[key] = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _clients[key]


# ── Reasoning model detection ──────────────────────────────────────────────

def _is_reasoning_model(model: str) -> bool:
    """GPT-5+, o-series use reasoning tokens (max_completion_tokens, no temperature)."""
    return (
        model.startswith("gpt-5")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    )


def _needs_responses_api(model: str) -> bool:
    """Some models (gpt-5.4-pro, gpt-5.2-pro) only work via the responses API."""
    return "pro" in model and model.startswith("gpt-5")


# ── Provider-specific call functions ────────────────────────────────────────
# Following Reason_AI/eval/judges.py patterns that are proven to work.

async def _call_anthropic(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> LLMResponse:
    t0 = time.perf_counter()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )
    latency = int((time.perf_counter() - t0) * 1000)
    text = resp.content[0].text if resp.content else ""
    return LLMResponse(
        text=text,
        usage={"input_tokens": resp.usage.input_tokens,
               "output_tokens": resp.usage.output_tokens},
        latency_ms=latency,
        model=model,
    )


async def _call_openai(
    client: openai.AsyncOpenAI,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> LLMResponse:
    """Call OpenAI-compatible API. Handles reasoning models and responses API."""
    t0 = time.perf_counter()

    if _needs_responses_api(model):
        # Pro models need the responses API, not chat completions
        input_text = system + "\n\n" + "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        )
        resp = await client.responses.create(
            model=model,
            input=input_text,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        text = resp.output_text or ""
        usage = {"input_tokens": resp.usage.input_tokens or 0,
                 "output_tokens": resp.usage.output_tokens or 0}
        return LLMResponse(text=text, usage=usage, latency_ms=latency, model=model)

    # Standard chat completions
    msgs = [{"role": "system", "content": system}] + messages
    kwargs: dict[str, Any] = {"model": model, "messages": msgs}

    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = max(max_tokens, 2048)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature

    resp = await client.chat.completions.create(**kwargs)
    latency = int((time.perf_counter() - t0) * 1000)
    text = resp.choices[0].message.content or ""
    usage = {"input_tokens": resp.usage.prompt_tokens or 0,
             "output_tokens": resp.usage.completion_tokens or 0}
    return LLMResponse(text=text, usage=usage, latency_ms=latency, model=model)


async def _call_google(
    client: genai.Client,
    model: str,
    system: str,
    contents: str | list,
    max_tokens: int,
    temperature: float,
) -> LLMResponse:
    """Call Google Gemini API via sync wrapper (run_in_executor).

    Following Reason_AI pattern: combine system+user, bump max_output_tokens
    to 8192 for thinking models, use sync client in executor.
    """
    # Thinking models (Gemini 2.5+, 3+) use reasoning tokens from the output budget.
    # Budget 8K total so thinking tokens don't starve the actual response.
    effective_max = max(max_tokens, 8192)

    # If contents is a string (single-turn), combine with system
    if isinstance(contents, str):
        full_prompt = f"{system}\n\n{contents}"
    else:
        # Multi-turn: prepend system to first user message
        full_prompt = contents  # will be handled below

    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()

    if isinstance(contents, str):
        resp = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=full_prompt,
                config={"temperature": temperature, "max_output_tokens": effective_max},
            ),
        )
    else:
        # Multi-turn: pass contents list with system prepended
        multi_prompt = f"{system}\n\n{contents[0]['content']}" if contents else system
        remaining = [{"role": "model" if m["role"] == "assistant" else m["role"],
                       "parts": [{"text": m["content"]}]}
                      for m in contents[1:]] if len(contents) > 1 else []

        all_contents = [{"role": "user", "parts": [{"text": multi_prompt}]}] + remaining
        resp = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=all_contents,
                config={"temperature": temperature, "max_output_tokens": effective_max},
            ),
        )

    latency = int((time.perf_counter() - t0) * 1000)
    text = resp.text or ""
    usage = {}
    if resp.usage_metadata:
        usage = {
            "input_tokens": resp.usage_metadata.prompt_token_count or 0,
            "output_tokens": resp.usage_metadata.candidates_token_count or 0,
        }
    return LLMResponse(text=text, usage=usage, latency_ms=latency, model=model)


# ── Unified call interface ──────────────────────────────────────────────────

def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("rate_limit", "rate limit", "429", "quota",
                                   "resource_exhausted", "too many requests"))


async def call_llm(
    cfg: JudgeConfig | TargetModelConfig,
    system: str,
    user: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> LLMResponse:
    """Call an LLM (single-turn) with retry on rate-limit errors."""
    client = _get_client(cfg.provider, cfg.api_key_env,
                         getattr(cfg, "base_url", ""))

    for attempt in range(max_retries):
        try:
            if cfg.provider == "anthropic":
                return await _call_anthropic(
                    client, cfg.model, system,
                    [{"role": "user", "content": user}],
                    max_tokens, temperature)
            elif cfg.provider == "openai":
                return await _call_openai(
                    client, cfg.model, system,
                    [{"role": "user", "content": user}],
                    max_tokens, temperature)
            elif cfg.provider == "google":
                return await _call_google(
                    client, cfg.model, system, user,
                    max_tokens, temperature)
            else:
                raise ValueError(f"Unknown provider: {cfg.provider}")
        except Exception as e:
            if _is_rate_limit(e) and attempt < max_retries - 1:
                wait = 2 ** attempt + 1
                await asyncio.sleep(wait)
                continue
            raise


async def call_llm_multi_turn(
    cfg: JudgeConfig | TargetModelConfig,
    system: str,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> LLMResponse:
    """Call an LLM with a multi-turn conversation."""
    client = _get_client(cfg.provider, cfg.api_key_env,
                         getattr(cfg, "base_url", ""))

    for attempt in range(max_retries):
        try:
            if cfg.provider == "anthropic":
                return await _call_anthropic(
                    client, cfg.model, system, messages,
                    max_tokens, temperature)
            elif cfg.provider == "openai":
                return await _call_openai(
                    client, cfg.model, system, messages,
                    max_tokens, temperature)
            elif cfg.provider == "google":
                return await _call_google(
                    client, cfg.model, system, messages,
                    max_tokens, temperature)
            else:
                raise ValueError(f"Unknown provider: {cfg.provider}")
        except Exception as e:
            if _is_rate_limit(e) and attempt < max_retries - 1:
                wait = 2 ** attempt + 1
                await asyncio.sleep(wait)
                continue
            raise


# ── JSON parsing helpers ────────────────────────────────────────────────────

def parse_json(text: str) -> dict | None:
    """Extract a JSON object from LLM response text."""
    text = (text or "").strip()
    if not text:
        return None

    # Strip thinking tags (Qwen3, DeepSeek, etc.)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Find first JSON object in text
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None
