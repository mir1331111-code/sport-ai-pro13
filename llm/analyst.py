"""llm/analyst.py — LLM-аналитик (OpenAI-совместимые + нативный Gemini)."""
from __future__ import annotations
import hashlib, json, re
from typing import Optional

import requests
import streamlit as st

from config import CACHE_TTL, LLM_PROVIDERS
from security import cache_get, cache_put
from storage import usage

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0"})
    s.trust_env = False
    s.proxies = {"http": None, "https": None, "all": None}
    return s

_sess = _session()
NO_PROXY = {"http": None, "https": None, "all": None}

LLM_SYSTEM_PROMPT = """Ты — эксперт-аналитик футбола и ставок. Отвечай на русском.
Тебе дают данные о матче: команды, лига, модельные xG, вероятности, форма, вердикт ML-модели.
Твоя задача:
1. Дать КРАТКОЕ мнение (1-3 предложения) по этому матчу.
2. Согласиться или возразить модели, если видишь что-то упущенное.
3. Указать главные факторы (форма, травмы, мотивация, стиль).
4. Предупредить о рисках если есть.

Формат ответа СТРОГО: JSON с полями:
{"opinion": "краткое мнение 1-3 предложения", "agree": true/false, "risks": "короткий список рисков или пусто"}

НЕ добавляй пояснений, markdown, код. Только валидный JSON."""


def _call_openai_compatible(base, model, api_key, system_msg, user_msg):
    """Вызов OpenAI-совместимого API (Groq, DeepSeek, OpenRouter, Grok, OpenAI)."""
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    r = _sess.post(url, headers=headers, json=payload, timeout=30, proxies=NO_PROXY)
    if r.status_code != 200:
        st.warning(f"🤖 LLM HTTP {r.status_code}: {r.text[:300]}")
        return None
    data = r.json()
    content = (data.get("choices", [{}])[0]
               .get("message", {}).get("content", "")).strip()
    return content if content else None


def _call_gemini_native(model, api_key, system_msg, user_msg):
    """Вызов нативного Gemini API."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    payload = {
        "contents": [{"parts": [{"text": f"{system_msg}\n\n{user_msg}"}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 300,
        },
    }
    headers = {"Content-Type": "application/json"}
    r = _sess.post(url, headers=headers, json=payload, timeout=30, proxies=NO_PROXY)
    if r.status_code != 200:
        st.warning(f"🤖 Gemini HTTP {r.status_code}: {r.text[:300]}")
        return None
    data = r.json()
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return content if content else None
    except (KeyError, IndexError):
        st.warning(f"🤖 Gemini пустой ответ: {json.dumps(data)[:300]}")
        return None


def _parse_response(content: str) -> Optional[str]:
    """Парсит JSON-ответ LLM."""
    if not content:
        return None
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    parsed = None
    try:
        parsed = json.loads(content)
    except Exception:
        m = re.search(r"\{[^{}]*\}", content)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                pass
    if parsed and "opinion" in parsed:
        opinion = str(parsed["opinion"]).strip()
        agree = parsed.get("agree")
        risks = parsed.get("risks")
        tag = "✅ Согласен" if agree else "🤔 Спорно"
        result = f"{tag}. {opinion}"
        if risks:
            result += f" · Риски: {risks}"
        return result
    return content[:300] if content else None


def analyze_match(ctx: dict) -> Optional[str]:
    api_key = ctx.get("api_key", "").strip()
    if not api_key:
        return None

    provider = ctx.get("provider") or "Groq (бесплатно, быстро)"
    if provider not in LLM_PROVIDERS:
        provider = "Groq (бесплатно, быстро)"
    cfg = LLM_PROVIDERS[provider]
    model = ctx.get("model") or cfg["model"]
    base = cfg["base"]

    # Кэш
    ctx_hash = hashlib.md5(
        json.dumps(ctx, sort_keys=True, default=str).encode()).hexdigest()
    ck = f"llm_{ctx_hash}"
    cached = cache_get(ck, CACHE_TTL.get("llm", 86400 * 3))
    if cached is not None:
        return cached or None

    if usage.llm_remaining() <= 0:
        return None

    user_msg = (
        f"Матч: {ctx.get('home','')} — {ctx.get('away','')}\n"
        f"Лига: {ctx.get('league','')}\n"
        f"Дата: {ctx.get('date','')}\n"
        f"Модельный xG: хозяева {ctx.get('lam_h',0):.2f}, "
        f"гости {ctx.get('lam_a',0):.2f}\n"
        f"Вероятности: П1 {ctx.get('p1',0)*100:.0f}%, "
        f"X {ctx.get('px',0)*100:.0f}%, П2 {ctx.get('p2',0)*100:.0f}%\n"
        f"ТБ 2.5: {ctx.get('over',0)*100:.0f}%, "
        f"BTTS: {ctx.get('btts',0)*100:.0f}%\n"
        f"Форма хозяев: {ctx.get('fh','—')}, гостей: {ctx.get('fa','—')}\n"
        f"Вердикт ML-модели: {ctx.get('pick','')} с "
        f"P={ctx.get('prob',0)*100:.0f}% "
        f"(уверенность: {ctx.get('confidence','')})\n"
        f"EV: {ctx.get('ev',0)*100:+.1f}%\n"
    )

    try:
        # Gemini — нативный формат
        if "gemini" in provider.lower() or "google" in provider.lower():
            st.caption(f"🤖 Gemini native → {model}")
            content = _call_gemini_native(model, api_key, LLM_SYSTEM_PROMPT, user_msg)
        else:
            # Все остальные — OpenAI-совместимый формат
            st.caption(f"🤖 OpenAI-compat → {provider} / {model}")
            content = _call_openai_compatible(base, model, api_key, LLM_SYSTEM_PROMPT, user_msg)

        usage.llm_increment(1)

        if not content:
            return None

        result = _parse_response(content)
        if result:
            cache_put(ck, result)
        return result

    except requests.exceptions.Timeout:
        st.warning("🤖 LLM: таймаут (30 сек)")
        return None
    except requests.exceptions.ConnectionError as e:
        st.warning(f"🤖 LLM: ошибка соединения — {str(e)[:150]}")
        return None
    except Exception as e:
        st.warning(f"🤖 LLM ошибка: {type(e).__name__}: {str(e)[:200]}")
        return None
