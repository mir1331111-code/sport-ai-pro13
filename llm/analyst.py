"""llm/analyst.py — LLM-аналитик (6 провайдеров)."""
from __future__ import annotations
import hashlib, json, re
from typing import Optional

import requests

from config import CACHE_TTL, LLM_PROVIDERS
from security import cache_get, cache_put
from storage import usage


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0"})
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
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


def analyze_match(ctx: dict) -> Optional[str]:
    api_key = ctx.get("api_key")
    if not api_key:
        return None
    provider = ctx.get("provider") or "Groq (бесплатно, быстро)"
    if provider not in LLM_PROVIDERS:
        provider = "Groq (бесплатно, быстро)"
    cfg = LLM_PROVIDERS[provider]
    model = ctx.get("model") or cfg["model"]
    base = cfg["base"]

    ctx_hash = hashlib.md5(
        json.dumps(ctx, sort_keys=True, default=str).encode()).hexdigest()
    ck = f"llm_{ctx_hash}"
    cached = cache_get(ck, CACHE_TTL["llm"])
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
        r = _sess.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [
                      {"role": "system", "content": LLM_SYSTEM_PROMPT},
                      {"role": "user", "content": user_msg}],
                  "temperature": 0.3, "max_tokens": 300},
            timeout=30, proxies=NO_PROXY)
        usage.llm_increment(1)
        if r.status_code != 200:
            return None
        data = r.json()
        content = (data.get("choices", [{}])[0]
                   .get("message", {}).get("content", "")).strip()
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
                    parsed = None
        if parsed and "opinion" in parsed:
            opinion = str(parsed["opinion"]).strip()
            agree = parsed.get("agree")
            risks = parsed.get("risks")
            tag = "✅ Согласен" if agree else "🤔 Спорно"
            result = f"{tag}. {opinion}"
            if risks:
                result += f" · Риски: {risks}"
            cache_put(ck, result)
            return result
        cache_put(ck, content[:300])
        return content[:300] if content else None
    except Exception:
        return None
