"""llm/analyst.py — LLM-аналитик с логированием и fallback-парсингом."""
from __future__ import annotations
import hashlib, json, re, sys
from typing import Optional

import requests

from config import CACHE_TTL, LLM_PROVIDERS
from security import cache_get, cache_put
from storage import usage


def _log(msg: str):
    """Логирует в stdout — видно в Streamlit Cloud → Manage app → Logs."""
    try:
        print(f"[LLM] {msg}", file=sys.stdout, flush=True)
    except Exception:
        pass


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0"})
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()
NO_PROXY = {"http": None, "https": None, "all": None}


LLM_SYSTEM_PROMPT = """Ты — эксперт-аналитик футбола. Отвечай на русском.
Тебе дают данные о матче: команды, лига, модельные xG, вероятности, форма.
Твоя задача:
1. Дать КРАТКОЕ мнение (1-3 предложения) по матчу.
2. Указать главные факторы (форма, мотивация, стиль).
3. Предупредить о рисках если есть.

Формат ответа СТРОГО: JSON с полями:
{"opinion": "краткое мнение 1-3 предложения", "agree": true/false, "risks": "короткий список рисков или пусто"}

НЕ добавляй пояснений, markdown, код. Только валидный JSON."""


def _parse_llm_response(content: str) -> Optional[str]:
    """Парсит ответ LLM. Три уровня fallback."""
    if not content:
        return None
    content = content.strip()

    # Убираем markdown-обёртку ```json ... ```
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()

    # Попытка 1: чистый JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "opinion" in parsed:
            opinion = str(parsed["opinion"]).strip()
            if opinion:
                agree = parsed.get("agree")
                risks = parsed.get("risks")
                if agree is True:
                    tag = "✅ Согласен"
                elif agree is False:
                    tag = "🤔 Спорно"
                else:
                    tag = "ℹ️"
                result = f"{tag}. {opinion}"
                if risks:
                    result += f" · Риски: {risks}"
                return result
    except Exception:
        pass

    # Попытка 2: JSON внутри текста
    m = re.search(r"\{[^{}]*\"opinion\"[^{}]*\}", content)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if "opinion" in parsed:
                opinion = str(parsed["opinion"]).strip()
                if opinion:
                    agree = parsed.get("agree")
                    risks = parsed.get("risks")
                    if agree is True:
                        tag = "✅ Согласен"
                    elif agree is False:
                        tag = "🤔 Спорно"
                    else:
                        tag = "ℹ️"
                    result = f"{tag}. {opinion}"
                    if risks:
                        result += f" · Риски: {risks}"
                    return result
        except Exception:
            pass

    # Попытка 3: plain text — возвращаем как есть (обрезаем до 300)
    if len(content) > 10:
        return content[:300]

    return None


def analyze_match(ctx: dict) -> Optional[str]:
    """Анализирует матч через LLM. Возвращает строку или None."""
    api_key = ctx.get("api_key")
    if not api_key:
        _log("ERROR: api_key пустой")
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
    ck = f"llm_v3_{ctx_hash}"
    cached = cache_get(ck, CACHE_TTL["llm"])
    if cached is not None:
        _log(f"CACHE HIT: {ctx_hash[:8]} → {str(cached)[:50]}")
        return cached or None

    if usage.llm_remaining() <= 0:
        _log("ERROR: лимит LLM исчерпан")
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
        f"Форма: хозяева {ctx.get('fh','—')}, гости {ctx.get('fa','—')}\n"
        f"Прогноз модели: {ctx.get('pick','')} с "
        f"P={ctx.get('prob',0)*100:.0f}%\n"
    )

    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }

    _log(f"REQUEST: provider={provider}, model={model}")

    try:
        r = _sess.post(url, headers=headers, json=body,
                       timeout=30, proxies=NO_PROXY)
        usage.llm_increment(1)

        _log(f"RESPONSE: HTTP {r.status_code}, len={len(r.text)}")

        if r.status_code != 200:
            _log(f"ERROR HTTP {r.status_code}: {r.text[:300]}")
            cache_put(ck, None)
            return None

        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            _log(f"ERROR: нет choices: {str(data)[:300]}")
            cache_put(ck, None)
            return None

        content = (choices[0].get("message") or {}).get("content", "")
        if not content:
            _log("ERROR: content пустой")
            cache_put(ck, None)
            return None

        _log(f"CONTENT: {content[:200]}")

        result = _parse_llm_response(content)
        if result:
            _log(f"PARSED OK: {result[:100]}")
            cache_put(ck, result)
            return result
        else:
            _log(f"ERROR: не удалось распарсить: {content[:200]}")
            cache_put(ck, None)
            return None

    except requests.exceptions.Timeout:
        _log("ERROR: timeout 30s")
        return None
    except requests.exceptions.ConnectionError as e:
        _log(f"ERROR: connection failed: {str(e)[:200]}")
        return None
    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {str(e)[:200]}")
        return None
