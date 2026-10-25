"""Blind, evidence-limited LLM opinion for football fixtures."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Optional

import requests

from config import CACHE_TTL, LLM_PROVIDERS
from security import cache_get, cache_put
from storage import usage


def _log(msg: str):
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
NO_PROXY = {"http": None, "https": None}

LLM_SYSTEM_PROMPT = """Ты — независимый футбольный аналитик. Отвечай по-русски и не пытайся подтвердить чужое мнение.
ВАЖНО: тебе НЕ сообщают прогноз основной статистической модели, её вероятности, выбранный исход, букмекерские коэффициенты или рекомендацию. Не угадывай и не восстанавливай их. Сначала сформируй собственную оценку только по фактам во входных данных.
Используй только предоставленные данные: ожидаемые голы хозяев/гостей (λ) и последовательность последних результатов команд. Не выдумывай травмы, составы, мотивацию, стиль, личные встречи, погоду или новости — их нет во входных данных. При нехватке информации прямо укажи это и снизь качество данных.
Дай самостоятельное распределение вероятностей 1X2: П1, X, П2 в процентах от 0 до 100, сумма ровно 100. Выбранный independent_pick обязан совпадать с исходом с наибольшей вероятностью. Не завышай уверенность по столь ограниченным данным; «high» используй только при действительно сильной и достаточной доказательной базе, которая здесь обычно отсутствует.
Кратко объясни, что именно поддерживает прогноз и что может его опровергнуть. Не советуй ставить и не обещай прибыль.
Верни строго JSON без markdown:
{"independent_pick":"П1|X|П2","probabilities":{"П1":40,"X":30,"П2":30},"opinion":"1-3 коротких предложения с факторами и ограничениями данных","risks":"конкретные риски или недостаток данных","data_quality":"low|medium|high"}"""


def _as_number(value) -> float:
    try:
        number = float(str(value).strip().replace(",", "."))
        return number if number == number and abs(number) != float("inf") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pick_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"П1", "1", "HOME", "HOME_WIN"}:
        return "П1"
    if text in {"X", "Н", "DRAW", "TIE"}:
        return "X"
    if text in {"П2", "2", "AWAY", "AWAY_WIN"}:
        return "П2"
    return ""


def _response_json(content: str) -> Optional[dict]:
    if not content:
        return None
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_llm_response(content: str, model_pick_code: str = "") -> Optional[str]:
    """Validate an independent 1X2 estimate and compare it outside the LLM prompt."""
    parsed = _response_json(content)
    if not parsed or not str(parsed.get("opinion") or "").strip():
        return None

    raw = parsed.get("probabilities")
    if not isinstance(raw, dict):
        return None
    aliases = {"П1": ("П1", "p1", "home"), "X": ("X", "px", "draw"),
               "П2": ("П2", "p2", "away")}
    probs = {}
    for code, keys in aliases.items():
        val = next((raw[k] for k in keys if k in raw), None)
        try:
            number = float(val)
        except (TypeError, ValueError):
            return None
        if not 0 <= number <= 100:
            return None
        probs[code] = number

    total = sum(probs.values())
    if total <= 0:
        return None

    normalized = {code: int(round(value * 100.0 / total))
                  for code, value in probs.items()}
    drift = 100 - sum(normalized.values())
    largest = max(probs, key=probs.get)
    normalized[largest] += drift
    order = ("П1", "X", "П2")
    chosen = max(order, key=lambda code: (normalized[code], -order.index(code)))

    labels = {"П1": "победа хозяев (П1)", "X": "ничья (X)", "П2": "победа гостей (П2)"}
    quality_raw = str(parsed.get("data_quality") or "low").strip().lower()
    quality = {"high": "высокое", "medium": "среднее", "low": "низкое"}.get(quality_raw, "низкое")
    opinion = re.sub(r"\s+", " ", str(parsed.get("opinion") or "").strip())[:600]
    risks = re.sub(r"\s+", " ", str(parsed.get("risks") or "").strip())[:300]

    result = (
        f"САМОСТОЯТЕЛЬНЫЙ ПРОГНОЗ ИИ: {labels[chosen]}. "
        f"Оценка ИИ: П1 {normalized['П1']}% · X {normalized['X']}% · П2 {normalized['П2']}%. "
        f"Качество исходных данных: {quality}. {opinion}"
    )
    if risks:
        result += f" Риски: {risks}"

    model_pick = _pick_code(model_pick_code)
    if model_pick:
        if chosen == model_pick:
            result += f" Сверка с основной моделью: исход совпадает ({model_pick}); это не доказывает правильность прогноза."
        else:
            result += (
                f" РАСХОЖДЕНИЕ с основной моделью: ИИ выбирает {chosen}, "
                f"основная модель — {model_pick}. Причину расхождения стоит проверить отдельно."
            )
    return result


def analyze_match(ctx: dict) -> Optional[str]:
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

    cache_context = {
        key: ctx.get(key) for key in (
            "home", "away", "league", "date", "lam_h", "lam_a", "fh", "fa",
            "model_pick_code", "provider", "model",
        )
    }
    cache_context.update({"provider_resolved": provider, "model_resolved": model})
    ctx_hash = hashlib.sha256(
        json.dumps(cache_context, sort_keys=True, default=str).encode()
    ).hexdigest()
    ck = f"llm_v5_blind_{ctx_hash}"
    cached = cache_get(ck, CACHE_TTL["llm"])
    if cached is not None:
        _log(f"CACHE HIT: {ctx_hash[:8]}")
        return cached or None

    if usage.llm_remaining() <= 0:
        _log("ERROR: лимит LLM исчерпан")
        return None

    # Deliberately omit model probabilities, pick, confidence, EV and all bookmaker odds.
    user_msg = (
        f"Матч: {ctx.get('home', '—')} — {ctx.get('away', '—')}\n"
        f"Лига: {ctx.get('league', '—')}\n"
        f"Дата: {ctx.get('date', '—')}\n"
        f"Ожидаемые голы по предматчевой статистике (λ): хозяева "
        f"{_as_number(ctx.get('lam_h')):.2f}, гости {_as_number(ctx.get('lam_a')):.2f}\n"
        f"Последние результаты (В — выигрыш, Н — ничья, П — поражение): "
        f"хозяева {ctx.get('fh') or 'нет данных'}, гости {ctx.get('fa') or 'нет данных'}\n"
        "Сформируй собственный вероятностный прогноз 1X2, не ссылаясь на отсутствующие данные."
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
        "temperature": 0.4,
        "max_tokens": 650,
    }

    _log(f"REQUEST: provider={provider}, model={model}, mode=blind-independent-1x2")
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

        _log(f"CONTENT: {content[:300]}")
        result = _parse_llm_response(content, ctx.get("model_pick_code", ""))
        if result:
            _log(f"PARSED OK: {result[:120]}")
            cache_put(ck, result)
            return result

        _log("ERROR: ответ не содержит валидного независимого прогноза 1X2")
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
