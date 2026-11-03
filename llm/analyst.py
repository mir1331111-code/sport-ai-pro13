"""Blind, evidence-limited LLM opinion for football fixtures."""
from __future__ import annotations

import hashlib
import json
import math
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

LLM_SYSTEM_PROMPT = """Ты — независимый футбольный аналитик. Отвечай по-русски и оценивай НЕ только победителя.
Ты не видишь прогноз основной модели, её λ/xG, её вероятности, выбранную ставку, EV и букмекерские коэффициенты. Не подстраивайся под них и не угадывай линию букмекера.
Используй только фактические исторические показатели и размеры выборок, переданные во входных данных. Не выдумывай травмы, составы, мотивацию, стиль, личные встречи, погоду, владение, удары или новости, если таких данных нет.
Обязательные рынки: П1/X/П2. Дополнительные рынки оценивай независимо:
1) тоталы голов ТБ 1.5, 2.5, 3.5, 4.5 и обе забьют Да;
2) угловые ТБ 8.5, 9.5, 10.5, 11.5;
3) карточки ТБ 2.5, 3.5, 4.5, 5.5.
Для каждого дополнительного рынка используй отдельные переданные фактические данные. Если выборки нет или она мала, не угадывай вероятность: верни null для полей этого рынка и объясни причину в risks. Не подменяй статистику угловых голами или средними по чужой лиге. Вероятности Over должны быть монотонными: с ростом линии вероятность не растёт.
expected_* — прогнозируемое среднее значение, не коэффициент букмекера. Все вероятности выражай числами от 0 до 100. Не советуй ставить и не обещай прибыль.
Ответ строго JSON без markdown по схеме:
{"independent_pick":"П1|X|П2","probabilities":{"П1":40,"X":30,"П2":30},"goal_markets":{"expected_home_goals":1.4,"expected_away_goals":1.1,"over_1_5":78,"over_2_5":56,"over_3_5":34,"over_4_5":17,"btts_yes":52},"corner_markets":{"expected_home_corners":5.2,"expected_away_corners":4.1,"over_8_5":61,"over_9_5":52,"over_10_5":40,"over_11_5":29},"card_markets":{"expected_home_cards":2.1,"expected_away_cards":2.0,"over_2_5":77,"over_3_5":62,"over_4_5":43,"over_5_5":25},"opinion":"краткий общий разбор факторов","risks":"недостаток данных и ограничения по рынкам","data_quality":"low|medium|high"}"""


def _as_number(value) -> float:
    try:
        number = float(str(value).strip().replace(",", "."))
        return number if number == number and abs(number) != float("inf") else 0.0
    except (TypeError, ValueError):
        return 0.0


MIN_TEAM_VENUE_SAMPLE = 5


def has_sufficient_history(features: object) -> bool:
    """Require real venue-specific history before asking an LLM for probabilities."""
    if not isinstance(features, dict):
        return False
    try:
        home_sample = int(features.get("home_home_sample") or 0)
        away_sample = int(features.get("away_away_sample") or 0)
    except (TypeError, ValueError):
        return False
    if min(home_sample, away_sample) < MIN_TEAM_VENUE_SAMPLE:
        return False
    for name in (
        "home_scored_at_home_avg", "home_conceded_at_home_avg",
        "away_scored_away_avg", "away_conceded_away_avg",
    ):
        try:
            value = float(features.get(name))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value) or value < 0:
            return False
    return True


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


def _append_model_comparison(base_result: str, model_pick_code: str = "") -> str:
    """Attach comparison after the blind LLM answer, never inside its prompt/cache."""
    model_pick = _pick_code(model_pick_code)
    if not base_result or not model_pick:
        return base_result or ""
    match = re.search(
        r"Оценка ИИ:\s*П1\s+(\d+(?:\.\d+)?)%\s*[·|]\s*X\s+"
        r"(\d+(?:\.\d+)?)%\s*[·|]\s*П2\s+(\d+(?:\.\d+)?)%",
        base_result,
        flags=re.IGNORECASE,
    )
    if not match:
        return base_result
    probs = {
        "П1": float(match.group(1)),
        "X": float(match.group(2)),
        "П2": float(match.group(3)),
    }
    order = ("П1", "X", "П2")
    ai_pick = max(order, key=lambda code: (probs[code], -order.index(code)))
    if ai_pick == model_pick:
        return base_result + (
            f" Сверка с основной моделью: исход совпадает ({model_pick}); "
            "это не доказывает правильность прогноза."
        )
    return base_result + (
        f" РАСХОЖДЕНИЕ с основной моделью: ИИ выбирает {ai_pick}, "
        f"основная модель — {model_pick}. Причину расхождения стоит проверить отдельно."
    )


def _parse_llm_response(
    content: str, model_pick_code: str = "", quality_cap: str = "medium"
) -> Optional[str]:
    """Validate the independent probability distribution; the chosen pick is never trusted blindly."""
    parsed = _response_json(content)
    if not parsed or not str(parsed.get("opinion") or "").strip():
        return None
    raw = parsed.get("probabilities")
    if not isinstance(raw, dict):
        return None
    aliases = {
        "П1": ("П1", "p1", "home"),
        "X": ("X", "px", "draw"),
        "П2": ("П2", "p2", "away"),
    }
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
    normalized = {
        code: int(round(value * 100.0 / total))
        for code, value in probs.items()
    }
    drift = 100 - sum(normalized.values())
    largest = max(probs, key=probs.get)
    normalized[largest] += drift
    order = ("П1", "X", "П2")
    chosen = max(order, key=lambda code: (normalized[code], -order.index(code)))
    labels = {
        "П1": "победа хозяев (П1)",
        "X": "ничья (X)",
        "П2": "победа гостей (П2)",
    }
    quality_raw = str(parsed.get("data_quality") or "low").strip().lower()
    quality_rank = {"low": 0, "medium": 1, "high": 2}
    cap = quality_cap if quality_cap in quality_rank else "low"
    effective_quality = min(
        quality_rank.get(quality_raw, 0), quality_rank[cap]
    )
    quality = ("низкое", "среднее", "высокое")[effective_quality]
    opinion = re.sub(r"\s+", " ", str(parsed.get("opinion") or "").strip())[:600]
    risks = re.sub(r"\s+", " ", str(parsed.get("risks") or "").strip())[:300]
    result = (
        f"САМОСТОЯТЕЛЬНЫЙ ПРОГНОЗ ИИ: {labels[chosen]}. "
        f"Оценка ИИ: П1 {normalized['П1']}% · X {normalized['X']}% · "
        f"П2 {normalized['П2']}%. "
        f"Качество исходных данных: {quality}. {opinion}"
    )
    if risks:
        result += f" Риски: {risks}"
    return _append_model_comparison(result, model_pick_code)
def _valid_market_section(
    raw: object,
    *,
    expected_keys: tuple[str, str],
    probability_keys: tuple[str, ...],
    expected_max: float,
    features: dict,
    market_name: str,
) -> dict | None:
    """Validate a Groq count-market section against available real history."""
    if not isinstance(raw, dict):
        return None
    history = (features.get("markets") or {}).get(market_name) or {}
    try:
        history_n = int(history.get("history_matches") or 0)
        home_n = int(history.get("home_team_games") or 0)
        away_n = int(history.get("away_team_games") or 0)
    except (TypeError, ValueError):
        return None
    if history_n < 30 or home_n < 4 or away_n < 4:
        return None
    values = {}
    for key in expected_keys:
        try:
            number = float(raw.get(key))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not 0.0 <= number <= expected_max:
            return None
        values[key] = number
    probabilities = {}
    for key in probability_keys:
        try:
            number = float(raw.get(key))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not 0.0 <= number <= 100.0:
            return None
        probabilities[key] = number
    ordered = [probabilities[key] for key in probability_keys]
    if any(ordered[i] + 0.5 < ordered[i + 1] for i in range(len(ordered) - 1)):
        return None
    values.update(probabilities)
    values.update({
        "history_matches": history_n,
        "home_team_games": home_n,
        "away_team_games": away_n,
    })
    return values


def _format_multi_market_sections(parsed: dict, features: dict) -> str:
    """Format only independent probabilities backed by sufficient market history."""
    sections = []
    goal_raw = parsed.get("goal_markets")
    if isinstance(goal_raw, dict) and has_sufficient_history(features):
        try:
            home = float(goal_raw["expected_home_goals"])
            away = float(goal_raw["expected_away_goals"])
            over = [float(goal_raw[k]) for k in ("over_1_5", "over_2_5", "over_3_5", "over_4_5")]
            btts = float(goal_raw["btts_yes"])
            all_values = [home, away, *over, btts]
            if (
                all(math.isfinite(x) for x in all_values)
                and 0 <= home <= 6 and 0 <= away <= 6 and home + away <= 10
                and all(0 <= x <= 100 for x in (*over, btts))
                and all(over[i] + 0.5 >= over[i + 1] for i in range(len(over) - 1))
            ):
                sections.append(
                    "ГОЛЫ (оценка Groq): "
                    f"ожидаемо {home:.2f}+{away:.2f}={home+away:.2f}; "
                    + ", ".join(
                        f"ТБ {line} {prob:.0f}%"
                        for line, prob in zip((1.5, 2.5, 3.5, 4.5), over)
                    )
                    + f"; Обе забьют — Да {btts:.0f}%."
                )
        except (TypeError, ValueError, KeyError):
            pass

    specs = (
        (
            "corner_markets", "corners", "УГЛОВЫЕ",
            ("expected_home_corners", "expected_away_corners"),
            ("over_8_5", "over_9_5", "over_10_5", "over_11_5"),
            (8.5, 9.5, 10.5, 11.5), 20.0,
        ),
        (
            "card_markets", "cards", "КАРТОЧКИ",
            ("expected_home_cards", "expected_away_cards"),
            ("over_2_5", "over_3_5", "over_4_5", "over_5_5"),
            (2.5, 3.5, 4.5, 5.5), 8.0,
        ),
    )
    for raw_key, market_key, label, expected_keys, probability_keys, lines, maximum in specs:
        market_history = (features.get("markets") or {}).get(market_key) or {}
        try:
            enough = (
                int(market_history.get("history_matches") or 0) >= 30
                and int(market_history.get("home_team_games") or 0) >= 4
                and int(market_history.get("away_team_games") or 0) >= 4
            )
        except (TypeError, ValueError):
            enough = False
        if not enough:
            sections.append(f"{label}: недостаточно подтверждённой истории — вероятность не публикуется.")
            continue
        clean = _valid_market_section(
            parsed.get(raw_key),
            expected_keys=expected_keys,
            probability_keys=probability_keys,
            expected_max=maximum,
            features=features,
            market_name=market_key,
        )
        if clean is None:
            sections.append(f"{label}: Groq не вернул корректную оценку по имеющейся истории.")
            continue
        sections.append(
            f"{label} (оценка Groq): "
            f"{clean[expected_keys[0]]:.2f}+{clean[expected_keys[1]]:.2f}="
            f"{clean[expected_keys[0]] + clean[expected_keys[1]]:.2f}; "
            + ", ".join(
                f"ТБ {line} {clean[key]:.0f}%"
                for line, key in zip(lines, probability_keys)
            )
            + f"; история {clean['history_matches']} матчей."
        )
    return " ".join(sections)


def analyze_match(ctx: dict) -> Optional[str]:
    api_key = ctx.get("api_key")
    if not api_key:
        _log("ERROR: api_key пустой")
        return None

    features = ctx.get("independent_features")
    if not has_sufficient_history(features):
        _log(
            "SKIP: недостаточно фактической истории для независимого прогноза "
            f"(нужно >= {MIN_TEAM_VENUE_SAMPLE} домашних матчей хозяев и "
            f">= {MIN_TEAM_VENUE_SAMPLE} выездных матчей гостей)"
        )
        return None

    provider = ctx.get("provider") or "Groq (бесплатно, быстро)"
    if provider not in LLM_PROVIDERS:
        provider = "Groq (бесплатно, быстро)"
    cfg = LLM_PROVIDERS[provider]
    model = ctx.get("model") or cfg["model"]
    base = cfg["base"]

    # The primary model's pick is intentionally excluded from the LLM prompt and
    # cache key. Cached independent analysis can be compared to a fresh model pick.
    cache_context = {
        key: ctx.get(key) for key in (
            "home", "away", "league", "date", "fh", "fa",
            "independent_features", "provider", "model",
        )
    }
    cache_context.update({"provider_resolved": provider, "model_resolved": model})
    ctx_hash = hashlib.sha256(
        json.dumps(cache_context, sort_keys=True, default=str).encode()
    ).hexdigest()
    ck = f"llm_v8_multimarket_blind_{ctx_hash}"
    cached = cache_get(ck, CACHE_TTL["llm"])
    if cached is not None:
        _log(f"CACHE HIT: {ctx_hash[:8]}")
        return _append_model_comparison(str(cached), ctx.get("model_pick_code", ""))

    if usage.llm_remaining() <= 0:
        _log("ERROR: лимит LLM исчерпан")
        return None

    features = ctx.get("independent_features")
    features = features if isinstance(features, dict) else {}
    def fmt(name: str) -> str:
        value = features.get(name)
        try:
            number = float(value)
            return f"{number:.2f}" if number == number and abs(number) != float("inf") else "нет данных"
        except (TypeError, ValueError):
            return "нет данных"

    market_history = features.get("markets") or {}

    def market_block(market_key: str, label: str) -> str:
        item = market_history.get(market_key) or {}
        if not item:
            return f"{label}: исторические данные отсутствуют.\\n"
        return (
            f"{label}: среднее хозяев в домашних матчах={item.get('home_for_avg', 'нет данных')}; "
            f"пропускают дома={item.get('home_against_avg', 'нет данных')}; "
            f"среднее гостей на выезде={item.get('away_for_avg', 'нет данных')}; "
            f"пропускают на выезде={item.get('away_against_avg', 'нет данных')}; "
            f"лиговая база={item.get('history_matches', 0)} матчей; "
            f"выборки хозяев дома={item.get('home_team_games', 0)}, "
            f"гостей на выезде={item.get('away_team_games', 0)}.\\n"
        )

    user_msg = (
        f"Матч: {ctx.get('home', '—')} — {ctx.get('away', '—')}\\n"
        f"Лига: {ctx.get('league', '—')}\\n"
        f"Дата матча: {ctx.get('date', '—')}\\n"
        "ИСТОРИЯ ГОЛОВ (фактические завершённые матчи, не прогноз основной модели):\\n"
        f"Голы хозяев дома за матч: {fmt('home_scored_at_home_avg')} "
        f"(выборка {features.get('home_home_sample', 0)}); "
        f"пропущенные хозяевами дома: {fmt('home_conceded_at_home_avg')}.\\n"
        f"Голы гостей на выезде за матч: {fmt('away_scored_away_avg')} "
        f"(выборка {features.get('away_away_sample', 0)}); "
        f"пропущенные гостями на выезде: {fmt('away_conceded_away_avg')}.\\n"
        f"Последняя форма: хозяева {ctx.get('fh') or 'нет данных'}, "
        f"гости {ctx.get('fa') or 'нет данных'}.\\n"
        "ИСТОРИЯ СЧЁТНЫХ РЫНКОВ (фактические данные, не коэффициенты):\\n"
        + market_block("corners", "Угловые")
        + market_block("cards", "Карточки")
        + "Оцени 1X2 обязательно. Дополнительно оцени тоталы голов/ОЗ, угловые и карточки "
        "только по соответствующей переданной истории. Если данных мало или они отсутствуют, "
        "верни null для соответствующего раздела и перечисли ограничение в risks. "
        "Не используй букмекерские коэффициенты и не подстраивай вероятности под рынок."
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
        "max_tokens": 1600,
    }

    _log(f"REQUEST: provider={provider}, model={model}, mode=blind-independent-multimarket")
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
        # Cache the blind opinion without any comparison text; the comparison is local.
        home_sample = int(features.get("home_home_sample") or 0)
        away_sample = int(features.get("away_away_sample") or 0)
        # No lineup/news data is available, so the source-data label is capped at medium.
        quality_cap = "medium" if min(home_sample, away_sample) >= 5 else "low"
        base_result = _parse_llm_response(content, "", quality_cap=quality_cap)
        if base_result:
            parsed_payload = _response_json(content) or {}
            market_summary = _format_multi_market_sections(parsed_payload, features)
            if market_summary:
                base_result = base_result + " " + market_summary
            cache_put(ck, base_result)
            result = _append_model_comparison(base_result, ctx.get("model_pick_code", ""))
            _log(f"PARSED OK: {result[:120]}")
            return result

        _log("ERROR: ответ не содержит валидного независимого прогноза 1X2")
        cache_put(ck, None)
        return None

    except requests.exceptions.Timeout:
        _log("ERROR: timeout 30s")
        return None
    except requests.exceptions.ConnectionError as e:
        _log("ERROR: connection failed: " + str(e)[:200])
        return None
    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {str(e)[:200]}")
        return None
