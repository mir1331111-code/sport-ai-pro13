"""VALUE ENGINE: exact-market prices and existing validation gates."""
from __future__ import annotations
import pandas as pd
import streamlit as st

def render() -> None:
    data = st.session_state.get("data", {})
    cards = [card for card in (data.get("cards") or []) if isinstance(card, dict)]
    st.header("💎 VALUE ENGINE")
    st.caption("Проверяются только котировки на тот же рынок и линию. Отсутствие цены не заменяется коэффициентом другого рынка; новые рынки не получают Value автоматически.")
    if not cards:
        st.info("После сканирования здесь появится список подтверждённых котировок.")
        return
    rows, blocked = [], 0
    for card in cards:
        match = str(card.get("match_ru") or card.get("match") or "Матч")
        recommendations = card.get("best_by_market") or {}
        for family, recommendation in recommendations.items():
            if not isinstance(recommendation, dict):
                continue
            best = recommendation.get("best")
            verdict = recommendation.get("verdict") or {}
            if not isinstance(best, (tuple, list)) or len(best) < 5:
                continue
            try:
                market, selection = str(best[0]), str(best[1])
                odd, ev, probability = float(best[2]), float(best[3]), float(best[4])
            except (TypeError, ValueError):
                continue
            if odd <= 1.01:
                continue
            context_ok = bool((card.get("context_quality") or {}).get("strong_value_eligible"))
            market_validated = bool(
                card.get("winner_market_validation") if market == "1X2"
                else card.get("total_market_validation") if market == "OU"
                else card.get("btts_market_validation") if market == "BTTS"
                else (card.get("stat_market_validation") or {}).get(market, False)
            )
            passed = bool(context_ok and market_validated and verdict.get("value_passed") and verdict.get("is_bet"))
            if not passed:
                blocked += 1
            rows.append({
                "Матч": match, "Рынок": market, "Выбор": selection,
                "Коэффициент": odd, "P модели": probability, "EV": ev,
                "Статус": "Подтверждённые фильтры" if passed else "Наблюдение / заблокировано",
                "Букмекер": str((card.get("bookmaker_by_pick") or {}).get(selection) or "не указан"),
                "Источник": str((card.get("market_odds_source_by_pick") or {}).get(selection) or card.get("odds_source") or "неизвестен"),
            })
    confirmed = sum(row["Статус"] == "Подтверждённые фильтры" for row in rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Рыночных сигналов с ценой", len(rows))
    c2.metric("Прошли существующие фильтры", confirmed)
    c3.metric("Заблокировано / наблюдение", blocked)
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Пока нет сигналов с реальной котировкой на соответствующий рынок.")
    st.warning("Этот экран не размещает ставки. BTTS, угловые, карточки и Asian Handicap остаются информационными, пока для каждого рынка не доказано преимущество на независимой выборке.")
