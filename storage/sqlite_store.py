@_cache(ttl=30, show_spinner=False)
def clv_summary() -> dict:
    """Возвращает {'n': int, 'avg_clv': float, 'positive_share': float}.
    Безопасен для любой row_factory и пустой таблицы."""
    empty = {"n": 0, "avg_clv": 0.0, "positive_share": 0.0}
    try:
        with _db() as c:
            # 1) Сколько вообще есть CLV-записей
            r1 = c.execute("SELECT COUNT(*) FROM bets WHERE clv IS NOT NULL").fetchone()
            n = int(r1[0]) if r1 and r1[0] is not None else 0
            if n <= 0:
                return empty
            # 2) Средний CLV
            r2 = c.execute("SELECT AVG(clv) FROM bets WHERE clv IS NOT NULL").fetchone()
            avg_val = float(r2[0]) if r2 and r2[0] is not None else 0.0
            # 3) Доля положительных CLV
            r3 = c.execute("SELECT SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) FROM bets WHERE clv IS NOT NULL").fetchone()
            pos_count = int(r3[0]) if r3 and r3[0] is not None else 0
            pos_share = pos_count / n if n else 0.0
            return {"n": n, "avg_clv": avg_val, "positive_share": pos_share}
    except Exception:
        return empty
