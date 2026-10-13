"""Pure settled-bet performance aggregation, independent of the UI."""
def _safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else default
    except (TypeError, ValueError):
        return default


def performance_rows(bets, group_key):
    groups = {}
    for bet in bets:
        if not isinstance(bet, dict):
            continue
        status = str(bet.get("status") or "pending").lower()
        if status not in ("won", "lost"):
            continue
        stake = _safe_float(bet.get("stake"), 0.0)
        odds = _safe_float(bet.get("odds"), 0.0)
        if stake <= 0 or odds <= 1:
            continue
        if group_key == "odds":
            if odds < 1.5:
                label = "< 1.50"
            elif odds < 2.0:
                label = "1.50–1.99"
            elif odds < 3.0:
                label = "2.00–2.99"
            elif odds < 5.0:
                label = "3.00–4.99"
            else:
                label = "5.00+"
        elif group_key == "market":
            label = str(bet.get("market") or "—")
        else:
            label = str(bet.get("league") or bet.get("div") or "—")
        group = groups.setdefault(label, {"Группа": label, "N": 0, "Won": 0,
                                         "Lost": 0, "Stake": 0.0, "PnL": 0.0})
        pnl = stake * (odds - 1.0) if status == "won" else -stake
        group["N"] += 1
        group["Won"] += int(status == "won")
        group["Lost"] += int(status == "lost")
        group["Stake"] += stake
        group["PnL"] += pnl
    rows = []
    for group in groups.values():
        group["Win rate"] = group["Won"] / group["N"] if group["N"] else 0.0
        group["ROI"] = group["PnL"] / group["Stake"] if group["Stake"] else 0.0
        rows.append(group)
    return sorted(rows, key=lambda item: item["N"], reverse=True)
