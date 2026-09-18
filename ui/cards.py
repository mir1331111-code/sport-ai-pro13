

# ============ ХЕЛПЕРЫ ДЛЯ СОРТИРОВКИ ============
def parse_card_date(c: dict):
    """Возвращает datetime карточки для сортировки."""
    from datetime import datetime
    try:
        iso = c.get("date_iso") or ""
        if iso:
            return datetime.strptime(iso[:10], "%Y-%m-%d")
    except Exception:
        pass
    return datetime(2099, 1, 1)


def day_label(dt) -> str:
    """Сегодня / Завтра / Послезавтра / Пт 20.09 / Прошлое."""
    from datetime import datetime, timedelta
    if not dt:
        return "—"
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    diff = (d - today).days
    if diff == 0:
        return "📅 СЕГОДНЯ"
    if diff == 1:
        return "📅 ЗАВТРА"
    if diff == 2:
        return "📅 ПОСЛЕЗАВТРА"
    if diff < 0:
        return f"📅 ПРОШЛОЕ ({abs(diff)}д назад)"
    weekdays = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    return f"📅 {weekdays[dt.weekday()]} {dt.strftime('%d.%m')}"
