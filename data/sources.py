def _tsdb_league_meta(league_name: str):
    """Определяет (div, kind, women, national) по названию лиги TheSportsDB."""
    if not league_name:
        return ("G", "club", False, False)
    ln = league_name.lower()

    # ВАЖНО: сначала "национальные" признаки — U21/U19/U23 имеют приоритет
    # над "championship", иначе U21-турнир мапится на Английский Чемпионшип.
    national = any(n in ln for n in (
        "u21", "u19", "u23", "u20", "u18", "u17",
        "national", "international", "world cup", "euro",
        "nations league", "friendly", "fifa", "uefa nations",
        "copa america", "africa cup", "afc asian",
    ))

    women = any(w in ln for w in (
        "women", "womens", "ladies", "female", "женск"
    ))

    # Если это сборная/U21 — НЕ мапим на клубную лигу
    if national or women:
        kind = "national" if national else "women"
        div = "NT_U21" if "u21" in ln else (
              "NT_U19" if "u19" in ln else (
              "NT_U23" if "u23" in ln else (
              "NT_WC" if "world cup" in ln else (
              "NT_EURO" if "euro" in ln else "NT"))))
        return (div, kind, women, national)

    # Клубные лиги
    mapping = [
        (("premier league", "epl"), "E0"),
        (("championship",), "E1"),
        (("la liga", "laliga"), "SP1"),
        (("segunda",), "SP2"),
        (("serie a",), "I1"),
        (("serie b",), "I2"),
        (("bundesliga",), "D1"),
        (("2. bundesliga",), "D2"),
        (("3. liga",), "D3"),
        (("ligue 1",), "F1"),
        (("ligue 2",), "F2"),
        (("eredivisie",), "N1"),
        (("pro league", "first division a"), "B1"),
        (("primeira liga",), "P1"),
        (("super lig", "super league"), "T1"),
        (("champions league",), "C1"),
        (("europa league",), "EL"),
        (("conference league",), "EC"),
        (("russian premier",), "R1"),
        (("super league greece",), "G1"),
    ]
    div = "G"
    for keys, code in mapping:
        if any(k in ln for k in keys):
            div = code
            break

    return (div, "club", False, False)
