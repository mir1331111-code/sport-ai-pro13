"""Competition classification helpers shared by fixture feeds and the model."""
from __future__ import annotations

import re
import unicodedata


_WOMEN_MARKERS = (
    "women", "woman", "womens", "ladies", "female", "feminine",
    "feminin", "femmes", "femenina", "femenino", "frauen", "damen",
    "kvinner", "женск", "девуш", "女子",
)


def is_women_competition(*values: object) -> bool:
    """Return True only when explicit competition/team text identifies women.

    Generic labels such as "Soccer" intentionally remain unknown rather than
    being guessed as men's football.
    """
    for value in values:
        text = unicodedata.normalize("NFKD", str(value or "").casefold())
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        if any(marker in text for marker in _WOMEN_MARKERS):
            return True
        # Common compact suffixes used by feeds: "(W)", "- W", "WFC".
        if re.search(r"(?:^|[\s(/_-])w(?:$|[\s)/_-])", text):
            return True
    return False


def competition_gender(
    *, division: object = "", league: object = "", women_flag: object = None,
    home: object = "", away: object = "",
) -> str:
    """Classify as women or unknown; never infer men's football from missing labels."""
    div = str(division or "").strip().upper()
    league_text = str(league or "").strip().casefold()
    if women_flag is True or div.startswith("W_") or is_women_competition(
        league, home, away
    ):
        return "women"
    known_division = div not in {"", "G", "GENERAL", "UNKNOWN", "OTHER"}
    known_league = league_text not in {
        "", "soccer", "football", "матч", "лига", "unknown", "other",
        "неизвестно", "—", "-",
    }
    if known_division or known_league:
        return "men"
    return "unknown"


def model_team_identity(team: object, *, gender: str = "unknown") -> str:
    """Namespace women's team identities so men's club history cannot leak into them."""
    name = str(team or "").strip()
    group = str(gender or "unknown").casefold()
    return f"{group}::{name}" if group in {"women", "unknown"} else name
