"""Regression tests for the Prognozist Marathon odds adapter."""
from data.marathon import _find_team_match, _parse_events


def test_parse_marathon_1x2_html():
    html = """
    <table><tbody data-event-treeid="12345">
      <tr><td class="date">18.10.2026 20:00</td>
      <td><span class="member-name">Фулхэм</span></td>
      <td><span class="member-name">Халл Сити</span></td>
      <td class="js-price"><span class="selection-link">1.666</span></td>
      <td class="js-price"><span class="selection-link">4.05</span></td>
      <td class="js-price"><span class="selection-link">5.10</span></td>
      </tr></tbody></table>
    """
    events = _parse_events(html)
    assert len(events) == 1
    assert events[0]["event_id"] == "12345"
    assert events[0]["odds"] == {"П1": 1.666, "X": 4.05, "П2": 5.1}


def test_russian_team_mapping_works_after_normalization():
    assert _find_team_match("Fulham", "Фулхэм")
    assert _find_team_match("Hull", "Халл Сити")
