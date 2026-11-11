"""model/engine.py — Poisson + Dixon-Coles + Elo + H2H + calibration."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict

from model.calibration import PlattCalibrator
from model.temporal import days_since_last_match, estimate_dixon_coles_rho, weighted_mean


def _new_team():
    return {
        "hs": [],
        "hc": [],
        "as": [],
        "ac": [],
        "form": [],
        "hs_dates": [],
        "hc_dates": [],
        "as_dates": [],
        "ac_dates": [],
        "hs_xg": [],
        "hs_xga": [],
        "as_xg": [],
        "as_xga": [],
        "hs_xg_dates": [],
        "hs_xga_dates": [],
        "as_xg_dates": [],
        "as_xga_dates": [],
        "home_matches": [],
        "away_matches": [],
    }


def _number(value):
    try:
        result = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0.0 <= result <= 8.0 else None


def _row_xg_pair(row):
    """Extract home/away expected goals from common historical-feed schemas."""
    if not isinstance(row, dict):
        return None, None

    def first(*keys):
        for key in keys:
            value = _number(row.get(key))
            if value is not None:
                return value
        return None

    home_xg = first("HxG", "home_xg", "homeXG", "xG_home", "xg_home",
                    "HomeXG", "home_expected_goals")
    away_xg = first("AxG", "away_xg", "awayXG", "xG_away", "xg_away",
                    "AwayXG", "away_expected_goals")
    # Some feeds expose the same match pair as xG (home) and xGA (away).
    if home_xg is None or away_xg is None:
        fallback_home = first("xG", "xg")
        fallback_away = first("xGA", "xga")
        if home_xg is None:
            home_xg = fallback_home
        if away_xg is None:
            away_xg = fallback_away
    if home_xg is None or away_xg is None:
        return None, None
    return home_xg, away_xg


def _new_lp():
    return {
        "rho": -0.13,
        "w_dc": 0.72,
    }

def _team_namespace(division: str) -> str:
    """Keep men's, women's and national-team records in separate identities."""
    value = str(division or "").upper()
    if value.startswith("W_"):
        return "women"
    if value.startswith("NT"):
        return "national"
    return ""


def _team_identity(name: str, division: str) -> str:
    namespace = _team_namespace(division)
    return f"{namespace}::{name}" if namespace else str(name)


def is_cup(div: str) -> bool:
    return div in ("C1", "EL", "EC")


def _norm(name: str) -> str:
    """Normalize club suffixes without deleting meaningful name words."""
    if not name:
        return ""

    s = unicodedata.normalize("NFKD", str(name).casefold())
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).strip()
    # Club designators are only removable at the end. Words such as City,
    # United and County are part of many real club names and must be retained.
    s = re.sub(r"(?:\s+(?:football club|fc|afc|cf|sc|ac))+$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


class Engine:
    def __init__(self, matrix_n: int = 12, *, temporal_weighting: bool = True, league_rho_fitting: bool = True):
        self.matrix_n = int(matrix_n)
        self.temporal_weighting = bool(temporal_weighting)
        self.league_rho_fitting = bool(league_rho_fitting)

        self.elo: dict = {}
        self.st = defaultdict(_new_team)

        self.hg: list = []
        self.ag: list = []
        self.hg_dates: list = []
        self.ag_dates: list = []
        # Global fallback averages are split by gender/competition namespace.
        self.namespace_hg = defaultdict(list)
        self.namespace_ag = defaultdict(list)
        self.namespace_hg_dates = defaultdict(list)
        self.namespace_ag_dates = defaultdict(list)

        self.lg_hg = defaultdict(list)
        self.lg_ag = defaultdict(list)
        self.lg_hg_dates = defaultdict(list)
        self.lg_ag_dates = defaultdict(list)
        self.league_scores = defaultdict(list)

        # xG-хранилища лиг (инициализация перенесена из ленивого hasattr в add())
        self.lg_hxg = defaultdict(list)
        self.lg_axg = defaultdict(list)
        self.lg_hxg_dates = defaultdict(list)
        self.lg_axg_dates = defaultdict(list)

        # Параметры, доступные для A/B через forecast ledger
        self.half_life_days = 180.0
        self.rest_days_adjustment = True

        # Кэш оценки rho: пересчёт только когда история реально изменилась
        self._rho_cache = {}

        # H2H хранится в перспективе команды, указанной первой.
        #
        # Например:
        # A 2:1 B
        #
        # h2h[(A, B)] -> [+1]
        # h2h[(B, A)] -> [-1]
        #
        # Поэтому матч учитывается независимо от того,
        # кто в будущем будет хозяином.
        self.h2h = defaultdict(list)

        self.calib_logits: list = []
        self.calib_outcomes: list = []
        self.namespace_calibration = defaultdict(lambda: {"logits": [], "outcomes": []})
        self.namespace_calibrators = {}

        self.calibrator = PlattCalibrator()

        self.lp = defaultdict(_new_lp)

        self.match_count = 0
        self.last_match_date: dict = {}

        self.trained_n = 0

    def known_teams(self) -> set:
        out = set()

        for team in self.st.keys():
            n = _norm(team)

            if n:
                out.add(n)

        return out

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def _m(self, values, default=1.0, dates=None, as_of=None):
        if not values:
            return float(default)
        if not self.temporal_weighting:
            return sum(values) / len(values)
        return weighted_mean(
            values, dates, as_of,
            default=default,
            half_life_days=self.half_life_days,
        )

    def _p(self, lam, k):
        try:
            return (
                math.exp(-lam)
                * lam ** k
                / math.factorial(k)
            )
        except Exception:
            return 0.0

    def _form(self, team: str) -> float:
        form = self.st[team]["form"][-5:]

        if not form:
            return 0.5

        return sum(form) / (len(form) * 3)

    def form_str(self, team: str) -> str:
        out = ""

        for value in self.st[team]["form"][-5:]:
            out += {
                "3": "В",
                "1": "Н",
                "0": "П",
            }[str(int(value))]

        return out or "—"

    def calibrate(self, p: float, ci: int = 0, namespace: str = "") -> float:
        if namespace:
            # Never borrow the men's calibrator for a women's or national-team
            # fixture. Until that namespace has enough outcomes, use raw P.
            calibrator = self.namespace_calibrators.get(namespace)
            return calibrator.calibrate(p, ci) if calibrator is not None else float(p)
        return self.calibrator.calibrate(p, ci)

    def refit_calibrator(self) -> None:
        """
        Переобучаем калибратор только на уже сыгранных матчах.

        В walk-forward текущий матч сначала получает prediction,
        затем его результат добавляется в calib_*.
        Поэтому будущие результаты не используются для текущего
        прогноза.
        """
        if len(self.calib_logits) >= 240:
            self.calibrator.fit(
                self.calib_logits[-6000:],
                self.calib_outcomes[-6000:],
            )
        for namespace, history in self.namespace_calibration.items():
            logits = history["logits"]
            outcomes = history["outcomes"]
            if len(logits) < 240:
                continue
            calibrator = self.namespace_calibrators.setdefault(
                namespace, PlattCalibrator()
            )
            calibrator.fit(logits[-6000:], outcomes[-6000:])

    def _p1px(
        self,
        lh: float,
        la: float,
        rho: float,
    ):
        N = self.matrix_n

        matrix = [
            [
                self._p(lh, i) * self._p(la, j)
                for j in range(N)
            ]
            for i in range(N)
        ]

        # Dixon-Coles low-score correction (standard rho convention).
        # Reference: tau(0,0)=1-lh*la*rho; tau(0,1)=1+lh*rho;
        # tau(1,0)=1+la*rho; tau(1,1)=1-rho.
        tau = {
            (0, 0): 1 - lh * la * rho,
            (1, 0): 1 + la * rho,
            (0, 1): 1 + lh * rho,
            (1, 1): 1 - rho,
        }

        for i in range(N):
            for j in range(N):
                if (i, j) in tau:
                    matrix[i][j] *= tau[(i, j)]

        total = sum(map(sum, matrix)) or 1.0

        matrix = [
            [value / total for value in row]
            for row in matrix
        ]

        p1 = sum(
            matrix[i][j]
            for i in range(N)
            for j in range(N)
            if i > j
        )

        px = sum(
            matrix[i][i]
            for i in range(N)
        )

        return p1, px, matrix

    def add(
        self,
        h,
        a,
        hg,
        ag,
        row=None,
        match_num=None,
        total=None,
        match_date=None,
    ):
        """
        Добавляет уже сыгранный матч в состояние модели.

        Важно:
        этот метод вызывается ПОСЛЕ prediction в walk-forward.
        """

        raw_home, raw_away = str(h), str(a)
        div = str((row or {}).get("Div") or "G") if isinstance(row, dict) else "G"
        h = _team_identity(raw_home, div)
        a = _team_identity(raw_away, div)

        k = (
            48
            - 32
            * min(
                1.0,
                (match_num or 0) / max(1, total or 1),
            )
        )

        rh = self.elo.get(h, 1500)
        ra = self.elo.get(a, 1500)

        expected_home = 1 / (
            1
            + 10 ** (
                (ra - (rh + 60)) / 400
            )
        )

        score_home = (
            1.0
            if hg > ag
            else (0.5 if hg == ag else 0.0)
        )

        self.elo[h] = rh + k * (
            score_home - expected_home
        )

        self.elo[a] = ra + k * (
            (1 - score_home)
            - (1 - expected_home)
        )

        teams = self.st

        teams[h]["hs"].append(hg)
        teams[h]["hc"].append(ag)

        teams[a]["as"].append(ag)
        teams[a]["ac"].append(hg)

        stamp = match_date.isoformat() if hasattr(match_date, "isoformat") else match_date
        teams[h]["hs_dates"].append(stamp)
        teams[h]["hc_dates"].append(stamp)
        teams[a]["as_dates"].append(stamp)
        teams[a]["ac_dates"].append(stamp)

        home_xg, away_xg = _row_xg_pair(row)
        # Store dated, pre-match opponent-strength snapshots for evidence
        # reporting. These records are diagnostic; they do not alter lambdas.
        home_record = {
            "date": str(stamp or ""),
            "opponent": raw_away,
            "goals_for": float(hg),
            "goals_against": float(ag),
            "opponent_elo_before": float(ra),
            "xg_for": home_xg,
            "xg_against": away_xg,
        }
        away_record = {
            "date": str(stamp or ""),
            "opponent": raw_home,
            "goals_for": float(ag),
            "goals_against": float(hg),
            "opponent_elo_before": float(rh),
            "xg_for": away_xg,
            "xg_against": home_xg,
        }
        teams[h]["home_matches"].append(home_record)
        teams[a]["away_matches"].append(away_record)
        teams[h]["home_matches"] = teams[h]["home_matches"][-20:]
        teams[a]["away_matches"] = teams[a]["away_matches"][-20:]
        if home_xg is not None and away_xg is not None:
            teams[h]["hs_xg"].append(home_xg)
            teams[h]["hs_xga"].append(away_xg)
            teams[h]["hs_xg_dates"].append(stamp)
            teams[h]["hs_xga_dates"].append(stamp)
            teams[a]["as_xg"].append(away_xg)
            teams[a]["as_xga"].append(home_xg)
            teams[a]["as_xg_dates"].append(stamp)
            teams[a]["as_xga_dates"].append(stamp)

        teams[h]["form"].append(
            3 if hg > ag
            else (1 if hg == ag else 0)
        )

        teams[a]["form"].append(
            3 if ag > hg
            else (1 if hg == ag else 0)
        )

        namespace = _team_namespace(div)
        self.namespace_hg[namespace].append(hg)
        self.namespace_ag[namespace].append(ag)
        self.namespace_hg_dates[namespace].append(stamp)
        self.namespace_ag_dates[namespace].append(stamp)
        # Preserve legacy global arrays as men's-only averages.
        if not namespace:
            self.hg.append(hg)
            self.ag.append(ag)
            self.hg_dates.append(stamp)
            self.ag_dates.append(stamp)

        self.lg_hg[div].append(hg)
        self.lg_ag[div].append(ag)
        self.lg_hg_dates[div].append(stamp)
        self.lg_ag_dates[div].append(stamp)
        if home_xg is not None and away_xg is not None:
            self.lg_hxg[div].append(home_xg)
            self.lg_axg[div].append(away_xg)
            self.lg_hxg_dates[div].append(stamp)
            self.lg_axg_dates[div].append(stamp)
            self.lg_hxg[div] = self.lg_hxg[div][-500:]
            self.lg_axg[div] = self.lg_axg[div][-500:]
            self.lg_hxg_dates[div] = self.lg_hxg_dates[div][-500:]
            self.lg_axg_dates[div] = self.lg_axg_dates[div][-500:]
        self.league_scores[div].append((stamp, hg, ag))
        self.league_scores[div] = self.league_scores[div][-500:]

        # ---------------------------------------------------------
        # H2H
        # ---------------------------------------------------------
        #
        # Старая реализация записывала только:
        #
        #   h2h[(home, away)]
        #
        # Поэтому при следующей встрече тех же команд с поменявшимися
        # местами хозяевами история фактически терялась.
        #
        # Теперь сохраняем обе перспективы:
        #
        # A 2:1 B
        #
        # A -> B = +1
        # B -> A = -1
        #
        # Это позволяет h2h_adjust() корректно работать независимо
        # от того, кто играет дома.

        diff = hg - ag

        self.h2h[(h, a)].append(diff)
        self.h2h[(h, a)] = self.h2h[(h, a)][-8:]

        self.h2h[(a, h)].append(-diff)
        self.h2h[(a, h)] = self.h2h[(a, h)][-8:]

        # Ограничиваем историю команды.
        for team in (h, a):
            for key in teams[team]:
                teams[team][key] = teams[team][key][-12:]

        if match_date:
            self.last_match_date[h] = match_date
            self.last_match_date[a] = match_date

        self.trained_n += 1

    def h2h_adjust(
        self,
        h,
        a,
        lh,
        la,
    ):
        """
        Осторожная поправка по личным встречам.

        Используем только при наличии минимум 6 матчей.
        Эффект дополнительно shrink-ится, поэтому H2H не может
        полностью переопределить основную Poisson-модель.
        """

        hist = self.h2h.get((h, a), [])
        n = len(hist)

        if n < 6:
            return lh, la, n

        shrink = min(
            1.0,
            (n - 5) / 8.0,
        )

        avg_diff = sum(hist) / n

        shift = (
            avg_diff
            * 0.04
            * shrink
        )

        new_lh = max(
            0.3,
            lh + shift / 2,
        )

        new_la = max(
            0.25,
            la - shift / 2,
        )

        return new_lh, new_la, n

    def predict(
        self,
        h,
        a,
        lg="G",
        match_date=None,
        cup=False,
        *,
        apply_h2h=True,
    ) -> dict:
        """
        Строит прогноз до добавления результата текущего матча.

        match_date и cup оставлены в интерфейсе для совместимости
        с остальной системой.
        """

        namespace = _team_namespace(lg)
        h = _team_identity(str(h), lg)
        a = _team_identity(str(a), lg)
        P0 = self.lp[lg]

        # ---------------------------------------------------------
        # Лиговые средние
        # ---------------------------------------------------------

        if len(self.lg_hg.get(lg, [])) >= 20:
            lh_g = max(
                0.05,
                self._m(
                    self.lg_hg[lg],
                    1.5,
                    self.lg_hg_dates[lg],
                    match_date,
                ),
            )

            la_g = max(
                0.05,
                self._m(
                    self.lg_ag[lg],
                    1.2,
                    self.lg_ag_dates[lg],
                    match_date,
                ),
            )
        else:
            global_home = self.namespace_hg[namespace]
            global_away = self.namespace_ag[namespace]
            global_home_dates = self.namespace_hg_dates[namespace]
            global_away_dates = self.namespace_ag_dates[namespace]
            lh_g = max(
                0.05,
                self._m(
                    global_home,
                    1.5,
                    global_home_dates,
                    match_date,
                ),
            )

            la_g = max(
                0.05,
                self._m(
                    global_away,
                    1.2,
                    global_away_dates,
                    match_date,
                ),
            )

        sh = self.st[h]
        sa = self.st[a]

        # ---------------------------------------------------------
        # Силы атаки / защиты
        # ---------------------------------------------------------

        # Empirical-Bayes shrinkage toward league scoring rates. Sparse team
        # samples should not create extreme attack/defence multipliers.
        shrink_k = 6.0

        def shrunk_rate(values, dates, league_rate):
            if not values:
                return league_rate
            team_rate = self._m(values, league_rate, dates, match_date)
            return (team_rate * len(values) + shrink_k * league_rate) / (len(values) + shrink_k)

        attack_home = shrunk_rate(sh["hs"], sh["hs_dates"], lh_g) / lh_g
        defense_home = shrunk_rate(sh["hc"], sh["hc_dates"], la_g) / la_g
        attack_away = shrunk_rate(sa["as"], sa["as_dates"], la_g) / la_g
        defense_away = shrunk_rate(sa["ac"], sa["ac_dates"], lh_g) / lh_g

        # xG is blended only when this league has a useful dated sample and
        # the individual team has enough same-venue observations. Missing xG
        # never becomes zero and never disables the goals-based model.
        xg_weights = []
        xg_used = False
        xg_home_samples = len(sh["hs_xg"])
        xg_away_samples = len(sa["as_xg"])
        league_home_xg = None
        league_away_xg = None
        league_xg_n = len(self.lg_hxg.get(lg, []))
        if league_xg_n >= 30:
            league_home_xg = self._m(
                self.lg_hxg[lg], lh_g, self.lg_hxg_dates[lg], match_date)
            league_away_xg = self._m(
                self.lg_axg[lg], la_g, self.lg_axg_dates[lg], match_date)

        def blend_strength(goal_strength, values, dates, league_xg):
            nonlocal xg_used
            if league_xg is None or len(values) < 4 or league_xg <= 0:
                return goal_strength
            xg_strength = self._m(values, league_xg, dates, match_date) / league_xg
            weight = 0.35 * min(1.0, len(values) / 8.0)
            xg_used = True
            xg_weights.append(weight)
            return (1.0 - weight) * goal_strength + weight * xg_strength

        if league_home_xg is not None and league_away_xg is not None:
            attack_home = blend_strength(
                attack_home, sh["hs_xg"], sh["hs_xg_dates"], league_home_xg)
            defense_away = blend_strength(
                defense_away, sa["as_xga"], sa["as_xga_dates"], league_home_xg)
            attack_away = blend_strength(
                attack_away, sa["as_xg"], sa["as_xg_dates"], league_away_xg)
            defense_home = blend_strength(
                defense_home, sh["hs_xga"], sh["hs_xga_dates"], league_away_xg)

        fh = self._form(h)
        fa = self._form(a)

        # ---------------------------------------------------------
        # Ожидаемые голы
        # ---------------------------------------------------------

        lam_g_h = max(
            0.3,
            min(
                5.0,
                lh_g
                * attack_home
                * defense_away,
            ),
        )

        lam_g_a = max(
            0.25,
            min(
                4.5,
                la_g
                * attack_away
                * defense_home,
            ),
        )

        # ---------------------------------------------------------
        # Rest days
        # ---------------------------------------------------------
        # Коэффициенты — приор, не истина: гонять A/B через ledger
        # (self.rest_days_adjustment выключить/включить).
        home_rest = days_since_last_match(
            self.last_match_date.get(h), match_date)
        away_rest = days_since_last_match(
            self.last_match_date.get(a), match_date)

        def _rest_factor(rest):
            if rest is None:
                return 1.0
            if rest <= 2:
                return 0.94   # <=2 дней отдыха — заметный минус
            if rest == 3:
                return 0.97
            if rest >= 7:
                return 1.02   # полная неделя — лёгкий плюс
            return 1.0

        if self.rest_days_adjustment:
            lam_g_h = max(0.3, lam_g_h * _rest_factor(home_rest))
            lam_g_a = max(0.25, lam_g_a * _rest_factor(away_rest))

        # ---------------------------------------------------------
        # H2H
        # ---------------------------------------------------------

        if apply_h2h:
            lam_h, lam_a, h2h_n = self.h2h_adjust(
                h, a, lam_g_h, lam_g_a,
            )
        else:
            # Counterfactual path for live A/B auditing; do not mutate state.
            lam_h, lam_a = lam_g_h, lam_g_a
            h2h_n = len(self.h2h.get((h, a), []))

        # ---------------------------------------------------------
        # Elo
        # ---------------------------------------------------------

        gh = len(sh["hs"]) + len(sh["as"])
        ga = len(sa["hs"]) + len(sa["as"])

        eh_eff = (
            1500
            + (self.elo.get(h, 1500) - 1500)
            * min(1.0, gh / 10.0)
        )

        ea_eff = (
            1500
            + (self.elo.get(a, 1500) - 1500)
            * min(1.0, ga / 10.0)
        )

        elo_home = 1 / (
            1
            + 10 ** (
                (ea_eff - eh_eff - 60)
                / 400
            )
        )

        # Вероятность ничьей для Elo-компоненты.
        pde = (
            0.20
            + 0.12
            * (
                1
                - abs(elo_home - 0.5) * 2
            )
        )

        # ---------------------------------------------------------
        # Dixon-Coles / Poisson
        # ---------------------------------------------------------

        # Мемоизация: история меняется только в add(), поэтому
        # оценка rho не должна пересчитываться на каждый predict.
        rho_cache_key = (
            lg,
            len(self.league_scores.get(lg, [])),
            str(match_date)[:10] if match_date is not None else "",
        )
        rho = self._rho_cache.get(rho_cache_key)
        if rho is None:
            rho = (
                estimate_dixon_coles_rho(
                    self.league_scores.get(lg, []),
                    lh_g,
                    la_g,
                    prior_rho=P0["rho"],
                    min_matches=80,
                    prior_strength=40.0,
                    as_of=match_date,
                )
                if self.league_rho_fitting else P0["rho"]
            )
            self._rho_cache[rho_cache_key] = rho
            if len(self._rho_cache) > 500:
                self._rho_cache.clear()

        p1, px, matrix = self._p1px(
            lam_h,
            lam_a,
            rho,
        )

        # ---------------------------------------------------------
        # Объединяем Poisson/DC и Elo
        # ---------------------------------------------------------

        f1 = (
            P0["w_dc"] * p1
            + (1 - P0["w_dc"])
            * elo_home
            * (1 - pde)
        )

        fd = (
            P0["w_dc"] * px
            + (1 - P0["w_dc"])
            * pde
        )

        f2 = max(
            0.0,
            1 - f1 - fd,
        )

        # ---------------------------------------------------------
        # Calibration
        # ---------------------------------------------------------

        namespace = _team_namespace(lg)
        c1 = self.calibrate(f1, 0, namespace)
        cx = self.calibrate(fd, 1, namespace)
        c2 = self.calibrate(f2, 2, namespace)

        total_calibrated = c1 + cx + c2 or 1.0

        c1 /= total_calibrated
        cx /= total_calibrated
        c2 /= total_calibrated

        # ---------------------------------------------------------
        # Total / BTTS
        # ---------------------------------------------------------

        over = 1 - sum(
            self._p(
                lam_h + lam_a,
                k,
            )
            for k in range(3)
        )

        N = self.matrix_n

        btts = sum(
            matrix[i][j]
            for i in range(1, N)
            for j in range(1, N)
        )

        games = min(gh, ga)

        return {
            "p1": c1,
            "x": cx,
            "p2": c2,

            "p1_raw": f1,
            "x_raw": fd,
            "p2_raw": f2,

            "over": over,
            "btts": btts,

            "M": matrix,

            "lams": (
                lam_h,
                lam_a,
            ),

            "games": games,
            "h2h_n": h2h_n,
            "namespace": namespace,
            "home_history_n": gh,
            "away_history_n": ga,
            "home_venue_history_n": len(sh["hs"]),
            "away_venue_history_n": len(sa["as"]),
            "home_opponent_elo_avg": (
                sum(float(item["opponent_elo_before"]) for item in sh["home_matches"][-5:])
                / len(sh["home_matches"][-5:]) if sh["home_matches"] else None
            ),
            "away_opponent_elo_avg": (
                sum(float(item["opponent_elo_before"]) for item in sa["away_matches"][-5:])
                / len(sa["away_matches"][-5:]) if sa["away_matches"] else None
            ),

            "e": elo_home,
            "pde": pde,
            "rho": rho,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "xg_used": xg_used,
            "xg_home_samples": xg_home_samples,
            "xg_away_samples": xg_away_samples,
            "xg_league_samples": league_xg_n,
            "xg_weight": (sum(xg_weights) / len(xg_weights)) if xg_weights else 0.0,
        }

    def _learn_observed_match(self, item: dict, prediction: dict) -> None:
        """Attach one observed result after its forecast has already been made."""
        hg = float(item["home_goals"])
        ag = float(item["away_goals"])
        if hg > ag:
            outcome = 0
        elif hg == ag:
            outcome = 1
        else:
            outcome = 2

        namespace = str(prediction.get("namespace") or "")
        logits = [
            self._logit(prediction["p1_raw"]),
            self._logit(prediction["x_raw"]),
            self._logit(prediction["p2_raw"]),
        ]
        outcomes = [
            1.0 if outcome == 0 else 0.0,
            1.0 if outcome == 1 else 0.0,
            1.0 if outcome == 2 else 0.0,
        ]
        if namespace:
            history = self.namespace_calibration[namespace]
            history["logits"].extend(logits)
            history["outcomes"].extend(outcomes)
            if len(history["logits"]) > 12000:
                del history["logits"][:-12000]
                del history["outcomes"][:-12000]
        else:
            self.calib_logits.extend(logits)
            self.calib_outcomes.extend(outcomes)
            if len(self.calib_logits) > 12000:
                del self.calib_logits[:-12000]
                del self.calib_outcomes[:-12000]

        self.match_count += 1
        if self.match_count % 150 == 0:
            self.refit_calibrator()

        row = dict(item.get("row") or {})
        row["Div"] = item.get("league") or "G"
        self.add(
            item["home"], item["away"], hg, ag, row,
            match_num=item.get("match_num"),
            total=item.get("total"),
            match_date=item.get("match_date"),
        )

    def learn_batch(self, matches: list[dict]) -> list[dict]:
        """Forecast a same-date batch before ingesting any result from that batch.

        The production trainer uses this for all leagues on a calendar date.
        It prevents one fixture's same-day final score from entering the
        probabilities or calibration targets of another fixture whose kickoff
        order is unknown in the historical CSV.
        """
        if not isinstance(matches, list):
            raise TypeError("matches must be a list of mappings")
        pending = []
        predictions = []
        for item in matches:
            if not isinstance(item, dict):
                raise TypeError("each match must be a mapping")
            home = str(item.get("home") or "").strip()
            away = str(item.get("away") or "").strip()
            if not home or not away:
                raise ValueError("home and away team names are required")
            try:
                hg = float(item["home_goals"])
                ag = float(item["away_goals"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("completed home/away scores are required") from exc
            if (
                not math.isfinite(hg) or not math.isfinite(ag)
                or hg < 0 or ag < 0 or not hg.is_integer() or not ag.is_integer()
            ):
                raise ValueError("completed scores must be finite non-negative integers")
            league = str(item.get("league") or "G")
            prediction = self.predict(
                home, away, league,
                match_date=item.get("match_date"),
                cup=is_cup(league),
            )
            clean = dict(item)
            clean.update({
                "home": home, "away": away,
                "home_goals": int(hg), "away_goals": int(ag),
                "league": league,
            })
            pending.append((clean, prediction))
            predictions.append(prediction)

        # No model state, Elo, form, team rates or calibrator targets are changed
        # until every prediction in this batch has been produced.
        for item, prediction in pending:
            self._learn_observed_match(item, prediction)
        return predictions

    def learn_step(
        self,
        h,
        a,
        hg,
        ag,
        row=None,
        lg="G",
        match_num=None,
        total=None,
        match_date=None,
    ) -> dict:
        """Single-match convenience wrapper around the leakage-safe batch API."""
        item = {
            "home": h, "away": a, "home_goals": hg, "away_goals": ag,
            "row": row or {}, "league": lg, "match_num": match_num,
            "total": total, "match_date": match_date,
        }
        return self.learn_batch([item])[0]
