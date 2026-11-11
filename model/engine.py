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


def _division_key(value) -> str:
    """Normalize a competition code used to isolate learned model state."""
    return str(value or "G").strip().upper() or "G"


def _team_state_key(division, team):
    """Keep team form/Elo/H2H/rest history inside one competition."""
    code = _division_key(division)
    name = str(team or "").strip()
    return name if code == "G" else (code, name)


def _h2h_state_key(division, home, away):
    code = _division_key(division)
    return (home, away) if code == "G" else (code, home, away)


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

        # Calibration observations and transforms are isolated by competition.
        self.calib_logits = defaultdict(list)
        self.calib_outcomes = defaultdict(list)
        self.calibration_counts = defaultdict(int)
        self.calibrators = defaultdict(PlattCalibrator)
        self.calibrator = self.calibrators["G"]  # legacy default-division alias

        self.lp = defaultdict(_new_lp)

        self.match_count = 0
        self.last_match_date: dict = {}

        self.trained_n = 0

    def known_teams(self) -> set:
        out = set()

        for state_key in self.st.keys():
            team = state_key[1] if isinstance(state_key, tuple) and len(state_key) == 2 else state_key
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

    def form_str(self, team: str, lg: str = "G") -> str:
        out = ""
        team_key = _team_state_key(lg, team)

        for value in self.st[team_key]["form"][-5:]:
            out += {
                "3": "В",
                "1": "Н",
                "0": "П",
            }[str(int(value))]

        return out or "—"

    def calibrate(self, p: float, ci: int = 0, lg: str = "G") -> float:
        division = _division_key(lg)
        return self.calibrators[division].calibrate(p, ci)

    def refit_calibrator(self, lg: str = "G") -> None:
        """Fit one competition-specific calibration transform on past fixtures."""
        division = _division_key(lg)
        logits = self.calib_logits[division]
        outcomes = self.calib_outcomes[division]
        if len(logits) < 240 or len(logits) != len(outcomes):
            return
        self.calibrators[division].fit(logits[-6000:], outcomes[-6000:])

    def record_calibration(self, lg: str, prediction: dict, outcome: str | int) -> None:
        """Append one settled training label to its own competition calibrator."""
        division = _division_key(lg)
        outcome_code = (
            {"H": 0, "D": 1, "A": 2}.get(str(outcome).upper())
            if isinstance(outcome, str) else int(outcome)
        )
        if outcome_code not in (0, 1, 2):
            raise ValueError("outcome must be H/D/A or 0/1/2")
        raw = (prediction.get("p1_raw"), prediction.get("x_raw"), prediction.get("p2_raw"))
        if any(value is None or not math.isfinite(float(value)) for value in raw):
            raise ValueError("prediction must contain finite raw 1X2 probabilities")
        self.calib_logits[division].extend(self._logit(float(value)) for value in raw)
        self.calib_outcomes[division].extend(
            1.0 if outcome_code == cls else 0.0 for cls in range(3)
        )
        if len(self.calib_logits[division]) > 12000:
            del self.calib_logits[division][:-12000]
            del self.calib_outcomes[division][:-12000]
        self.calibration_counts[division] += 1
        if self.calibration_counts[division] % 150 == 0:
            self.refit_calibrator(division)

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

        div = _division_key(row.get("Div") if isinstance(row, dict) else "G")
        home_key = _team_state_key(div, h)
        away_key = _team_state_key(div, a)

        k = (
            48
            - 32
            * min(
                1.0,
                (match_num or 0) / max(1, total or 1),
            )
        )

        rh = self.elo.get(home_key, 1500)
        ra = self.elo.get(away_key, 1500)

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

        self.elo[home_key] = rh + k * (
            score_home - expected_home
        )

        self.elo[away_key] = ra + k * (
            (1 - score_home)
            - (1 - expected_home)
        )

        teams = self.st

        teams[home_key]["hs"].append(hg)
        teams[home_key]["hc"].append(ag)

        teams[away_key]["as"].append(ag)
        teams[away_key]["ac"].append(hg)

        stamp = match_date.isoformat() if hasattr(match_date, "isoformat") else match_date
        teams[home_key]["hs_dates"].append(stamp)
        teams[home_key]["hc_dates"].append(stamp)
        teams[away_key]["as_dates"].append(stamp)
        teams[away_key]["ac_dates"].append(stamp)

        home_xg, away_xg = _row_xg_pair(row)
        if home_xg is not None and away_xg is not None:
            teams[home_key]["hs_xg"].append(home_xg)
            teams[home_key]["hs_xga"].append(away_xg)
            teams[home_key]["hs_xg_dates"].append(stamp)
            teams[home_key]["hs_xga_dates"].append(stamp)
            teams[away_key]["as_xg"].append(away_xg)
            teams[away_key]["as_xga"].append(home_xg)
            teams[away_key]["as_xg_dates"].append(stamp)
            teams[away_key]["as_xga_dates"].append(stamp)

        teams[home_key]["form"].append(
            3 if hg > ag
            else (1 if hg == ag else 0)
        )

        teams[away_key]["form"].append(
            3 if ag > hg
            else (1 if hg == ag else 0)
        )

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

        home_h2h_key = _h2h_state_key(div, h, a)
        away_h2h_key = _h2h_state_key(div, a, h)
        self.h2h[home_h2h_key].append(diff)
        self.h2h[home_h2h_key] = self.h2h[home_h2h_key][-8:]

        self.h2h[away_h2h_key].append(-diff)
        self.h2h[away_h2h_key] = self.h2h[away_h2h_key][-8:]

        # Ограничиваем историю команды.
        for team in (home_key, away_key):
            for key in teams[team]:
                teams[team][key] = teams[team][key][-12:]

        if match_date:
            self.last_match_date[home_key] = match_date
            self.last_match_date[away_key] = match_date

        self.trained_n += 1

    def h2h_adjust(
        self,
        h,
        a,
        lh,
        la,
        lg="G",
    ):
        """
        Осторожная поправка по личным встречам.

        Используем только при наличии минимум 6 матчей.
        Эффект дополнительно shrink-ится, поэтому H2H не может
        полностью переопределить основную Poisson-модель.
        """

        hist = self.h2h.get(_h2h_state_key(lg, h, a), [])
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

        P0 = self.lp[lg]
        home_key = _team_state_key(lg, h)
        away_key = _team_state_key(lg, a)

        # ---------------------------------------------------------
        # Лиговые средние
        # ---------------------------------------------------------

        # Never borrow fallback averages from other competitions. Sparse
        # competitions shrink only toward neutral football priors, while their
        # own historical scoring rates gain weight as that competition grows.
        home_samples = self.lg_hg.get(lg, [])
        away_samples = self.lg_ag.get(lg, [])
        home_weight = min(1.0, len(home_samples) / 20.0)
        away_weight = min(1.0, len(away_samples) / 20.0)
        home_rate = self._m(home_samples, 1.5, self.lg_hg_dates.get(lg, []), match_date)
        away_rate = self._m(away_samples, 1.2, self.lg_ag_dates.get(lg, []), match_date)
        lh_g = max(0.05, 1.5 * (1.0 - home_weight) + home_rate * home_weight)
        la_g = max(0.05, 1.2 * (1.0 - away_weight) + away_rate * away_weight)

        sh = self.st[home_key]
        sa = self.st[away_key]

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

        fh = self._form(home_key)
        fa = self._form(away_key)

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
            self.last_match_date.get(home_key), match_date)
        away_rest = days_since_last_match(
            self.last_match_date.get(away_key), match_date)

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
                h, a, lam_g_h, lam_g_a, lg=lg,
            )
        else:
            # Counterfactual path for live A/B auditing; do not mutate state.
            lam_h, lam_a = lam_g_h, lam_g_a
            h2h_n = len(self.h2h.get(_h2h_state_key(lg, h, a), []))

        # ---------------------------------------------------------
        # Elo
        # ---------------------------------------------------------

        gh = len(sh["hs"]) + len(sh["as"])
        ga = len(sa["hs"]) + len(sa["as"])

        eh_eff = (
            1500
            + (self.elo.get(home_key, 1500) - 1500)
            * min(1.0, gh / 10.0)
        )

        ea_eff = (
            1500
            + (self.elo.get(away_key, 1500) - 1500)
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

        c1 = self.calibrate(f1, 0, lg)
        cx = self.calibrate(fd, 1, lg)
        c2 = self.calibrate(f2, 2, lg)

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

        row = dict(item.get("row") or {})
        division = _division_key(item.get("league") or row.get("Div"))
        self.record_calibration(division, prediction, outcome)

        self.match_count += 1
        row["Div"] = division
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
