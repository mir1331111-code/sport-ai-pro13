"""model/engine.py — Poisson + Dixon-Coles + Elo + H2H + calibration."""
from __future__ import annotations

import math
import re
from collections import defaultdict

from model.calibration import PlattCalibrator


def _new_team():
    return {
        "hs": [],
        "hc": [],
        "as": [],
        "ac": [],
        "form": [],
    }


def _new_lp():
    return {
        "rho": -0.13,
        "w_dc": 0.72,
    }


def is_cup(div: str) -> bool:
    return div in ("C1", "EL", "EC")


def _norm(name: str) -> str:
    if not name:
        return ""

    # Remove punctuation/spacing only. Club words such as "City" and
    # "United" are identity-bearing: stripping them merges distinct teams.
    s = str(name).lower().strip()
    return re.sub(r"[^a-z0-9]", "", s)


class Engine:
    def __init__(self, matrix_n: int = 12):
        self.matrix_n = int(matrix_n)

        self.elo: dict = {}
        self.st = defaultdict(_new_team)

        self.hg: list = []
        self.ag: list = []

        self.lg_hg = defaultdict(list)
        self.lg_ag = defaultdict(list)

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

        self.calibrator = PlattCalibrator()

        self.lp = defaultdict(_new_lp)

        self.match_count = 0
        self.last_match_date: dict = {}
        self.team_history: dict = defaultdict(list)

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

    def _m(self, values, default=1.0):
        return (
            sum(values) / len(values)
            if values
            else default
        )

    @staticmethod
    def _as_day(value):
        if value is None:
            return None
        if hasattr(value, "date") and not isinstance(value, str):
            try:
                return value.date()
            except (TypeError, ValueError):
                return None
        return value if hasattr(value, "toordinal") else None

    def _time_weighted_mean(self, team, field, as_of, fallback, half_life=55.0):
        """Exponential time-decayed mean; excludes observations after as_of."""
        day = self._as_day(as_of)
        weighted_sum = 0.0
        weight_sum = 0.0
        for observed_day, record in self.team_history.get(team, []):
            if field not in record:
                continue
            if day is not None and observed_day is not None and observed_day > day:
                continue
            age = max(0, (day - observed_day).days) if day is not None and observed_day is not None else 0
            weight = 0.5 ** (age / half_life)
            weighted_sum += float(record[field]) * weight
            weight_sum += weight
        return weighted_sum / weight_sum if weight_sum else fallback

    def _rest_days(self, team, match_date):
        current = self._as_day(match_date)
        previous = self._as_day(self.last_match_date.get(team))
        if current is None or previous is None:
            return None
        return max(0, (current - previous).days)

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

    def calibrate(self, p: float, ci: int = 0) -> float:
        return self.calibrator.calibrate(p, ci)

    def refit_calibrator(self) -> None:
        """
        Переобучаем калибратор только на уже сыгранных матчах.

        В walk-forward текущий матч сначала получает prediction,
        затем его результат добавляется в calib_*.
        Поэтому будущие результаты не используются для текущего
        прогноза.
        """
        if len(self.calib_logits) < 240:
            return

        self.calibrator.fit(
            self.calib_logits[-6000:],
            self.calib_outcomes[-6000:],
        )

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

        tau = {
            (0, 0): 1 + lh * la * rho,
            (1, 0): 1 - la * rho,
            (0, 1): 1 - lh * rho,
            (1, 1): 1 + rho,
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

        teams[h]["form"].append(
            3 if hg > ag
            else (1 if hg == ag else 0)
        )

        teams[a]["form"].append(
            3 if ag > hg
            else (1 if hg == ag else 0)
        )

        observed_day = self._as_day(match_date)
        self.team_history[h].append((observed_day, {
            "home_scored": hg, "home_conceded": ag,
            "form": 3 if hg > ag else (1 if hg == ag else 0),
        }))
        self.team_history[a].append((observed_day, {
            "away_scored": ag, "away_conceded": hg,
            "form": 3 if ag > hg else (1 if ag == hg else 0),
        }))
        self.team_history[h] = self.team_history[h][-500:]
        self.team_history[a] = self.team_history[a][-500:]

        self.hg.append(hg)
        self.ag.append(ag)

        if isinstance(row, dict):
            div = row.get("Div") or "G"
        else:
            div = "G"

        self.lg_hg[div].append(hg)
        self.lg_ag[div].append(ag)

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
        features=None,
    ) -> dict:
        """
        Строит прогноз до добавления результата текущего матча.

        match_date and cup remain backward-compatible. Optional features
        may include timestamp-filtered xG as {"xg": {"home": float, "away": float}}.
        Callers must obtain them via FeatureStore.features_as_of(fixture_id, cutoff)
        so post-cutoff lineup/xG data cannot leak into historical predictions.
        """

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
                ),
            )

            la_g = max(
                0.05,
                self._m(
                    self.lg_ag[lg],
                    1.2,
                ),
            )
        else:
            lh_g = max(
                0.05,
                self._m(
                    self.hg,
                    1.5,
                ),
            )

            la_g = max(
                0.05,
                self._m(
                    self.ag,
                    1.2,
                ),
            )

        sh = self.st[h]
        sa = self.st[a]
        rest_days_home = self._rest_days(h, match_date)
        rest_days_away = self._rest_days(a, match_date)

        # ---------------------------------------------------------
        # Силы атаки / защиты
        # ---------------------------------------------------------

        home_scored = self._time_weighted_mean(h, "home_scored", match_date, self._m(sh["hs"], lh_g))
        home_conceded = self._time_weighted_mean(h, "home_conceded", match_date, self._m(sh["hc"], la_g))
        away_scored = self._time_weighted_mean(a, "away_scored", match_date, self._m(sa["as"], la_g))
        away_conceded = self._time_weighted_mean(a, "away_conceded", match_date, self._m(sa["ac"], lh_g))

        attack_home = home_scored / lh_g
        defense_home = home_conceded / la_g
        attack_away = away_scored / la_g
        defense_away = away_conceded / lh_g

        fh = self._time_weighted_mean(h, "form", match_date, self._form(h) * 3.0) / 3.0
        fa = self._time_weighted_mean(a, "form", match_date, self._form(a) * 3.0) / 3.0

        # ---------------------------------------------------------
        # Ожидаемые голы
        # ---------------------------------------------------------

        lam_g_h = max(
            0.3,
            min(
                5.0,
                lh_g
                * attack_home
                * defense_away
                * 1.10
                * (0.85 + 0.30 * fh),
            ),
        )

        lam_g_a = max(
            0.25,
            min(
                4.5,
                la_g
                * attack_away
                * defense_home
                * 0.95
                * (0.85 + 0.30 * fa),
            ),
        )

        # Optional xG blend. This is deliberately conservative and only
        # activates when both sides have finite, plausible xG values.
        xg_used = False
        xg = features.get("xg") if isinstance(features, dict) else None
        if isinstance(xg, dict):
            try:
                xg_home, xg_away = float(xg["home"]), float(xg["away"])
                if (math.isfinite(xg_home) and math.isfinite(xg_away)
                        and 0.0 <= xg_home <= 6.0 and 0.0 <= xg_away <= 6.0):
                    lam_g_h = max(0.3, min(5.0, 0.70 * lam_g_h + 0.30 * xg_home))
                    lam_g_a = max(0.25, min(4.5, 0.70 * lam_g_a + 0.30 * xg_away))
                    xg_used = True
            except (KeyError, TypeError, ValueError, OverflowError):
                pass

        # ---------------------------------------------------------
        # H2H
        # ---------------------------------------------------------

        lam_h, lam_a, h2h_n = self.h2h_adjust(
            h,
            a,
            lam_g_h,
            lam_g_a,
        )

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

        p1, px, matrix = self._p1px(
            lam_h,
            lam_a,
            P0["rho"],
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

        c1 = self.calibrate(f1, 0)
        cx = self.calibrate(fd, 1)
        c2 = self.calibrate(f2, 2)

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
            "rest_days_home": rest_days_home,
            "rest_days_away": rest_days_away,
            "xg_used": xg_used,
            "h2h_n": h2h_n,

            "e": elo_home,
            "pde": pde,
        }

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
        prediction=None,
    ) -> dict:
        """
        Walk-forward learning step.

        Порядок критически важен:

        1. predict() — результат текущего матча ещё неизвестен.
        2. сохраняем prediction для calibration history.
        3. добавляем фактический результат в модель.

        Поэтому текущий матч не обучает сам себя до prediction.
        """

        P = prediction if prediction is not None else self.predict(
            h,
            a,
            lg,
            match_date=match_date,
            cup=is_cup(lg),
        )

        # ---------------------------------------------------------
        # Calibration target
        # ---------------------------------------------------------

        if hg > ag:
            outcome = 0
        elif hg == ag:
            outcome = 1
        else:
            outcome = 2

        self.calib_logits += [
            self._logit(P["p1_raw"]),
            self._logit(P["x_raw"]),
            self._logit(P["p2_raw"]),
        ]

        self.calib_outcomes += [
            1.0 if outcome == 0 else 0.0,
            1.0 if outcome == 1 else 0.0,
            1.0 if outcome == 2 else 0.0,
        ]

        # Ограничиваем историю calibration.
        if len(self.calib_logits) > 12000:
            del self.calib_logits[:-12000]

        if len(self.calib_outcomes) > 12000:
            del self.calib_outcomes[:-12000]

        self.match_count += 1

        # Переобучаем calibration только после накопления
        # достаточного количества новых результатов.
        if self.match_count % 150 == 0:
            self.refit_calibrator()

        if row is None:
            row = {}

        row = dict(row)
        row["Div"] = lg

        # Фактический результат попадает в Engine только здесь,
        # после формирования prediction.
        self.add(
            h,
            a,
            hg,
            ag,
            row,
            match_num=match_num,
            total=total,
            match_date=match_date,
        )

        return P
