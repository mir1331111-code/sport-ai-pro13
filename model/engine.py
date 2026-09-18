"""model/engine.py — Poisson + Dixon-Coles + Elo."""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Optional

from model.calibration import PlattCalibrator


def _new_team():
    return {"hs": [], "hc": [], "as": [], "ac": [], "form": []}


def _new_lp():
    return {"rho": -0.13, "w_dc": 0.72}


def is_cup(div: str) -> bool:
    return div in ("C1", "EL", "EC")


class Engine:
    def __init__(self, matrix_n: int = 12):
        self.matrix_n = int(matrix_n)
        self.elo: dict = {}
        self.st = defaultdict(_new_team)
        self.hg: list = []
        self.ag: list = []
        self.lg_hg = defaultdict(list)
        self.lg_ag = defaultdict(list)
        self.h2h = defaultdict(list)
        self.calib_logits: list = []
        self.calib_outcomes: list = []
        self.calibrator = PlattCalibrator()
        self.lp = defaultdict(_new_lp)
        self.match_count = 0
        self.last_match_date: dict = {}
        self.trained_n = 0

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def _m(self, l, d=1.0):
        return sum(l) / len(l) if l else d

    def _p(self, l, k):
        try:
            return math.exp(-l) * l ** k / math.factorial(k)
        except Exception:
            return 0.0

    def _form(self, t: str) -> float:
        f = self.st[t]["form"][-5:]
        return (sum(f) / (len(f) * 3)) if f else 0.5

    def form_str(self, t: str) -> str:
        out = ""
        for x in self.st[t]["form"][-5:]:
            out += {"3": "В", "1": "Н", "0": "П"}[str(int(x))]
        return out or "—"

    def calibrate(self, p: float, ci: int = 0) -> float:
        return self.calibrator.calibrate(p, ci)

    def refit_calibrator(self) -> None:
        if len(self.calib_logits) < 240:
            return
        self.calibrator.fit(self.calib_logits[-6000:], self.calib_outcomes[-6000:])

    def _p1px(self, lh: float, la: float, rho: float):
        N = self.matrix_n
        M = [[self._p(lh, i) * self._p(la, j) for j in range(N)]
             for i in range(N)]
        tau = {(0, 0): 1 + lh * la * rho, (1, 0): 1 - la * rho,
               (0, 1): 1 - lh * rho, (1, 1): 1 + rho}
        for i in range(N):
            for j in range(N):
                if (i, j) in tau:
                    M[i][j] *= tau[(i, j)]
        tot = sum(map(sum, M)) or 1.0
        M = [[v / tot for v in r] for r in M]
        p1 = sum(M[i][j] for i in range(N) for j in range(N) if i > j)
        px = sum(M[i][i] for i in range(N))
        return p1, px, M

    def add(self, h, a, hg, ag, row=None, match_num=None, total=None,
            match_date=None):
        k = 48 - 32 * min(1.0, (match_num or 0) / max(1, total or 1))
        rh, ra = self.elo.get(h, 1500), self.elo.get(a, 1500)
        eh = 1 / (1 + 10 ** ((ra - (rh + 60)) / 400))
        s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        self.elo[h] = rh + k * (s - eh)
        self.elo[a] = ra + k * ((1 - s) - (1 - eh))
        t = self.st
        t[h]["hs"].append(hg)
        t[h]["hc"].append(ag)
        t[a]["as"].append(ag)
        t[a]["ac"].append(hg)
        t[h]["form"].append(3 if hg > ag else (1 if hg == ag else 0))
        t[a]["form"].append(3 if ag > hg else (1 if hg == ag else 0))
        self.hg.append(hg)
        self.ag.append(ag)
        div = (row or {}).get("Div") or "G" if isinstance(row, dict) else "G"
        self.lg_hg[div].append(hg)
        self.lg_ag[div].append(ag)
        self.h2h[(h, a)].append(hg - ag)
        self.h2h[(h, a)] = self.h2h[(h, a)][-8:]
        for team in (h, a):
            for key in t[team]:
                t[team][key] = t[team][key][-12:]
        if match_date:
            self.last_match_date[h] = match_date
            self.last_match_date[a] = match_date

    def h2h_adjust(self, h, a, lh, la):
        hist = self.h2h.get((h, a), [])
        n = len(hist)
        if n < 6:
            return lh, la, n
        shrink = min(1.0, (n - 5) / 8.0)
        shift = (sum(hist) / n) * 0.04 * shrink
        return max(0.3, lh + shift / 2), max(0.25, la - shift / 2), n

    def predict(self, h, a, lg="G", match_date=None, cup=False) -> dict:
        P0 = self.lp[lg]
        if len(self.lg_hg.get(lg, [])) >= 20:
            lh_g = max(0.05, self._m(self.lg_hg[lg], 1.5))
            la_g = max(0.05, self._m(self.lg_ag[lg], 1.2))
        else:
            lh_g = max(0.05, self._m(self.hg, 1.5))
            la_g = max(0.05, self._m(self.ag, 1.2))
        sh, sa = self.st[h], self.st[a]
        ah_ = self._m(sh["hs"], lh_g) / lh_g
        dh_ = self._m(sh["hc"], la_g) / la_g
        aa_ = self._m(sa["as"], la_g) / la_g
        da_ = self._m(sa["ac"], lh_g) / lh_g
        fh, fa = self._form(h), self._form(a)
        lam_g_h = max(0.3, min(5.0, lh_g * ah_ * da_ * 1.10 * (0.85 + 0.30 * fh)))
        lam_g_a = max(0.25, min(4.5, la_g * aa_ * dh_ * 0.95 * (0.85 + 0.30 * fa)))
        lam_h, lam_a, h2h_n = self.h2h_adjust(h, a, lam_g_h, lam_g_a)
        gh = len(sh["hs"]) + len(sh["as"])
        ga = len(sa["hs"]) + len(sa["as"])
        eh_eff = 1500 + (self.elo.get(h, 1500) - 1500) * min(1.0, gh / 10.0)
        ea_eff = 1500 + (self.elo.get(a, 1500) - 1500) * min(1.0, ga / 10.0)
        e = 1 / (1 + 10 ** ((ea_eff - eh_eff - 60) / 400))
        pde = 0.20 + 0.12 * (1 - abs(e - 0.5) * 2)
        p1, px, M = self._p1px(lam_h, lam_a, P0["rho"])
        f1 = P0["w_dc"] * p1 + (1 - P0["w_dc"]) * e * (1 - pde)
        fd = P0["w_dc"] * px + (1 - P0["w_dc"]) * pde
        f2 = max(0.0, 1 - f1 - fd)
        games = min(gh, ga)
        c1 = self.calibrate(f1, 0)
        cx = self.calibrate(fd, 1)
        c2 = self.calibrate(f2, 2)
        ct = c1 + cx + c2 or 1.0
        c1 /= ct
        cx /= ct
        c2 /= ct
        N = self.matrix_n
        over = 1 - sum(self._p(lam_h + lam_a, k) for k in range(3))
        btts = sum(M[i][j] for i in range(1, N) for j in range(1, N))
        return {
            "p1": c1, "x": cx, "p2": c2,
            "p1_raw": f1, "x_raw": fd, "p2_raw": f2,
            "over": over, "btts": btts, "M": M,
            "lams": (lam_h, lam_a),
            "games": games, "h2h_n": h2h_n, "e": e, "pde": pde,
        }

    def learn_step(self, h, a, hg, ag, row=None, lg="G", match_num=None,
                   total=None, match_date=None) -> dict:
        P = self.predict(h, a, lg, match_date=match_date, cup=is_cup(lg))
        out = 0 if hg > ag else (1 if hg == ag else 2)
        self.calib_logits += [self._logit(P["p1_raw"]),
                              self._logit(P["x_raw"]),
                              self._logit(P["p2_raw"])]
        self.calib_outcomes += [1.0 if out == 0 else 0.0,
                                1.0 if out == 1 else 0.0,
                                1.0 if out == 2 else 0.0]
        if len(self.calib_logits) > 12000:
            del self.calib_logits[:-12000]
            del self.calib_outcomes[:-12000]
        self.match_count += 1
        if self.match_count % 150 == 0:
            self.refit_calibrator()
        if row is None:
            row = {}
        row = dict(row)
        row["Div"] = lg
        self.add(h, a, hg, ag, row, match_num=match_num, total=total,
                 match_date=match_date)
        return P
