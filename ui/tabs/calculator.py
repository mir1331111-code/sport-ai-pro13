"""ui/tabs/calculator.py — калькулятор PRO."""
from __future__ import annotations
import math
import streamlit as st


def _poisson_matrix(lam_h, lam_a, n):
    def p(l, k):
        return math.exp(-l) * l ** k / math.factorial(k)
    M = [[p(lam_h, i) * p(lam_a, j) for j in range(n)] for i in range(n)]
    tot = sum(map(sum, M)) or 1.0
    p1 = sum(M[i][j] for i in range(n) for j in range(n) if i > j) / tot
    px = sum(M[i][i] for i in range(n)) / tot
    p2 = max(0.0, 1 - p1 - px)
    over = 1 - sum(M[i][j] for i in range(n) for j in range(n)
                   if i + j <= 2) / tot
    btts = sum(M[i][j] for i in range(1, n) for j in range(1, n)) / tot
    return {"p1": p1, "px": px, "p2": p2, "over": over, "btts": btts,
            "lam_h": lam_h, "lam_a": lam_a}


def _manual_poisson(ha, hd, hf, he, aa, ad, af, ae, n):
    LG_H, LG_A = 1.45, 1.20
    lam_h = (LG_H * (1 + ha * 0.25) * max(0.3, 1 - ad * 0.20)
             * (1 + hf * 0.10) * (1 + (he - 1500) / 1000 * 0.15) + 0.25)
    lam_a = (LG_A * (1 + aa * 0.25) * max(0.3, 1 - hd * 0.20)
             * (1 + af * 0.10) * (1 + (ae - 1500) / 1000 * 0.15))
    lam_h = max(0.2, min(4.5, lam_h))
    lam_a = max(0.2, min(4.5, lam_a))
    return _poisson_matrix(lam_h, lam_a, int(n))


def render(matrix_n):
    D = st.session_state.data
    st.header("Калькулятор PRO")
    ca, cb = st.columns(2)
    with ca:
        st.text_input("Хозяева", "Home FC")
        ha = st.slider("Атака хозяев", -2.0, 2.0, 0.25, 0.05)
        hd = st.slider("Защита хозяев", -2.0, 2.0, 0.0, 0.05)
        hf = st.slider("Форма хозяев", -1.0, 1.0, 0.1, 0.05)
        he = st.number_input("Elo хозяев", 1000, 2200, 1500, 10)
    with cb:
        st.text_input("Гости", "Away FC")
        aa = st.slider("Атака гостей", -2.0, 2.0, 0.0, 0.05)
cat > ui/tabs/calculator.py <<'EOF'
"""ui/tabs/calculator.py — калькулятор PRO."""
from __future__ import annotations
import math
import streamlit as st


def _poisson_matrix(lam_h, lam_a, n):
    def p(l, k):
        return math.exp(-l) * l ** k / math.factorial(k)
    M = [[p(lam_h, i) * p(lam_a, j) for j in range(n)] for i in range(n)]
    tot = sum(map(sum, M)) or 1.0
    p1 = sum(M[i][j] for i in range(n) for j in range(n) if i > j) / tot
    px = sum(M[i][i] for i in range(n)) / tot
    p2 = max(0.0, 1 - p1 - px)
    over = 1 - sum(M[i][j] for i in range(n) for j in range(n)
                   if i + j <= 2) / tot
    btts = sum(M[i][j] for i in range(1, n) for j in range(1, n)) / tot
    return {"p1": p1, "px": px, "p2": p2, "over": over, "btts": btts,
            "lam_h": lam_h, "lam_a": lam_a}


def _manual_poisson(ha, hd, hf, he, aa, ad, af, ae, n):
    LG_H, LG_A = 1.45, 1.20
    lam_h = (LG_H * (1 + ha * 0.25) * max(0.3, 1 - ad * 0.20)
             * (1 + hf * 0.10) * (1 + (he - 1500) / 1000 * 0.15) + 0.25)
    lam_a = (LG_A * (1 + aa * 0.25) * max(0.3, 1 - hd * 0.20)
             * (1 + af * 0.10) * (1 + (ae - 1500) / 1000 * 0.15))
    lam_h = max(0.2, min(4.5, lam_h))
    lam_a = max(0.2, min(4.5, lam_a))
    return _poisson_matrix(lam_h, lam_a, int(n))


def render(matrix_n):
    D = st.session_state.data
    st.header("Калькулятор PRO")
    ca, cb = st.columns(2)
    with ca:
        st.text_input("Хозяева", "Home FC")
        ha = st.slider("Атака хозяев", -2.0, 2.0, 0.25, 0.05)
        hd = st.slider("Защита хозяев", -2.0, 2.0, 0.0, 0.05)
        hf = st.slider("Форма хозяев", -1.0, 1.0, 0.1, 0.05)
        he = st.number_input("Elo хозяев", 1000, 2200, 1500, 10)
    with cb:
        st.text_input("Гости", "Away FC")
        aa = st.slider("Атака гостей", -2.0, 2.0, 0.0, 0.05)
        ad = st.slider("Защита гостей", -2.0, 2.0, 0.15, 0.05)
        af = st.slider("Форма гостей", -1.0, 1.0, -0.05, 0.05)
        ae = st.number_input("Elo гостей", 1000, 2200, 1500, 10)
    st.subheader("Рыночные коэффициенты 1X2")
    c1, c2, c3 = st.columns(3)
    o1 = c1.number_input("П1", 1.01, 100.0, 2.10, 0.01)
    ox = c2.number_input("X", 1.01, 100.0, 3.30, 0.01)
    o2 = c3.number_input("П2", 1.01, 100.0, 3.40, 0.01)
    mk = st.slider("Лимит Kelly", 0.01, 0.20, 0.05, 0.01)
    mev = st.slider("Минимальный EV", 0.00, 0.30, 0.03, 0.01)

    if st.button("Рассчитать", type="primary"):
        p = _manual_poisson(ha, hd, hf, he, aa, ad, af, ae, matrix_n)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("xG хозяев", f"{p['lam_h']:.2f}")
        m2.metric("xG гостей", f"{p['lam_a']:.2f}")
        m3.metric("Тотал xG", f"{p['lam_h']+p['lam_a']:.2f}")
        m4.metric("Обе забьют", f"{p['btts']*100:.1f}%")
        m5.metric("ТБ 2.5", f"{p['over']*100:.1f}%")
        rows = []
        for key, label, od in [("p1", "П1", o1), ("px", "X", ox),
                                ("p2", "П2", o2)]:
            prob = p[key]
            fair = 1.0 / max(prob, 0.01)
            ev = prob * od - 1
            k = max(0.0, min((prob * (od - 1) - (1 - prob)) / (od - 1), mk))
            stake = round(k * D["bank"], 2)
            rows.append({"Исход": label,
                         "Вероятность": f"{prob*100:.1f}%",
                         "Fair": f"{fair:.2f}",
                         "Рынок": f"{od:.2f}",
                         "EV": f"{ev*100:+.1f}%",
                         "Kelly": f"{k*100:.1f}%",
                         "Ставка": f"{stake:.2f}"})
        st.dataframe(rows, use_container_width=True, hide_index=True)
