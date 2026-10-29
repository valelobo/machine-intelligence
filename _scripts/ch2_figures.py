"""Generate the figures, animations and result tables for Chapter 2.

Run from the repo root:  python3 _scripts/ch2_figures.py

Writes into ch2/chapter-02-linear-regression/:
- figures/*.svg
- animations/least-squares.html and animations/sampling.html (generated; edit
  this script, not those files)
- _tables/*.md, Markdown snippets with the fitted numbers, included in
  chapter.qmd with {{< include _tables/<name>.md >}}

Study object (same as Chapter 1): annual income (thousand R$) of a person.
The population uses the S-shaped income_f(education) from figlib, plus
- experience: each year adds income, more so for people with more education
  (an interaction), and
- age = 6 + education + experience + noise, which has NO effect of its own
  but is strongly correlated with experience (collinearity).
Numbers use the Brazilian format (decimal comma).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: E402,F403  palette, svg(), text(), Axes, points(), step(), stepper_page(), income_f()

CHAPTER = Path("ch2/chapter-02-linear-regression")
OUT = CHAPTER / "figures"
ANIM = CHAPTER / "animations"
TABLES = CHAPTER / "_tables"
for d in (OUT, ANIM, TABLES):
    d.mkdir(parents=True, exist_ok=True)
for old in TABLES.glob("*.md"):
    old.unlink()  # tables are regenerated from scratch
GEN = "_scripts/ch2_figures.py"
SIGMA = 6.0
XLIM_EDU = (10, 22)


# ------------------------------------------------------------------ utilities

def br(x, d=3):
    """Number in Brazilian format: 1234.5 -> '1234,500'."""
    return f"{x:.{d}f}".replace(".", ",")


def pval(p):
    return "< 0,0001" if p < 1e-4 else br(p, 4)


def ols(X, y):
    """Least squares with an intercept, plus the usual inference."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    X1 = np.column_stack([np.ones(len(y)), X])
    n, k = X1.shape
    xtx_inv = np.linalg.inv(X1.T @ X1)
    w = xtx_inv @ X1.T @ y
    fitted = X1 @ w
    res = y - fitted
    rss = float(res @ res)
    tss = float(((y - y.mean()) ** 2).sum())
    sigma2 = rss / (n - k)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t = w / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    F = ((tss - rss) / (k - 1)) / (rss / (n - k))
    return dict(w=w, se=se, t=t, p=p, fitted=fitted, res=res, rss=rss, tss=tss,
                rse=np.sqrt(sigma2), r2=1 - rss / tss, F=F, pF=stats.f.sf(F, k - 1, n - k),
                n=n, k=k, xtx_inv=xtx_inv, sigma2=sigma2)


def coef_table(fit, names):
    rows = ["| | *Coefficient* | *Std. error* | *t-statistic* | *p-value* |", "|---|---:|---:|---:|---:|"]
    for name, w, se, t, p in zip(names, fit["w"], fit["se"], fit["t"], fit["p"]):
        rows.append(f"| {name} | {br(w, 4)} | {br(se, 4)} | {br(t, 2)} | {pval(p)} |")
    return "\n".join(rows) + "\n"


def write_table(name, md):
    (TABLES / f"{name}.md").write_text(md)


def nice_ticks(vmax, n=5):
    """Round tick values from 0 up to vmax (about n of them)."""
    for st in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if vmax / st <= n:
            return [round(v, 4) for v in np.arange(0, vmax + 1e-9, st)]
    return [0, vmax]


HALO = 'paint-order="stroke" stroke="#ffffff" stroke-width="5" stroke-linejoin="round"'


def frame_num(ax, xlabel, ylabel, xticks, yticks, xfmt="{:g}", yfmt="{:g}"):
    return ax.frame(xlabel, ylabel, lowhigh=False) + ax.ticks(xticks, yticks, xfmt, yfmt)


def segments(ax, x, y, target, colour, width=0.9, opacity=0.55):
    out = ""
    for u, v, t in zip(x, y, target):
        (a, b), (_, c) = ax.px(u, v), ax.px(u, t)
        out += (f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{a:.1f}" y2="{c:.1f}" stroke="{colour}" '
                f'stroke-width="{width}" stroke-opacity="{opacity}"/>\n')
    return out


# --------------------------------------------------------------- the data

def income_people(n=200, seed=1):
    """Main data set: education, experience, age and income for n people."""
    rng = np.random.default_rng(seed)
    educ = rng.uniform(*XLIM_EDU, n)
    exper = rng.uniform(0, 35, n)
    age = 6 + educ + exper + rng.normal(0, 2.5, n)
    income = (income_f(educ) + 0.6 * exper + 0.08 * (educ - 16) * exper
              + rng.normal(0, SIGMA, n))
    return educ, exper, age, income


EDUC, EXPER, AGE, INCOME = income_people()


def chapter1_sample(n, seed):
    """A fresh sample of income vs education from the Chapter 1 population."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(*XLIM_EDU, n))
    return x, income_f(x) + rng.normal(0, SIGMA, n)


def edu_axes(left, top, width=420, height=280, ymax=120):
    return Axes(left, top, width, height, xlim=XLIM_EDU, ylim=(0, ymax))


def edu_frame(ax, ylabel="renda (mil R$/ano)"):
    step_y = 20 if ax.ylim[1] > 60 else 10
    return frame_num(ax, "anos de estudo (x)", ylabel, range(10, 23, 2), range(0, int(ax.ylim[1]) + 1, step_y))


# -------------------------------------------------- 2.1 simple regression

def edu_income():
    fit = ols(EDUC, INCOME)
    ax = edu_axes(80, 20, 480, 290)
    body = edu_frame(ax) + segments(ax, EDUC, INCOME, fit["fitted"], AXIS, 0.8, 0.6)
    body += points(ax, EDUC, INCOME, RED, 3.6, 0.75)
    w0, w1 = fit["w"]
    body += f'<path d="{ax.path(lambda x: w0 + w1 * x)}" fill="none" stroke="{BLUE}" stroke-width="3.5"/>\n'
    body += text(ax.l + 12, ax.t + 16, f"ŷ = {br(w0, 2)} + {br(w1, 3)} · estudo", 15, BLUE, weight=700)
    (OUT / "edu-income.svg").write_text(svg(600, 370, body))
    return fit


def rss_label(mx, my, label, colour, k):
    """Place a marker label inside the parabola so it never crosses the curve:
    right of the left point, left of the right point, below the minimum."""
    if k == 0:
        return text(mx + 14, my + 5, label, 13, colour, "start", 700)
    if k == 1:
        return text(mx - 14, my + 5, label, 13, colour, "end", 700)
    return text(mx, my + 26, label, 13, colour, "middle", 700)


def least_squares_animation(fit):
    xbar, ybar = EDUC.mean(), INCOME.mean()

    def rss_for(w1):
        w0 = ybar - w1 * xbar  # every candidate line passes through (x̄, ȳ)
        return float(((INCOME - w0 - w1 * EDUC) ** 2).sum()) / 1000  # in thousands

    left = edu_axes(80, 30, 380, 290)
    slopes = np.linspace(0, 14, 141)
    rss = np.array([rss_for(s) for s in slopes])
    ticks = nice_ticks(rss.max())
    ymax = ticks[-1] if ticks[-1] >= rss.max() else ticks[-1] + ticks[1]
    right = Axes(590, 30, 360, 290, xlim=(0, 14), ylim=(0, ymax))
    cands = [(2.0, AMBER, "A"), (12.0, TEAL, "B"), (float(fit["w"][1]), BLUE, "LS")]

    cx, cy = left.px(xbar, ybar)
    body = step(1, edu_frame(left) + points(left, EDUC, INCOME, POINT, 3.2, 0.75)
                + f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{INK}" stroke="#fff" stroke-width="2"/>\n'
                + text(cx + 12, cy + 26, "(x̄, ȳ)", 15, INK, weight=700, extra=HALO)
                + frame_num(right, "w₁ (inclinação)", "RSS (mil)", range(0, 15, 2), nice_ticks(ymax), "{:g}", "{:g}"))
    for k, (w1, colour, name) in enumerate(cands):
        w0 = ybar - w1 * xbar
        res = segments(left, EDUC, INCOME, w0 + w1 * EDUC, colour)
        line = f'<path class="draw" clip-path="url(#plotL)" d="{left.path(lambda x: w0 + w1 * x)}" fill="none" stroke="{colour}" stroke-width="3.5"/>\n'
        mx, my = right.px(w1, rss_for(w1))
        marker = (f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="7" fill="{colour}"/>\n'
                  + rss_label(mx, my, f"{name}: RSS = {br(rss_for(w1), 1)} mil", colour, k))
        n = (2, 3, 5)[k]  # candidates A and B show for one step; LS stays
        until = n if k < 2 else None
        body += step(n, res + line, until=until)
        body += step(n, marker)
    body += step(4, f'<path class="draw" d="{right.polyline(slopes, rss)}" fill="none" stroke="{INK}" stroke-width="3"/>\n')
    w0, w1 = fit["w"]
    body += step(6, text(left.l + 10, left.t + 18, f"ŵ₀ = {br(w0, 2)}   ŵ₁ = {br(w1, 3)}", 15, BLUE, weight=700))
    defs = f'<defs><clipPath id="plotL"><rect x="{left.l}" y="{left.t}" width="{left.w}" height="{left.h}"/></clipPath></defs>\n'
    steps = [
        ("Os dados", "200 pessoas: anos de estudo (x) e renda anual (y), a mesma população do Cap. 1. A reta de <em>least squares</em> passa pelo ponto médio (x̄, ȳ)."),
        ("Uma reta qualquer", "Para cada pessoa, o <em>residual</em> eᵢ = yᵢ − ŷᵢ é a distância vertical até a reta. O RSS soma os quadrados: RSS = Σ eᵢ²."),
        ("Outra reta", "Uma inclinação grande demais também deixa resíduos grandes. Cada reta tem o seu RSS."),
        ("RSS em função de w₁", "Variando a inclinação, o RSS desenha uma parábola. Queremos o fundo dela."),
        ("A reta de <em>least squares</em>", "O mínimo do RSS: a reta que melhor se ajusta no sentido dos quadrados mínimos."),
        ("Os coeficientes", "ŵ₁ = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²  e  ŵ₀ = ȳ − ŵ₁x̄. Cada ano de estudo a mais está associado, em média, a ŵ₁ mil R$ a mais por ano."),
    ]
    stepper_page(ANIM / "least-squares.html", "Least squares", "0 0 1000 370", defs + body, steps,
                 "Renda versus anos de estudo, retas candidatas com seus resíduos e o RSS em função da inclinação, com mínimo na reta de least squares",
                 generator=GEN)


def sampling_animation():
    rng = np.random.default_rng(11)
    n, sigma, reps = 100, 8.0, 1000
    true_w0, true_w1 = -25.0, 4.5
    f_true = lambda x: true_w0 + true_w1 * x
    left = edu_axes(80, 30, 380, 290)
    est, samples = [], []
    for r in range(reps):
        x = rng.uniform(*XLIM_EDU, n)
        y = f_true(x) + rng.normal(0, sigma, n)
        w0, w1 = ols(x, y)["w"]
        est.append(w1)
        if r < 11:
            samples.append((x, y, w0, w1))
    est = np.array(est)
    se_theory = sigma / np.sqrt(n * (12 ** 2 / 12))  # Var(x) for U(10, 22) is 12²/12

    body = step(1, edu_frame(left)
                + f'<path d="{left.path(f_true)}" fill="none" stroke="{RED}" stroke-width="3.5"/>\n'
                + text(left.l + 10, left.t + 18, "f(x) = −25 + 4,5x  (population regression line)", 14, RED, weight=700))
    x, y, w0, w1 = samples[0]
    body += step(1, points(left, x, y, POINT, 3.2, 0.8), until=2)
    body += step(2, f'<path class="draw" d="{left.path(lambda q: w0 + w1 * q)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
                 + text(left.l + 10, left.t + 40, f"amostra 1: ŵ₀ = {br(w0, 1)}, ŵ₁ = {br(w1, 2)}", 14, BLUE, weight=700))
    more = "".join(f'<path d="{left.path(lambda q, a=a, b=b: a + b * q)}" fill="none" stroke="#8fb0de" stroke-width="1.6"/>\n'
                   for _, _, a, b in samples[1:])
    body += step(3, more + f'<path d="{left.path(f_true)}" fill="none" stroke="{RED}" stroke-width="3.5"/>\n')

    lo, hi = true_w1 - 0.9, true_w1 + 0.9
    counts, edges = np.histogram(est, bins=30, range=(lo, hi))
    right = Axes(590, 30, 360, 290, xlim=(lo, hi), ylim=(0, counts.max() * 1.15))
    bars = ""
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        (x0, y0), (x1, _) = right.px(a, c), right.px(b, 0)
        bars += f'<rect x="{x0 + 0.5:.1f}" y="{y0:.1f}" width="{x1 - x0 - 1:.1f}" height="{right.t + right.h - y0:.1f}" fill="{BLUE}" fill-opacity="0.55"/>\n'
    tx, _ = right.px(true_w1, 0)
    body += step(4, frame_num(right, "ŵ₁ em 1000 amostras", "frequência", [3.8, 4.1, 4.4, 4.7, 5.0, 5.3], [], "{:.1f}")
                 + bars
                 + f'<line x1="{tx:.1f}" y1="{right.t}" x2="{tx:.1f}" y2="{right.t + right.h}" stroke="{RED}" stroke-width="2.5"/>\n'
                 + text(tx + 6, right.t + 14, f"w₁ = 4,5 · média de ŵ₁ = {br(est.mean(), 3)}", 13, RED, weight=700))
    a, _ = right.px(true_w1 - 2 * se_theory, 0)
    b, _ = right.px(true_w1 + 2 * se_theory, 0)
    yb = right.t + right.h + 52
    body += step(5, f'<path d="M{a:.1f} {yb} H{b:.1f} M{a:.1f} {yb - 6} V{yb + 6} M{b:.1f} {yb - 6} V{yb + 6}" stroke="{PURPLE}" stroke-width="2.5" fill="none"/>\n'
                 + text((a + b) / 2, yb + 22, f"±2 SE(ŵ₁) = ±{br(2 * se_theory, 2)}: ~95% das estimativas", 13, PURPLE, "middle", 700))
    steps = [
        ("A <em>population regression line</em>", "Suponha que a renda média cresça linearmente com o estudo: y = −25 + 4,5x + ε. Na prática não conhecemos essa reta; aqui, por ser simulação, conhecemos."),
        ("Uma amostra, uma estimativa", "Com 100 pessoas, a reta de <em>least squares</em> fica perto da verdadeira, mas não igual."),
        ("Outras amostras", "Cada nova amostra de 100 pessoas dá uma reta diferente. As estimativas variam em torno da reta verdadeira."),
        ("<em>Unbiased</em>", "Em 1000 amostras, a média de ŵ₁ coincide com w₁ = 4,5: o estimador não tem viés (<em>unbiased</em>)."),
        ("<em>Standard error</em>", "O SE mede o quanto ŵ₁ varia de amostra para amostra. ŵ₁ ± 2·SE(ŵ₁) é aproximadamente um <em>confidence interval</em> de 95%."),
    ]
    stepper_page(ANIM / "sampling.html", "Population line e least squares lines", "0 0 1000 420", body, steps,
                 "Reta populacional da renda em função do estudo e retas de least squares de várias amostras, e o histograma das estimativas da inclinação",
                 generator=GEN)
    print(f"sampling: mean w1 = {est.mean():.3f}, sd = {est.std():.3f}, SE theory = {se_theory:.3f}")


def tss_rss():
    x, y = chapter1_sample(15, 5)
    fit = ols(x, y)
    w0, w1 = fit["w"]
    ybar = y.mean()
    body = ""
    for i, (title, colour, target) in enumerate([
            (f"TSS = Σ(yᵢ − ȳ)² = {br(fit['tss'], 0)}", AMBER, lambda q: np.full_like(np.asarray(q, float), ybar)),
            (f"RSS = Σ(yᵢ − ŷᵢ)² = {br(fit['rss'], 0)}", BLUE, lambda q: w0 + w1 * np.asarray(q, float))]):
        ax = edu_axes(80 + i * 450, 40, 350, 250, ymax=90)
        body += text(ax.l, 24, title, 16, colour, weight=700)
        body += edu_frame(ax, "renda")
        body += segments(ax, x, y, target(x), colour, 2, 1)
        body += f'<path d="{ax.path(target)}" fill="none" stroke="{colour}" stroke-width="3"/>\n'
        body += points(ax, x, y, INK, 5)
    body += text(470, 368, f"R² = 1 − RSS/TSS = {br(fit['r2'], 3)}: a fração da variabilidade da renda explicada pelo estudo", 16, PURPLE, "middle", 700)
    (OUT / "tss-rss.svg").write_text(svg(940, 385, body))


# ------------------------------------------------- 2.2 multiple regression

def regression_plane(fit2):
    rng = np.random.default_rng(2)
    idx = rng.choice(len(EDUC), 70, replace=False)
    w0, w1, w2 = fit2["w"]
    O, U, V, Z = np.array([110, 330]), np.array([360, 45]), np.array([190, -95]), np.array([0, -215])

    def P(e, x, inc):
        q = O + ((e - 10) / 12) * U + (x / 35) * V + (inc / 115) * Z
        return q[0], q[1]

    def plane(e, x):
        return w0 + w1 * e + w2 * x

    body = ""
    for a, b, lab, d in [((10, 0, 0), (22, 0, 0), "estudo", (30, 14)), ((10, 0, 0), (10, 35, 0), "experiência", (44, -2)),
                         ((10, 0, 0), (10, 0, 115), "renda", (0, -10))]:
        x0, y0 = P(*a)
        x1, y1 = P(*b)
        body += f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{AXIS}" stroke-width="1.5"/>\n'
        body += text(x1 + d[0], y1 + d[1], lab, 15, MUTED, "middle", 700)
    order = sorted(idx, key=lambda i: -EXPER[i])
    behind = [i for i in order if INCOME[i] < plane(EDUC[i], EXPER[i])]
    front = [i for i in order if INCOME[i] >= plane(EDUC[i], EXPER[i])]

    def draw(ids, colour):
        out = ""
        for i in ids:
            x0, y0 = P(EDUC[i], EXPER[i], INCOME[i])
            x1, y1 = P(EDUC[i], EXPER[i], plane(EDUC[i], EXPER[i]))
            out += f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{colour}" stroke-width="1.2" stroke-opacity="0.8"/>\n'
            out += f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="4.2" fill="{colour}"/>\n'
        return out

    body += draw(behind, "#c98c97")
    corners = [P(10, 0, plane(10, 0)), P(22, 0, plane(22, 0)), P(22, 35, plane(22, 35)), P(10, 35, plane(10, 35))]
    body += f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in corners)}" fill="{BLUE}" fill-opacity="0.18" stroke="{BLUE}" stroke-width="1.5"/>\n'
    for e in np.linspace(10, 22, 7):
        (a, b), (c, d) = P(e, 0, plane(e, 0)), P(e, 35, plane(e, 35))
        body += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" stroke="{BLUE}" stroke-opacity="0.35"/>\n'
    for x in np.linspace(0, 35, 6):
        (a, b), (c, d) = P(10, x, plane(10, x)), P(22, x, plane(22, x))
        body += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" stroke="{BLUE}" stroke-opacity="0.35"/>\n'
    body += draw(front, RED)
    body += text(30, 26, f"ŷ = {br(w0, 1)} + {br(w1, 2)}·estudo + {br(w2, 2)}·experiência", 16, BLUE, weight=700)
    body += text(30, 48, "pessoas acima do plano em vermelho, abaixo em rosa", 13, MUTED)
    (OUT / "regression-plane.svg").write_text(svg(720, 400, body))


def interaction_figure(fit_main, fit_int):
    body = ""
    lo, hi = EDUC < 13, EDUC > 19
    for i, (title, line) in enumerate([
            ("Sem interaction: retas paralelas", lambda x, e: fit_main["w"][0] + fit_main["w"][1] * e + fit_main["w"][2] * x),
            ("Com interaction (estudo × experiência)", lambda x, e: fit_int["w"][0] + fit_int["w"][1] * e + fit_int["w"][2] * x + fit_int["w"][3] * e * x)]):
        ax = Axes(80 + i * 470, 40, 370, 260, xlim=(0, 35), ylim=(0, 120))
        body += text(ax.l, 24, title, 16, INK, weight=700)
        body += frame_num(ax, "anos de experiência", "renda (mil R$/ano)", range(0, 36, 5), range(0, 121, 20))
        body += points(ax, EXPER[lo], INCOME[lo], BLUE, 3.6, 0.7) + points(ax, EXPER[hi], INCOME[hi], AMBER, 3.6, 0.7)
        body += f'<path d="{ax.path(lambda x: line(x, 11.5))}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
        body += f'<path d="{ax.path(lambda x: line(x, 20.5))}" fill="none" stroke="{AMBER}" stroke-width="3"/>\n'
    body += (f'<circle cx="86" cy="368" r="5" fill="{BLUE}"/>' + text(98, 373, "pouco estudo (&lt; 13 anos)", 14)
             + f'<circle cx="316" cy="368" r="5" fill="{AMBER}"/>' + text(328, 373, "muito estudo (&gt; 19 anos)", 14))
    (OUT / "interaction.svg").write_text(svg(940, 385, body))


def income_tables():
    names3 = ["estudo", "experiência", "idade"]
    simple = {n: ols(x, INCOME) for n, x in zip(names3, (EDUC, EXPER, AGE))}
    write_table("simple-educ", coef_table(simple["estudo"], ["Intercept ($w_0$)", "estudo ($w_1$)"]))
    f = simple["estudo"]
    write_table("simple-educ-fit",
                "| *Quantity* | Valor |\n|---|---:|\n"
                f"| *Residual standard error* (RSE) | {br(f['rse'], 2)} |\n"
                f"| $R^2$ | {br(f['r2'], 3)} |\n"
                f"| *F-statistic* | {br(f['F'], 1)} |\n")
    rows = ["| Regressão simples | $\\hat w_1$ | *Std. error* | *t-statistic* | *p-value* |", "|---|---:|---:|---:|---:|"]
    for name, fit in simple.items():
        rows.append(f"| renda ~ {name} | {br(fit['w'][1], 4)} | {br(fit['se'][1], 4)} | {br(fit['t'][1], 2)} | {pval(fit['p'][1])} |")
    write_table("simple-each", "\n".join(rows) + "\n")

    multi = ols(np.column_stack([EDUC, EXPER, AGE]), INCOME)
    write_table("multiple", coef_table(multi, ["Intercept", "estudo", "experiência", "idade"]))
    write_table("multiple-fit",
                "| *Quantity* | Valor |\n|---|---:|\n"
                f"| RSE | {br(multi['rse'], 2)} |\n"
                f"| $R^2$ | {br(multi['r2'], 3)} |\n"
                f"| *F-statistic* | {br(multi['F'], 1)} |\n")
    C = np.corrcoef(np.vstack([EDUC, EXPER, AGE, INCOME]))
    names = names3 + ["renda"]
    rows = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * 4]
    for i, n in enumerate(names):
        rows.append(f"| **{n}** | " + " | ".join(br(C[i, j], 2) for j in range(4)) + " |")
    write_table("correlation", "\n".join(rows) + "\n")
    write_table("corr-age-exper",
                f"A correlação entre idade e experiência é {br(C[1, 2], 2)}: quem tem mais "
                "experiência em geral é mais velho. Sozinha, a idade \"empresta\" o efeito da experiência.\n")

    main = ols(np.column_stack([EDUC, EXPER]), INCOME)
    inter = ols(np.column_stack([EDUC, EXPER, EDUC * EXPER]), INCOME)
    write_table("interaction", coef_table(inter, ["Intercept", "estudo", "experiência", "estudo × experiência"]))

    # Confidence vs prediction interval for 16 years of study, 10 of experience
    x0 = np.array([1, 16, 10])
    yhat = float(x0 @ main["w"])
    se_mean = float(np.sqrt(x0 @ main["xtx_inv"] @ x0 * main["sigma2"]))
    se_pred = float(np.sqrt(main["sigma2"] + se_mean ** 2))
    q = stats.t.ppf(0.975, main["n"] - main["k"])
    write_table("intervals",
                "| estudo = 16, experiência = 10 | Intervalo de 95% (mil R$/ano) |\n|---|---|\n"
                f"| *Confidence interval* (renda **média** dessas pessoas) | [{br(yhat - q * se_mean, 1)}; {br(yhat + q * se_mean, 1)}] |\n"
                f"| *Prediction interval* (renda de **uma** pessoa) | [{br(yhat - q * se_pred, 1)}; {br(yhat + q * se_pred, 1)}] |\n")

    # Is the relationship linear? Compare with a cubic in education (F-test).
    cubic = ols(np.column_stack([EDUC, EDUC ** 2, EDUC ** 3, EXPER]), INCOME)
    F_nl = ((main["rss"] - cubic["rss"]) / 2) / (cubic["rss"] / (cubic["n"] - cubic["k"]))
    p_nl = stats.f.sf(F_nl, 2, cubic["n"] - cubic["k"])

    vif_age = 1 / (1 - ols(np.column_stack([EDUC, EXPER]), AGE)["r2"])
    ci = lambda fit, j, d=2: f"[{br(fit['w'][j] - 1.96 * fit['se'][j], d)}; {br(fit['w'][j] + 1.96 * fit['se'][j], d)}]"
    write_table("study",
                "| Pergunta | Resposta com os dados |\n|---|---|\n"
                f"| 1. Há relação entre as *features* e a renda? | Sim: *F-statistic* = {br(multi['F'], 1)}, *p-value* {pval(multi['pF'])}. |\n"
                f"| 2. Quão forte é a relação? | RSE = {br(multi['rse'], 1)} mil R$/ano (~{br(100 * multi['rse'] / INCOME.mean(), 0)}% da renda média); $R^2$ = {br(multi['r2'], 3)}. |\n"
                f"| 3. Quais *features* contribuem? | estudo e experiência (*p-values* {pval(multi['p'][1])} e {pval(multi['p'][2])}); idade, não (*p-value* = {pval(multi['p'][3])}, VIF = {br(vif_age, 0)}). |\n"
                f"| 4. Qual o tamanho de cada efeito? | Por ano a mais, *confidence intervals* de 95% (mil R$/ano): estudo {ci(main, 1)}, experiência {ci(main, 2)}. |\n"
                "| 5. Com que precisão prevemos? | Renda média de um grupo: *confidence interval*; de uma pessoa: *prediction interval*, bem mais largo (Seção 2.2.2). |\n"
                f"| 6. A relação é linear? | Não: acrescentar estudo² e estudo³ reduz o RSS significativamente (*F-statistic* = {br(F_nl, 1)}, *p-value* {pval(p_nl)}); a forma em S do Cap. 1 reaparece. |\n"
                f"| 7. Há sinergia entre estudo e experiência? | Sim: o termo estudo × experiência tem *p-value* {pval(inter['p'][3])} e o $R^2$ sobe de {br(main['r2'], 3)} para {br(inter['r2'], 3)}. |\n")
    print(f"income: simple slope {simple['estudo']['w'][1]:.3f}, multi R2 {multi['r2']:.3f}, age p multi {multi['p'][3]:.3f}, "
          f"age p simple {simple['idade']['p'][1]:.2g}, exper p multi {multi['p'][2]:.2g}, VIF(age) {vif_age:.1f}, "
          f"interaction p {inter['p'][3]:.2g}, R2 {main['r2']:.3f}->{inter['r2']:.3f}, nonlinearity p {p_nl:.2g}")
    return main, inter


# ------------------------------------------------------- 2.3 other topics

def poly_fit(x, y, deg):
    """Polynomial least squares on a centred, scaled x (well conditioned)."""
    c = np.polyfit((x - 16) / 6, y, deg)
    return lambda q: np.polyval(c, (np.asarray(q, float) - 16) / 6)


def polynomial_figure():
    x, y = chapter1_sample(100, 8)
    ax = edu_axes(80, 20, 480, 290, ymax=100)
    body = edu_frame(ax)
    body += points(ax, x, y, POINT, 3.6, 0.75)
    for k, (deg, colour, name) in enumerate([(1, AMBER, "Linear"), (3, BLUE, "Degree 3"), (10, TEAL, "Degree 10")]):
        dash = ' stroke-dasharray="7 5"' if deg == 10 else ""
        body += f'<path clip-path="url(#pp)" d="{ax.path(poly_fit(x, y, deg))}" fill="none" stroke="{colour}" stroke-width="3"{dash}/>\n'
        body += text(ax.l + 12, ax.t + 18 + 20 * k, name, 14, colour, weight=700)
    defs = f'<defs><clipPath id="pp"><rect x="{ax.l}" y="{ax.t}" width="{ax.w}" height="{ax.h}"/></clipPath></defs>\n'
    (OUT / "polynomial.svg").write_text(svg(600, 370, defs + body))


def smooth(x, y, h):
    xs = np.linspace(x.min(), x.max(), 120)
    w = np.exp(-0.5 * ((xs[:, None] - x[None, :]) / h) ** 2)
    return xs, (w @ y) / w.sum(1)


def residual_panel(ax, fitted, res, title):
    out = text(ax.l, ax.t - 14, title, 15, INK, weight=700)
    out += frame_num(ax, "Fitted values", "Residuals", [], [])
    _, z = ax.px(ax.xlim[0], 0)
    out += f'<line x1="{ax.l}" y1="{z:.1f}" x2="{ax.l + ax.w}" y2="{z:.1f}" stroke="{AXIS}" stroke-dasharray="5 5"/>\n'
    out += points(ax, fitted, res, POINT, 3.2, 0.75)
    xs, ys = smooth(fitted, res, (fitted.max() - fitted.min()) / 10)
    out += f'<path d="{ax.polyline(xs, ys)}" fill="none" stroke="{RED}" stroke-width="3"/>\n'
    return out


def residual_plots():
    x, y = chapter1_sample(200, 8)
    body = ""
    for i, deg in enumerate((1, 3)):
        f = poly_fit(x, y, deg)
        fv, r = f(x), y - f(x)
        m = abs(r).max() * 1.1
        ax = Axes(60 + i * 310, 40, 250, 220, xlim=(fv.min() - 1, fv.max() + 1), ylim=(-m, m))
        body += residual_panel(ax, fv, r, "Linear: padrão em S" if deg == 1 else "Degree 3: sem padrão")
    rng = np.random.default_rng(4)
    xe = rng.uniform(*XLIM_EDU, 200)
    ye = (-25 + 4.5 * xe) * np.exp(rng.normal(0, 0.22, 200))  # multiplicative error
    fit = ols(xe, ye)
    m = abs(fit["res"]).max() * 1.1
    ax = Axes(680, 40, 250, 220, xlim=(fit["fitted"].min() - 1, fit["fitted"].max() + 1), ylim=(-m, m))
    body += residual_panel(ax, fit["fitted"], fit["res"], "Heteroscedasticity: funil")
    (OUT / "residual-plots.svg").write_text(svg(960, 320, body))


def outlier_leverage():
    rng = np.random.default_rng(6)
    x = rng.uniform(*XLIM_EDU, 30)
    y = -25 + 4.5 * x + rng.normal(0, 5, 30)
    body = ""
    cases = [("Outlier: renda incomum", (15, 115), RED), ("High leverage: estudo incomum", (30, 40), PURPLE)]
    for i, (title, (px_, py_), colour) in enumerate(cases):
        ax = Axes(80 + i * 450, 40, 350, 250, xlim=(9, 32), ylim=(0, 130))
        xa, ya = np.append(x, px_), np.append(y, py_)
        with_pt, without = ols(xa, ya), ols(x, y)
        body += text(ax.l, 24, title, 16, colour, weight=700)
        body += frame_num(ax, "anos de estudo", "renda", range(10, 33, 5), range(0, 131, 40))
        body += points(ax, x, y, POINT, 4)
        a, b = ax.px(px_, py_)
        body += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="7" fill="{colour}"/>\n'
        body += f'<path d="{ax.path(lambda q: without["w"][0] + without["w"][1] * q)}" fill="none" stroke="{INK}" stroke-width="2.5" stroke-dasharray="7 6"/>\n'
        body += f'<path d="{ax.path(lambda q: with_pt["w"][0] + with_pt["w"][1] * q)}" fill="none" stroke="{colour}" stroke-width="3"/>\n'
        h = 1 / len(xa) + (px_ - xa.mean()) ** 2 / ((xa - xa.mean()) ** 2).sum()
        body += text(ax.l + ax.w, ax.t + ax.h - 10, f"leverage hᵢ = {br(h, 2)}", 14, colour, "end", 700)
    body += (f'<line x1="80" y1="358" x2="114" y2="358" stroke="{INK}" stroke-width="2.5" stroke-dasharray="7 6"/>' + text(122, 363, "ajuste sem o ponto", 14)
             + f'<line x1="300" y1="358" x2="334" y2="358" stroke="{MUTED}" stroke-width="3"/>' + text(342, 363, "ajuste com o ponto", 14))
    (OUT / "outlier-leverage.svg").write_text(svg(900, 380, body))


def collinearity():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body = ""
    panels = [((EDUC, EXPER), ("w (estudo)", "w (experiência)"), "estudo e experiência: pouco correlacionados"),
              ((EXPER, AGE), ("w (experiência)", "w (idade)"), "experiência e idade: collinearity")]
    for i, ((a, b), (la, lb), title) in enumerate(panels):
        fit = ols(np.column_stack([a, b]), INCOME)
        ca, cb = fit["w"][1:]
        Ra, Rb = 4.2 * fit["se"][1], 4.2 * fit["se"][2]  # each axis spans ±4.2 SE
        ga, gb = np.meshgrid(np.linspace(ca - Ra, ca + Ra, 220), np.linspace(cb - Rb, cb + Rb, 220))
        # centring removes the intercept from the picture (it is re-fitted for each pair)
        ac, bc, yc = a - a.mean(), b - b.mean(), INCOME - INCOME.mean()
        rss = ((yc[None, None, :] - ga[..., None] * ac - gb[..., None] * bc) ** 2).sum(-1)
        levels = fit["rss"] + fit["sigma2"] * np.array([1, 4, 9, 16])
        cs = plt.contour(ga, gb, rss, levels=levels)
        ax = Axes(80 + i * 450, 40, 300, 270, xlim=(ca - Ra, ca + Ra), ylim=(cb - Rb, cb + Rb))
        body += text(ax.l, 24, title, 15, INK, weight=700)
        body += ax.frame(la, lb, lowhigh=False)
        for segs in cs.allsegs:
            for seg in segs:
                body += f'<path d="{ax.polyline(seg[:, 0], seg[:, 1])}" fill="none" stroke="{BLUE}" stroke-width="1.8"/>\n'
        mx, my = ax.px(ca, cb)
        body += f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="6" fill="{RED}"/>\n'
        vif = 1 / (1 - ols(a, b)["r2"])
        body += text(ax.l + ax.w, ax.t + 16, f"VIF = {br(vif, 1)}", 15, RED, "end", 700)
        plt.close("all")
    body += text(470, 375, "Curvas de nível do RSS (eixos: ±4 standard errors). Com collinearity, muitas combinações ajustam quase igualmente bem", 14, MUTED, "middle", 600)
    (OUT / "collinearity.svg").write_text(svg(940, 390, body))


# ---------------------------------------------- 2.5 regression versus KNN

def knn_reg(xq, x, y, k):
    xq = np.asarray(xq, float)
    d = np.abs(xq[:, None] - x[None, :]) if x.ndim == 1 else ((xq[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    return y[idx].mean(1)


F_LIN = lambda x: -25 + 4.5 * np.asarray(x, float)


def knn_regression_figure():
    x, y = chapter1_sample(50, 12)
    q = np.linspace(*XLIM_EDU, 400)
    body = ""
    for i, k in enumerate((1, 9)):
        ax = edu_axes(80 + i * 450, 40, 350, 250, ymax=100)
        body += text(ax.l, 24, f"KNN regression, K = {k}", 16, INK, weight=700)
        body += edu_frame(ax, "renda")
        body += f'<path d="{ax.path(income_f)}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>\n'
        body += points(ax, x, y, POINT, 4)
        body += f'<path d="{ax.polyline(q, knn_reg(q, x, y, k))}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
    body += (f'<line x1="80" y1="358" x2="114" y2="358" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>' + text(122, 363, "f verdadeira (Cap. 1)", 14)
             + f'<line x1="300" y1="358" x2="334" y2="358" stroke="{BLUE}" stroke-width="3"/>' + text(342, 363, "f̂ do KNN: média da renda dos K vizinhos", 14))
    (OUT / "knn-regression.svg").write_text(svg(900, 380, body))


def test_mse(predict, f, rng, p=1, n_test=2000):
    Xt = rng.uniform(*XLIM_EDU, (n_test, p))
    return float(np.mean((f(Xt[:, 0]) - predict(Xt)) ** 2) + SIGMA ** 2)


def knn_vs_lr():
    n, reps = 50, 30
    ks = [1, 2, 3, 4, 5, 7, 9, 12, 16, 20, 25, 33, 50]
    body = ""
    for i, (f, title) in enumerate([(F_LIN, "Se a renda fosse linear no estudo"), (income_f, "Renda em S (a f do Cap. 1)")]):
        rng = np.random.default_rng(20 + i)
        knn_m, lr_m = np.zeros(len(ks)), 0.0
        for _ in range(reps):
            x = rng.uniform(*XLIM_EDU, n)
            y = f(x) + rng.normal(0, SIGMA, n)
            for j, k in enumerate(ks):
                knn_m[j] += test_mse(lambda X: knn_reg(X[:, 0], x, y, k), f, rng) / reps
            fit = ols(x, y)
            lr_m += test_mse(lambda X: fit["w"][0] + fit["w"][1] * X[:, 0], f, rng) / reps
        ymax = max(knn_m.max(), lr_m) * 1.1
        ax = Axes(80 + i * 450, 40, 350, 250, xlim=(0.02, 1), ylim=(0, ymax), xlog=True)
        body += text(ax.l, 24, title, 16, INK, weight=700)
        body += frame_num(ax, "1/K  (flexibilidade →)", "Test MSE", [0.02, 0.05, 0.1, 0.2, 0.5, 1], nice_ticks(ymax))
        _, ly = ax.px(0.02, lr_m)
        body += f'<line x1="{ax.l}" y1="{ly:.1f}" x2="{ax.l + ax.w}" y2="{ly:.1f}" stroke="{AMBER}" stroke-width="3" stroke-dasharray="8 6"/>\n'
        body += f'<path d="{ax.polyline([1 / k for k in ks], knn_m)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
        print(f"knn vs lr ({title}): LR {lr_m:.1f}, best KNN {knn_m.min():.1f}")
    body += (f'<line x1="80" y1="358" x2="114" y2="358" stroke="{AMBER}" stroke-width="3" stroke-dasharray="8 6"/>' + text(122, 363, "linear regression", 14)
             + f'<line x1="300" y1="358" x2="334" y2="358" stroke="{BLUE}" stroke-width="3"/>' + text(342, 363, "KNN", 14))
    (OUT / "knn-vs-lr.svg").write_text(svg(900, 380, body))


def curse_of_dimensionality():
    n, reps = 50, 30
    ps = [1, 2, 3, 4, 10, 20]
    ks = [1, 2, 3, 5, 7, 10, 15, 20]
    rng = np.random.default_rng(31)
    lr_m, knn_m = [], []
    for p in ps:
        lr_acc, knn_acc = 0.0, np.zeros(len(ks))
        for _ in range(reps):
            X = rng.uniform(*XLIM_EDU, (n, p))
            y = income_f(X[:, 0]) + rng.normal(0, SIGMA, n)
            fit = ols(X, y)
            lr_acc += test_mse(lambda T: fit["w"][0] + T @ fit["w"][1:], income_f, rng, p, 1000) / reps
            for j, k in enumerate(ks):
                knn_acc[j] += test_mse(lambda T: knn_reg(T, X, y, k), income_f, rng, p, 1000) / reps
        lr_m.append(lr_acc)
        knn_m.append(float(knn_acc.min()))
    ymax = max(max(lr_m), max(knn_m)) * 1.1
    ax = Axes(80, 30, 480, 270, xlim=(0, len(ps) - 1), ylim=(0, ymax))
    pos = list(range(len(ps)))
    body = frame_num(ax, "p (features; só os anos de estudo importam)", "Test MSE", [], nice_ticks(ymax))
    for j, p in enumerate(ps):
        x, _ = ax.px(j, 0)
        body += text(x, ax.t + ax.h + 19, str(p), 12, MUTED, "middle")
    body += f'<path d="{ax.polyline(pos, lr_m)}" fill="none" stroke="{AMBER}" stroke-width="3" stroke-dasharray="8 6"/>\n'
    body += f'<path d="{ax.polyline(pos, knn_m)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
    body += points(ax, pos, lr_m, AMBER, 5, 1) + points(ax, pos, knn_m, BLUE, 5, 1)
    body += (text(ax.l + ax.w, ax.px(len(ps) - 1, knn_m[-1])[1] - 12, "KNN (melhor K)", 14, BLUE, "end", 700)
             + text(ax.px(3.5, 0)[0], ax.px(3.5, lr_m[3])[1] + 26, "linear regression", 14, AMBER, "middle", 700))
    (OUT / "curse-of-dimensionality.svg").write_text(svg(600, 360, body))
    print("curse: LR", [round(v, 1) for v in lr_m], "KNN", [round(v, 1) for v in knn_m])


if __name__ == "__main__":
    for old in ("tv-sales.svg",):
        if (OUT / old).exists():
            (OUT / old).unlink()  # replaced by edu-income.svg
    fit = edu_income()
    least_squares_animation(fit)
    sampling_animation()
    tss_rss()
    main_fit, int_fit = income_tables()
    regression_plane(main_fit)
    interaction_figure(main_fit, int_fit)
    polynomial_figure()
    residual_plots()
    outlier_leverage()
    collinearity()
    knn_regression_figure()
    knn_vs_lr()
    curse_of_dimensionality()
