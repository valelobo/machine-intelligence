"""Generate the figures, animations and result tables for Chapter 2.

Run from the repo root:  python3 _scripts/ch2_figures.py

Writes into ch2/chapter-02-linear-regression/:
- figures/*.svg
- animations/least-squares.html and animations/sampling.html (generated; edit
  this script, not those files)
- _tables/*.md, Markdown snippets with the fitted numbers, included in
  chapter.qmd with {{< include _tables/<name>.md >}}

All data are simulated, in the spirit of ISLP ch. 3 (Advertising, Auto,
Credit). Numbers use the Brazilian format (decimal comma).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: E402,F403  palette, svg(), text(), Axes, points(), step(), stepper_page()

CHAPTER = Path("ch2/chapter-02-linear-regression")
OUT = CHAPTER / "figures"
ANIM = CHAPTER / "animations"
TABLES = CHAPTER / "_tables"
for d in (OUT, ANIM, TABLES):
    d.mkdir(parents=True, exist_ok=True)
GEN = "_scripts/ch2_figures.py"


# ------------------------------------------------------------------ utilities

def br(x, d=3):
    """Number in Brazilian format: 1234.5 -> '1234,500'."""
    return f"{x:.{d}f}".replace(".", ",")


def pval(p):
    return "< 0,0001" if p < 1e-4 else br(p, 4)


def ols(X, y):
    """Least squares with an intercept, plus the usual inference."""
    X = np.atleast_2d(np.asarray(X, float).T).T if np.ndim(X) == 1 else np.asarray(X, float)
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
    for st in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        if vmax / st <= n:
            return [round(v, 4) for v in np.arange(0, vmax + 1e-9, st)]
    return [0, vmax]


def frame_num(ax, xlabel, ylabel, xticks, yticks, xfmt="{:g}", yfmt="{:g}"):
    return ax.frame(xlabel, ylabel, lowhigh=False) + ax.ticks(xticks, yticks, xfmt, yfmt)


# ------------------------------------------------------------- Advertising-like

def advertising(n=200, seed=1):
    rng = np.random.default_rng(seed)
    tv = rng.uniform(0.7, 296, n)
    radio = rng.uniform(0, 49.6, n)
    newspaper = np.clip(1.1 * radio + rng.normal(0, 16, n) + 6, 0.3, 114)
    sales = 6.75 + 0.019 * tv + 0.028 * radio + 0.0011 * tv * radio + rng.normal(0, 0.95, n)
    return tv, radio, newspaper, sales


TV, RADIO, NEWS, SALES = advertising()


def tv_axes(left, top, width=420, height=280):
    return Axes(left, top, width, height, xlim=(0, 300), ylim=(0, 30))


def tv_frame(ax):
    return frame_num(ax, "TV (mil US$)", "sales (mil unidades)", range(0, 301, 50), range(0, 31, 5))


def tv_sales():
    fit = ols(TV, SALES)
    ax = tv_axes(80, 20, 480, 290)
    body = tv_frame(ax)
    for x, y, f in zip(TV, SALES, fit["fitted"]):
        (a, b), (_, c) = ax.px(x, y), ax.px(x, f)
        body += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{a:.1f}" y2="{c:.1f}" stroke="{AXIS}" stroke-width="0.8" stroke-opacity="0.6"/>\n'
    body += points(ax, TV, SALES, RED, 3.6, 0.75)
    w0, w1 = fit["w"]
    body += f'<path d="{ax.path(lambda x: w0 + w1 * x)}" fill="none" stroke="{BLUE}" stroke-width="3.5"/>\n'
    body += text(ax.l + 12, ax.t + 16, f"ŷ = {br(w0, 2)} + {br(w1, 4)} · TV", 15, BLUE, weight=700)
    (OUT / "tv-sales.svg").write_text(svg(600, 370, body))
    return fit


def least_squares_animation(fit):
    xbar, ybar = TV.mean(), SALES.mean()

    def line_for(w1):
        return ybar - w1 * xbar, w1

    def rss_for(w1):
        w0, _ = line_for(w1)
        return float(((SALES - w0 - w1 * TV) ** 2).sum())

    left = tv_axes(80, 30, 380, 290)
    slopes = np.linspace(0.0, 0.1, 101)
    rss = np.array([rss_for(s) for s in slopes])
    ymax = float(np.ceil(rss.max() / 1000) * 1000)
    right = Axes(590, 30, 360, 290, xlim=(0, 0.1), ylim=(0, ymax))
    cands = [(0.015, AMBER, "A"), (0.085, TEAL, "B"), (float(fit["w"][1]), BLUE, "LS")]

    body = step(1, tv_frame(left) + points(left, TV, SALES, POINT, 3.2, 0.75)
                + f'<circle cx="{left.px(xbar, ybar)[0]:.1f}" cy="{left.px(xbar, ybar)[1]:.1f}" r="6" fill="{INK}"/>\n'
                + text(left.px(xbar, ybar)[0] + 10, left.px(xbar, ybar)[1] + 22, "(x̄, ȳ)", 14, INK, weight=700)
                + frame_num(right, "w₁ (inclinação)", "RSS", [0, 0.02, 0.04, 0.06, 0.08, 0.1],
                            np.arange(0, ymax + 1, ymax / 5), "{:.2f}", "{:.0f}"))
    for k, (w1, colour, name) in enumerate(cands):
        w0, _ = line_for(w1)
        res = ""
        for x, y in zip(TV, SALES):
            (a, b), (_, c) = left.px(x, y), left.px(x, w0 + w1 * x)
            res += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{a:.1f}" y2="{c:.1f}" stroke="{colour}" stroke-width="0.9" stroke-opacity="0.55"/>\n'
        line = f'<path class="draw" clip-path="url(#plotL)" d="{left.path(lambda x: w0 + w1 * x)}" fill="none" stroke="{colour}" stroke-width="3.5"/>\n'
        mx, my = right.px(w1, rss_for(w1))
        marker = (f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="7" fill="{colour}"/>\n'
                  + text(mx, my - 14, f"{name}: RSS = {br(rss_for(w1), 0)}", 13, colour, "middle", 700))
        n = (2, 3, 5)[k]  # candidates A and B show for one step; LS stays
        until = n if k < 2 else None
        body += step(n, res, until=until)
        body += step(n, line + marker if k == 2 else line, until=until)
        if k < 2:
            body += step(n, marker)
    body += step(4, f'<path class="draw" d="{right.polyline(slopes, rss)}" fill="none" stroke="{INK}" stroke-width="3"/>\n')
    w0, w1 = fit["w"]
    body += step(6, text(left.l + 10, left.t + 18, f"ŵ₀ = {br(w0, 3)}   ŵ₁ = {br(w1, 4)}", 15, BLUE, weight=700))
    defs = f'<defs><clipPath id="plotL"><rect x="{left.l}" y="{left.t}" width="{left.w}" height="{left.h}"/></clipPath></defs>\n'
    steps = [
        ("Os dados", "200 mercados: orçamento de TV (x) e vendas (y). A reta de <em>least squares</em> sempre passa pelo ponto médio (x̄, ȳ)."),
        ("Uma reta qualquer", "Para cada ponto, o <em>residual</em> eᵢ = yᵢ − ŷᵢ é a distância vertical até a reta. O RSS soma os quadrados: RSS = Σ eᵢ²."),
        ("Outra reta", "Uma inclinação grande demais também deixa resíduos grandes. Cada reta tem o seu RSS."),
        ("RSS em função de w₁", "Variando a inclinação, o RSS desenha uma parábola. Queremos o fundo dela."),
        ("A reta de <em>least squares</em>", "O mínimo do RSS: a reta que melhor se ajusta no sentido dos quadrados mínimos."),
        ("Os coeficientes", "ŵ₁ = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²  e  ŵ₀ = ȳ − ŵ₁x̄."),
    ]
    stepper_page(ANIM / "least-squares.html", "Least squares", "0 0 1000 370", defs + body, steps,
                 "Dados de vendas versus TV, retas candidatas com seus resíduos e o RSS em função da inclinação, com mínimo na reta de least squares",
                 generator=GEN)


def sampling_animation():
    rng = np.random.default_rng(11)
    n, sigma, reps = 100, 2.0, 1000
    left = Axes(80, 30, 380, 290, xlim=(-2, 2), ylim=(-8, 12))
    est = []
    samples = []
    for r in range(reps):
        x = rng.uniform(-2, 2, n)
        y = 2 + 3 * x + rng.normal(0, sigma, n)
        w0, w1 = ols(x, y)["w"]
        est.append(w1)
        if r < 11:
            samples.append((x, y, w0, w1))
    est = np.array(est)
    se_theory = sigma / np.sqrt(n * (16 / 12))  # Var(x) for U(-2, 2) is 16/12

    body = step(1, frame_num(left, "x", "y", range(-2, 3), range(-8, 13, 4))
                + f'<path d="{left.path(lambda x: 2 + 3 * x)}" fill="none" stroke="{RED}" stroke-width="3.5"/>\n'
                + text(left.l + 10, left.t + 18, "f(x) = 2 + 3x  (population regression line)", 14, RED, weight=700))
    x, y, w0, w1 = samples[0]
    body += step(1, points(left, x, y, POINT, 3.2, 0.8), until=2)
    body += step(2, f'<path class="draw" d="{left.path(lambda q: w0 + w1 * q)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
                 + text(left.l + 10, left.t + 40, f"amostra 1: ŵ₀ = {br(w0, 2)}, ŵ₁ = {br(w1, 2)}", 14, BLUE, weight=700))
    more = "".join(f'<path d="{left.path(lambda q, a=a, b=b: a + b * q)}" fill="none" stroke="#8fb0de" stroke-width="1.6"/>\n'
                   for _, _, a, b in samples[1:])
    body += step(3, more + f'<path d="{left.path(lambda x: 2 + 3 * x)}" fill="none" stroke="{RED}" stroke-width="3.5"/>\n')

    lo, hi = 2.4, 3.6
    counts, edges = np.histogram(est, bins=30, range=(lo, hi))
    right = Axes(590, 30, 360, 290, xlim=(lo, hi), ylim=(0, counts.max() * 1.15))
    bars = ""
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        (x0, y0), (x1, _) = right.px(a, c), right.px(b, 0)
        bars += f'<rect x="{x0 + 0.5:.1f}" y="{y0:.1f}" width="{x1 - x0 - 1:.1f}" height="{right.t + right.h - y0:.1f}" fill="{BLUE}" fill-opacity="0.55"/>\n'
    tx, _ = right.px(3, 0)
    body += step(4, frame_num(right, "ŵ₁ em 1000 amostras", "frequência", [2.4, 2.7, 3.0, 3.3, 3.6], [], "{:.1f}")
                 + bars
                 + f'<line x1="{tx:.1f}" y1="{right.t}" x2="{tx:.1f}" y2="{right.t + right.h}" stroke="{RED}" stroke-width="2.5"/>\n'
                 + text(tx + 6, right.t + 14, f"w₁ = 3 · média de ŵ₁ = {br(est.mean(), 3)}", 13, RED, weight=700))
    a, _ = right.px(3 - 2 * se_theory, 0)
    b, _ = right.px(3 + 2 * se_theory, 0)
    yb = right.t + right.h + 52
    body += step(5, f'<path d="M{a:.1f} {yb} H{b:.1f} M{a:.1f} {yb - 6} V{yb + 6} M{b:.1f} {yb - 6} V{yb + 6}" stroke="{PURPLE}" stroke-width="2.5" fill="none"/>\n'
                 + text((a + b) / 2, yb + 22, f"±2 SE(ŵ₁) = ±{br(2 * se_theory, 3)}: ~95% das estimativas", 13, PURPLE, "middle", 700))
    steps = [
        ("A <em>population regression line</em>", "Os dados vêm de y = 2 + 3x + ε. Na prática não conhecemos essa reta; aqui, por ser simulação, conhecemos."),
        ("Uma amostra, uma estimativa", "Com 100 pontos, a reta de <em>least squares</em> fica perto da verdadeira, mas não igual."),
        ("Outras amostras", "Cada nova amostra dá uma reta diferente. As estimativas variam em torno da reta verdadeira."),
        ("<em>Unbiased</em>", "Em 1000 amostras, a média de ŵ₁ coincide com w₁ = 3: o estimador não tem viés (<em>unbiased</em>)."),
        ("<em>Standard error</em>", "O SE mede o quanto ŵ₁ varia de amostra para amostra. ŵ₁ ± 2·SE(ŵ₁) é aproximadamente um <em>confidence interval</em> de 95%."),
    ]
    stepper_page(ANIM / "sampling.html", "Population line e least squares lines", "0 0 1000 420", body, steps,
                 "Reta populacional e retas de least squares de várias amostras, e o histograma das estimativas da inclinação",
                 generator=GEN)
    print(f"sampling: mean w1 = {est.mean():.3f}, sd = {est.std():.3f}, SE theory = {se_theory:.3f}")


def tss_rss():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0, 10, 15))
    y = 1 + 0.8 * x + rng.normal(0, 1.4, 15)
    fit = ols(x, y)
    w0, w1 = fit["w"]
    ybar = y.mean()
    body = ""
    for i, (title, colour, target) in enumerate([
            (f"TSS = Σ(yᵢ − ȳ)² = {br(fit['tss'], 1)}", AMBER, lambda q: np.full_like(q, ybar)),
            (f"RSS = Σ(yᵢ − ŷᵢ)² = {br(fit['rss'], 1)}", BLUE, lambda q: w0 + w1 * q)]):
        ax = Axes(70 + i * 450, 40, 360, 250, xlim=(0, 10), ylim=(-3, 12))
        body += text(ax.l, 24, title, 16, colour, weight=700)
        body += frame_num(ax, "x", "y", range(0, 11, 2), range(0, 13, 4))
        for u, v in zip(x, y):
            (a, b), (_, c) = ax.px(u, v), ax.px(u, float(target(np.array([u]))[0]))
            body += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{a:.1f}" y2="{c:.1f}" stroke="{colour}" stroke-width="2"/>\n'
        body += f'<path d="{ax.path(target)}" fill="none" stroke="{colour}" stroke-width="3"/>\n'
        body += points(ax, x, y, INK, 5)
    body += text(470, 368, f"R² = 1 − RSS/TSS = {br(fit['r2'], 3)}: a fração da variabilidade de y explicada por x", 16, PURPLE, "middle", 700)
    (OUT / "tss-rss.svg").write_text(svg(940, 385, body))


# ------------------------------------------------------- multiple regression

def regression_plane(fit2):
    rng = np.random.default_rng(2)
    idx = rng.choice(len(TV), 70, replace=False)
    w0, w1, w2 = fit2["w"]
    O, U, V, Z = np.array([110, 330]), np.array([360, 45]), np.array([190, -95]), np.array([0, -215])

    def P(tv, radio, s):
        q = O + (tv / 300) * U + (radio / 50) * V + (s / 28) * Z
        return q[0], q[1]

    def plane(tv, radio):
        return w0 + w1 * tv + w2 * radio

    body = ""
    # floor axes
    for a, b, lab in [((0, 0, 0), (300, 0, 0), "TV"), ((0, 0, 0), (0, 50, 0), "radio"), ((0, 0, 0), (0, 0, 28), "sales")]:
        x0, y0 = P(*a)
        x1, y1 = P(*b)
        body += f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{AXIS}" stroke-width="1.5"/>\n'
        dx, dy = {"TV": (22, 14), "radio": (26, -2), "sales": (0, -10)}[lab]
        body += text(x1 + dx, y1 + dy, lab, 15, MUTED, "middle", 700)
    # points behind the plane first, then the plane, then points in front
    order = sorted(idx, key=lambda i: -RADIO[i])
    behind = [i for i in order if SALES[i] < plane(TV[i], RADIO[i])]
    front = [i for i in order if SALES[i] >= plane(TV[i], RADIO[i])]

    def draw(ids, colour):
        out = ""
        for i in ids:
            x0, y0 = P(TV[i], RADIO[i], SALES[i])
            x1, y1 = P(TV[i], RADIO[i], plane(TV[i], RADIO[i]))
            out += f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{colour}" stroke-width="1.2" stroke-opacity="0.8"/>\n'
            out += f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="4.2" fill="{colour}"/>\n'
        return out

    body += draw(behind, "#c98c97")
    corners = [P(0, 0, plane(0, 0)), P(300, 0, plane(300, 0)), P(300, 50, plane(300, 50)), P(0, 50, plane(0, 50))]
    body += f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in corners)}" fill="{BLUE}" fill-opacity="0.18" stroke="{BLUE}" stroke-width="1.5"/>\n'
    for t in np.linspace(0, 300, 7):
        (a, b), (c, d) = P(t, 0, plane(t, 0)), P(t, 50, plane(t, 50))
        body += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" stroke="{BLUE}" stroke-opacity="0.35"/>\n'
    for r in np.linspace(0, 50, 6):
        (a, b), (c, d) = P(0, r, plane(0, r)), P(300, r, plane(300, r))
        body += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" stroke="{BLUE}" stroke-opacity="0.35"/>\n'
    body += draw(front, RED)
    body += text(30, 26, f"ŷ = {br(w0, 2)} + {br(w1, 4)}·TV + {br(w2, 4)}·radio", 16, BLUE, weight=700)
    body += text(30, 48, "pontos acima do plano em vermelho, abaixo em rosa", 13, MUTED)
    (OUT / "regression-plane.svg").write_text(svg(720, 400, body))


def interaction_figure(fit_main, fit_int):
    body = ""
    lo, hi = RADIO < 15, RADIO > 35
    for i, (title, line) in enumerate([
            ("Sem interaction: retas paralelas", lambda tv, r: fit_main["w"][0] + fit_main["w"][1] * tv + fit_main["w"][2] * r),
            ("Com interaction (TV × radio)", lambda tv, r: fit_int["w"][0] + fit_int["w"][1] * tv + fit_int["w"][2] * r + fit_int["w"][3] * tv * r)]):
        ax = tv_axes(80 + i * 470, 40, 370, 260)
        body += text(ax.l, 24, title, 16, INK, weight=700)
        body += tv_frame(ax)
        body += points(ax, TV[lo], SALES[lo], BLUE, 3.6, 0.7) + points(ax, TV[hi], SALES[hi], AMBER, 3.6, 0.7)
        body += f'<path d="{ax.path(lambda tv: line(tv, 7.5))}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
        body += f'<path d="{ax.path(lambda tv: line(tv, 42.5))}" fill="none" stroke="{AMBER}" stroke-width="3"/>\n'
    body += (f'<circle cx="86" cy="368" r="5" fill="{BLUE}"/>' + text(98, 373, "radio baixo (&lt; 15)", 14)
             + f'<circle cx="276" cy="368" r="5" fill="{AMBER}"/>' + text(288, 373, "radio alto (&gt; 35)", 14))
    (OUT / "interaction.svg").write_text(svg(940, 385, body))


def advertising_tables():
    simple = {name: ols(x, SALES) for name, x in [("TV", TV), ("radio", RADIO), ("newspaper", NEWS)]}
    write_table("simple-tv", coef_table(simple["TV"], ["Intercept ($w_0$)", "TV ($w_1$)"]))
    f = simple["TV"]
    write_table("simple-tv-fit",
                "| *Quantity* | Valor |\n|---|---:|\n"
                f"| *Residual standard error* (RSE) | {br(f['rse'], 2)} |\n"
                f"| $R^2$ | {br(f['r2'], 3)} |\n"
                f"| *F-statistic* | {br(f['F'], 1)} |\n")
    rows = ["| Regressão simples | $\\hat w_1$ | *Std. error* | *t-statistic* | *p-value* |", "|---|---:|---:|---:|---:|"]
    for name, fit in simple.items():
        rows.append(f"| sales ~ {name} | {br(fit['w'][1], 4)} | {br(fit['se'][1], 4)} | {br(fit['t'][1], 2)} | {pval(fit['p'][1])} |")
    write_table("simple-each", "\n".join(rows) + "\n")

    multi = ols(np.column_stack([TV, RADIO, NEWS]), SALES)
    write_table("multiple", coef_table(multi, ["Intercept", "TV", "radio", "newspaper"]))
    write_table("multiple-fit",
                "| *Quantity* | Valor |\n|---|---:|\n"
                f"| RSE | {br(multi['rse'], 2)} |\n"
                f"| $R^2$ | {br(multi['r2'], 3)} |\n"
                f"| *F-statistic* | {br(multi['F'], 1)} |\n")
    C = np.corrcoef(np.vstack([TV, RADIO, NEWS, SALES]))
    names = ["TV", "radio", "newspaper", "sales"]
    rows = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * 4]
    for i, n in enumerate(names):
        rows.append(f"| **{n}** | " + " | ".join(br(C[i, j], 2) for j in range(4)) + " |")
    write_table("correlation", "\n".join(rows) + "\n")
    write_table("corr-radio-news",
                f"A correlação entre radio e newspaper é {br(C[1, 2], 2)}: mercados que investem em radio "
                "também tendem a investir em newspaper.\n")

    main = ols(np.column_stack([TV, RADIO]), SALES)
    inter = ols(np.column_stack([TV, RADIO, TV * RADIO]), SALES)
    write_table("interaction", coef_table(inter, ["Intercept", "TV", "radio", "TV × radio"]))

    # Confidence vs prediction interval at TV = 100, radio = 20 (main-effects model)
    x0 = np.array([1, 100, 20])
    yhat = float(x0 @ main["w"])
    se_mean = float(np.sqrt(x0 @ main["xtx_inv"] @ x0 * main["sigma2"]))
    se_pred = float(np.sqrt(main["sigma2"] + se_mean ** 2))
    q = stats.t.ppf(0.975, main["n"] - main["k"])
    write_table("intervals",
                "| TV = 100, radio = 20 | Intervalo de 95% |\n|---|---|\n"
                f"| *Confidence interval* (vendas **médias** nesses mercados) | [{br(yhat - q * se_mean, 2)}; {br(yhat + q * se_mean, 2)}] |\n"
                f"| *Prediction interval* (vendas de **um** mercado) | [{br(yhat - q * se_pred, 2)}; {br(yhat + q * se_pred, 2)}] |\n")

    ci = lambda fit, j: f"[{br(fit['w'][j] - 1.96 * fit['se'][j], 3)}; {br(fit['w'][j] + 1.96 * fit['se'][j], 3)}]"
    write_table("marketing",
                "| Pergunta | Resposta com os dados |\n|---|---|\n"
                f"| 1. Há relação entre publicidade e vendas? | Sim: *F-statistic* = {br(multi['F'], 1)}, *p-value* {pval(multi['pF'])}. |\n"
                f"| 2. Quão forte é a relação? | RSE = {br(multi['rse'], 2)} (erro típico de ~{br(100 * multi['rse'] / SALES.mean(), 0)}% da média de vendas); $R^2$ = {br(multi['r2'], 3)}. |\n"
                f"| 3. Quais mídias contribuem? | TV e radio (*p-value* {pval(multi['p'][1])}); newspaper não (*p-value* = {pval(multi['p'][3])}). |\n"
                f"| 4. Qual o tamanho do efeito? | *Confidence intervals* de 95%: TV {ci(multi, 1)}, radio {ci(multi, 2)}, newspaper {ci(multi, 3)} (contém 0). |\n"
                f"| 5. Com que precisão prevemos? | Média de vendas: *confidence interval*; um mercado: *prediction interval*, mais largo (Seção 2.2.2). |\n"
                "| 6. A relação é linear? | Os *residual plots* do modelo sem interação mostram padrão: não totalmente. |\n"
                f"| 7. Há sinergia entre as mídias? | Sim: o termo TV × radio tem *p-value* {pval(inter['p'][3])} e o $R^2$ sobe de {br(main['r2'], 3)} para {br(inter['r2'], 3)}. |\n")
    print(f"advertising: TV slope {simple['TV']['w'][1]:.4f}, multi R2 {multi['r2']:.3f}, news p {multi['p'][3]:.3f}, "
          f"news simple p {simple['newspaper']['p'][1]:.2g}, interaction R2 {inter['r2']:.3f}, corr(radio,news) {C[1, 2]:.2f}")
    return main, inter


# --------------------------------------------------------- Auto-like (mpg)

def auto_data(n=200, seed=8):
    rng = np.random.default_rng(seed)
    hp = rng.uniform(46, 230, n)
    mpg = np.clip(56.9 - 0.466 * hp + 0.00123 * hp ** 2 + rng.normal(0, 4.3, n), 9, 47)
    return hp, mpg


def poly_fit(x, y, deg):
    c = np.polyfit(x, y, deg)
    return lambda q: np.polyval(c, q)


def polynomial_figure():
    hp, mpg = auto_data()
    ax = Axes(80, 20, 480, 290, xlim=(40, 235), ylim=(5, 50))
    body = frame_num(ax, "horsepower", "mpg", range(50, 231, 30), range(10, 51, 10))
    body += points(ax, hp, mpg, POINT, 3.6, 0.75)
    for k, (deg, colour, name) in enumerate([(1, AMBER, "Linear"), (2, BLUE, "Degree 2"), (5, TEAL, "Degree 5")]):
        dash = ' stroke-dasharray="7 5"' if deg == 5 else ""
        body += f'<path d="{ax.path(poly_fit(hp, mpg, deg))}" fill="none" stroke="{colour}" stroke-width="3"{dash}/>\n'
        body += text(ax.l + ax.w - 6, ax.t + 18 + 20 * k, name, 14, colour, "end", 700)
    (OUT / "polynomial.svg").write_text(svg(600, 370, body))


def smooth(x, y, h):
    order = np.argsort(x)
    xs = np.linspace(x.min(), x.max(), 120)
    w = np.exp(-0.5 * ((xs[:, None] - x[None, :]) / h) ** 2)
    return xs, (w @ y) / w.sum(1)


def residual_panel(ax, fitted, res, title, xl):
    out = text(ax.l, ax.t - 14, title, 15, INK, weight=700)
    out += frame_num(ax, xl, "Residuals", [], [])
    _, z = ax.px(ax.xlim[0], 0)
    out += f'<line x1="{ax.l}" y1="{z:.1f}" x2="{ax.l + ax.w}" y2="{z:.1f}" stroke="{AXIS}" stroke-dasharray="5 5"/>\n'
    out += points(ax, fitted, res, POINT, 3.2, 0.75)
    xs, ys = smooth(fitted, res, (fitted.max() - fitted.min()) / 10)
    out += f'<path d="{ax.polyline(xs, ys)}" fill="none" stroke="{RED}" stroke-width="3"/>\n'
    return out


def residual_plots():
    hp, mpg = auto_data()
    body = ""
    for i, deg in enumerate((1, 2)):
        f = poly_fit(hp, mpg, deg)
        fv, r = f(hp), mpg - f(hp)
        m = max(abs(r).max(), 1) * 1.1
        ax = Axes(60 + i * 310, 40, 250, 220, xlim=(fv.min() - 1, fv.max() + 1), ylim=(-m, m))
        body += residual_panel(ax, fv, r, ("Linear: padrão em U" if deg == 1 else "Degree 2: sem padrão"), "Fitted values")
    rng = np.random.default_rng(4)
    x = rng.uniform(1, 10, 150)
    y = 2 + 0.5 * x + rng.normal(0, 0.03 * x ** 1.8)
    fit = ols(x, y)
    m = abs(fit["res"]).max() * 1.1
    ax = Axes(680, 40, 250, 220, xlim=(fit["fitted"].min() - 0.2, fit["fitted"].max() + 0.2), ylim=(-m, m))
    body += residual_panel(ax, fit["fitted"], fit["res"], "Heteroscedasticity: funil", "Fitted values")
    (OUT / "residual-plots.svg").write_text(svg(960, 320, body))


def outlier_leverage():
    rng = np.random.default_rng(6)
    x = rng.uniform(-2, 2, 30)
    y = 1 + 2 * x + rng.normal(0, 1, 30)
    body = ""
    cases = [("Outlier: yᵢ incomum", (0.3, 9.5), RED), ("High leverage: xᵢ incomum", (4.5, 1.0), PURPLE)]
    for i, (title, (px_, py_), colour) in enumerate(cases):
        ax = Axes(70 + i * 450, 40, 360, 250, xlim=(-2.5, 5), ylim=(-5, 11))
        xa, ya = np.append(x, px_), np.append(y, py_)
        with_pt, without = ols(xa, ya), ols(x, y)
        body += text(ax.l, 24, title, 16, colour, weight=700)
        body += frame_num(ax, "x", "y", range(-2, 6, 2), range(-4, 12, 4))
        body += points(ax, x, y, POINT, 4)
        a, b = ax.px(px_, py_)
        body += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="7" fill="{colour}"/>\n'
        body += f'<path d="{ax.path(lambda q: without["w"][0] + without["w"][1] * q)}" fill="none" stroke="{INK}" stroke-width="2.5" stroke-dasharray="7 6"/>\n'
        body += f'<path d="{ax.path(lambda q: with_pt["w"][0] + with_pt["w"][1] * q)}" fill="none" stroke="{colour}" stroke-width="3"/>\n'
        h = 1 / len(xa) + (px_ - xa.mean()) ** 2 / ((xa - xa.mean()) ** 2).sum()
        body += text(ax.l + ax.w, ax.t + ax.h - 10, f"leverage hᵢ = {br(h, 2)}", 14, colour, "end", 700)
    body += (f'<line x1="70" y1="358" x2="104" y2="358" stroke="{INK}" stroke-width="2.5" stroke-dasharray="7 6"/>' + text(112, 363, "ajuste sem o ponto", 14)
             + f'<line x1="290" y1="358" x2="324" y2="358" stroke="{MUTED}" stroke-width="3"/>' + text(332, 363, "ajuste com o ponto", 14))
    (OUT / "outlier-leverage.svg").write_text(svg(900, 380, body))


def collinearity():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(9)
    body = ""
    for i, (rho, title) in enumerate([(0.1, "Preditores pouco correlacionados"), (0.98, "Collinearity: correlação 0,98")]):
        z1, z2 = rng.normal(size=(2, 200))
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        y = 1 * x1 + 1 * x2 + rng.normal(0, 1, 200)
        fit = ols(np.column_stack([x1, x2]), y)
        a, b = np.meshgrid(np.linspace(-1, 3, 200), np.linspace(-1, 3, 200))
        R = ((y[None, None, :] - fit["w"][0] - a[..., None] * x1 - b[..., None] * x2) ** 2).sum(-1)
        levels = fit["rss"] * np.array([1.02, 1.08, 1.25, 1.6, 2.4])
        cs = plt.contour(a, b, R, levels=levels)
        ax = Axes(70 + i * 450, 40, 330, 280, xlim=(-1, 3), ylim=(-1, 3))
        body += text(ax.l, 24, title, 16, INK, weight=700)
        body += frame_num(ax, "w₁", "w₂", range(-1, 4), range(-1, 4))
        for segs in cs.allsegs:
            for seg in segs:
                body += f'<path d="{ax.polyline(seg[:, 0], seg[:, 1])}" fill="none" stroke="{BLUE}" stroke-width="1.8"/>\n'
        mx, my = ax.px(*fit["w"][1:])
        body += f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="6" fill="{RED}"/>\n'
        r2 = ols(x2, x1)["r2"]
        body += text(ax.l + ax.w, ax.t + 16, f"VIF = {br(1 / (1 - r2), 1)}", 15, RED, "end", 700)
        plt.close("all")
    body += text(470, 392, "Curvas de nível do RSS: com collinearity, muitas combinações (w₁, w₂) ajustam quase igualmente bem", 15, MUTED, "middle", 600)
    (OUT / "collinearity.svg").write_text(svg(940, 405, body))


# ------------------------------------------------------ KNN versus regression

def knn_reg(xq, x, y, k):
    d = np.abs(np.asarray(xq)[:, None] - x[None, :]) if x.ndim == 1 else ((xq[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    return y[idx].mean(1)


F_LIN = lambda x: 1 + 2 * x
F_NONLIN = lambda x: 2 + 2.2 * np.sin(2.4 * x) + 0.4 * x


def knn_regression_figure():
    rng = np.random.default_rng(12)
    x = np.sort(rng.uniform(-1, 1, 50))
    y = F_NONLIN(x) + rng.normal(0, 0.5, 50)
    q = np.linspace(-1, 1, 400)
    body = ""
    for i, k in enumerate((1, 9)):
        ax = Axes(70 + i * 450, 40, 360, 250, xlim=(-1, 1), ylim=(-1, 5.5))
        body += text(ax.l, 24, f"KNN regression, K = {k}", 16, INK, weight=700)
        body += frame_num(ax, "x", "y", [-1, -0.5, 0, 0.5, 1], range(0, 6))
        body += f'<path d="{ax.path(F_NONLIN)}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>\n'
        body += points(ax, x, y, POINT, 4)
        body += f'<path d="{ax.polyline(q, knn_reg(q, x, y, k))}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
    body += (f'<line x1="70" y1="358" x2="104" y2="358" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>' + text(112, 363, "f verdadeira", 14)
             + f'<line x1="250" y1="358" x2="284" y2="358" stroke="{BLUE}" stroke-width="3"/>' + text(292, 363, "f̂ do KNN: média dos K vizinhos", 14))
    (OUT / "knn-regression.svg").write_text(svg(900, 380, body))


def test_mse(predict, f, sigma, rng, p=1, n_test=2000):
    Xt = rng.uniform(-1, 1, (n_test, p))
    return float(np.mean((f(Xt[:, 0]) - predict(Xt)) ** 2) + sigma ** 2)


def knn_vs_lr():
    sigma, n, reps = 0.5, 50, 30
    ks = [1, 2, 3, 4, 5, 7, 9, 12, 16, 20, 25, 33, 50]
    body = ""
    for i, (f, title) in enumerate([(F_LIN, "f linear"), (F_NONLIN, "f não linear")]):
        rng = np.random.default_rng(20 + i)
        knn_m, lr_m = np.zeros(len(ks)), 0.0
        for _ in range(reps):
            x = rng.uniform(-1, 1, n)
            y = f(x) + rng.normal(0, sigma, n)
            for j, k in enumerate(ks):
                knn_m[j] += test_mse(lambda X: knn_reg(X[:, 0], x, y, k), f, sigma, rng) / reps
            fit = ols(x, y)
            lr_m += test_mse(lambda X: fit["w"][0] + fit["w"][1] * X[:, 0], f, sigma, rng) / reps
        ymax = max(knn_m.max(), lr_m) * 1.15
        ax = Axes(70 + i * 450, 40, 360, 250, xlim=(0.02, 1), ylim=(0, ymax), xlog=True)
        body += text(ax.l, 24, title, 16, INK, weight=700)
        body += frame_num(ax, "1/K  (flexibilidade →)", "Test MSE", [0.02, 0.05, 0.1, 0.2, 0.5, 1],
                          nice_ticks(ymax), "{:g}", "{:g}")
        _, ly = ax.px(0.02, lr_m)
        body += f'<line x1="{ax.l}" y1="{ly:.1f}" x2="{ax.l + ax.w}" y2="{ly:.1f}" stroke="{AMBER}" stroke-width="3" stroke-dasharray="8 6"/>\n'
        body += f'<path d="{ax.polyline([1 / k for k in ks], knn_m)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
        print(f"knn vs lr ({title}): LR {lr_m:.3f}, best KNN {knn_m.min():.3f}")
    body += (f'<line x1="70" y1="358" x2="104" y2="358" stroke="{AMBER}" stroke-width="3" stroke-dasharray="8 6"/>' + text(112, 363, "linear regression", 14)
             + f'<line x1="290" y1="358" x2="324" y2="358" stroke="{BLUE}" stroke-width="3"/>' + text(332, 363, "KNN", 14))
    (OUT / "knn-vs-lr.svg").write_text(svg(900, 380, body))


def curse_of_dimensionality():
    sigma, n, reps = 0.5, 50, 30
    ps = [1, 2, 3, 4, 10, 20]
    ks = [1, 2, 3, 5, 7, 10, 15, 20]
    rng = np.random.default_rng(31)
    lr_m, knn_m = [], []
    for p in ps:
        lr_acc, knn_acc = 0.0, np.zeros(len(ks))
        for _ in range(reps):
            X = rng.uniform(-1, 1, (n, p))
            y = F_NONLIN(X[:, 0]) + rng.normal(0, sigma, n)
            fit = ols(X, y)
            lr_acc += test_mse(lambda T: fit["w"][0] + T @ fit["w"][1:], F_NONLIN, sigma, rng, p, 1000) / reps
            for j, k in enumerate(ks):
                knn_acc[j] += test_mse(lambda T: knn_reg(T, X, y, k), F_NONLIN, sigma, rng, p, 1000) / reps
        lr_m.append(lr_acc)
        knn_m.append(knn_acc.min())
    ymax = max(max(lr_m), max(knn_m)) * 1.15
    ax = Axes(80, 30, 480, 270, xlim=(0, len(ps) - 1), ylim=(0, ymax))
    pos = list(range(len(ps)))
    body = frame_num(ax, "p (número de features; só x₁ importa)", "Test MSE", [], nice_ticks(ymax), yfmt="{:g}")
    for j, p in enumerate(ps):
        x, _ = ax.px(j, 0)
        body += text(x, ax.t + ax.h + 19, str(p), 12, MUTED, "middle")
    body += f'<path d="{ax.polyline(pos, lr_m)}" fill="none" stroke="{AMBER}" stroke-width="3" stroke-dasharray="8 6"/>\n'
    body += f'<path d="{ax.polyline(pos, knn_m)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
    body += points(ax, pos, lr_m, AMBER, 5, 1) + points(ax, pos, knn_m, BLUE, 5, 1)
    body += (text(ax.l + ax.w, ax.px(len(ps) - 1, knn_m[-1])[1] - 12, "KNN (melhor K)", 14, BLUE, "end", 700)
             + text(ax.px(3.5, 0)[0], ax.px(3.5, lr_m[3])[1] + 26, "linear regression", 14, AMBER, "middle", 700))
    (OUT / "curse-of-dimensionality.svg").write_text(svg(600, 360, body))
    print("curse: LR", [round(v, 2) for v in lr_m], "KNN", [round(v, 2) for v in knn_m])


if __name__ == "__main__":
    tv_fit = tv_sales()
    least_squares_animation(tv_fit)
    sampling_animation()
    tss_rss()
    main_fit, int_fit = advertising_tables()
    regression_plane(main_fit)
    interaction_figure(main_fit, int_fit)
    polynomial_figure()
    residual_plots()
    outlier_leverage()
    collinearity()
    knn_regression_figure()
    knn_vs_lr()
    curse_of_dimensionality()
