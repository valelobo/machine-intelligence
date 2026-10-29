"""Generate the figures and the computed animation for Chapter 1.

Run from the repo root:  python3 _scripts/ch1_figures.py

Writes SVG figures into ch1/chapter-01-statistical-learning/figures/ and the
overfitting animation into .../animations/overfitting.html (do not edit that
file by hand; change it here). Also prints the curve paths used by the
hand-written animations/bias-variance.html.

Two simulated data sets are used throughout, in the spirit of ISLP ch. 2:
- income vs years of education (regression), and
- two classes on the unit square with a known Bayes boundary (classification).
"""
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import make_smoothing_spline

CHAPTER = Path("ch1/chapter-01-statistical-learning")
OUT = CHAPTER / "figures"
OUT.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import *  # noqa: E402,F403  palette, svg(), text(), Axes



# ------------------------------------------------- income vs years of education

SIGMA = 6.0
XLIM_EDU = (10, 22)


def income_data(n=30, seed=4):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(*XLIM_EDU, n))
    return x, income_f(x) + rng.normal(0, SIGMA, n)


def spline_fit(x, y, lam):
    """Smoothing spline (non-parametric); smaller lam = more flexible."""
    return make_smoothing_spline(x, y, lam=lam)


def spline_df(x, lam):
    """Effective degrees of freedom: trace of the linear smoother matrix."""
    eye = np.eye(len(x))
    return float(sum(make_smoothing_spline(x, eye[i], lam=lam)(x[i]) for i in range(len(x))))


def linear_fit(x, y):
    w1, w0 = np.polyfit(x, y, 1)
    return lambda q: w0 + w1 * np.asarray(q)


GRID_EDU = np.linspace(*XLIM_EDU, 400)


def test_mse_of(predict):
    """Expected test MSE over x0 uniform on the range: bias^2 + Var(eps)."""
    return float(np.mean((income_f(GRID_EDU) - predict(GRID_EDU)) ** 2) + SIGMA ** 2)


def edu_axes(left, top, width=380, height=250):
    return Axes(left, top, width, height, xlim=XLIM_EDU, ylim=(10, 90))


def edu_frame(ax):
    return (ax.frame("Anos de estudo (x)", "Renda (y)", lowhigh=False)
            + ax.ticks(xticks=(10, 12, 14, 16, 18, 20, 22), yticks=(20, 40, 60, 80)))


def points(ax, x, y, fill=POINT, r=4.5):
    return "".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{r}" fill="{fill}" fill-opacity="0.9"/>\n'
                   for a, b in (ax.px(u, v) for u, v in zip(x, y)))


def income_education():
    x, y = income_data()
    ax = edu_axes(80, 30, 480, 290)
    body = edu_frame(ax)
    for u, v in zip(x, y):
        (a, b), (_, c) = ax.px(u, v), ax.px(u, income_f(u))
        body += f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{a:.1f}" y2="{c:.1f}" stroke="{RED}" stroke-width="1.5"/>\n'
    body += f'<path d="{ax.path(income_f)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
    body += points(ax, x, y, INK)
    body += (text(ax.l + 12, ax.t + 14, "— f(x)", 14, BLUE, weight=700)
             + text(ax.l + 12, ax.t + 34, "| ε = y − f(x)", 14, RED, weight=700))
    (OUT / "income-education.svg").write_text(svg(600, 380, body))


def parametric_vs_nonparametric():
    x, y = income_data()
    lin = linear_fit(x, y)
    body = ""
    panels = [("Parametric: f(x) = w₀ + w₁x", lin, AMBER),
              ("Non-parametric: sem forma fixa", spline_fit(x, y, 0.5), BLUE)]
    for i, (title, fit, colour) in enumerate(panels):
        ax = edu_axes(80 + i * 470, 40)
        body += text(ax.l, 22, title, 16, INK, weight=700)
        body += edu_frame(ax)
        body += f'<path d="{ax.path(income_f)}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>\n'
        body += points(ax, x, y)
        body += f'<path d="{ax.path(fit)}" fill="none" stroke="{colour}" stroke-width="3.5"/>\n'
    body += (f'<line x1="80" y1="352" x2="110" y2="352" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>\n'
             + text(118, 357, "f verdadeira (desconhecida na prática)", 14, MUTED))
    (OUT / "parametric-vs-nonparametric.svg").write_text(svg(940, 370, body))


def overfitting_animation():
    """Fits of growing flexibility (left) and training/test MSE (right)."""
    x, y = income_data()
    lin = linear_fit(x, y)
    train_lin = float(np.mean((y - lin(x)) ** 2))
    rows = []
    for lam in np.geomspace(1e-7, 1e3, 90):
        df = spline_df(x, lam)
        if df < 2.05 or df > 24:
            continue
        fit_l = spline_fit(x, y, lam)
        rows.append((df, float(np.mean((y - fit_l(x)) ** 2)), test_mse_of(fit_l), lam))
    rows.sort()
    dfs = [2.0] + [r[0] for r in rows]
    trains = [train_lin] + [r[1] for r in rows]
    tests = [test_mse_of(lin)] + [r[2] for r in rows]
    best = min(rows, key=lambda r: r[2])
    wiggly = min(rows, key=lambda r: abs(r[0] - 22))

    fits = [("linear", lin, AMBER, 2.0, train_lin, tests[0]),
            ("flexível", spline_fit(x, y, best[3]), BLUE, best[0], best[1], best[2]),
            ("muito flexível", spline_fit(x, y, wiggly[3]), TEAL, wiggly[0], wiggly[1], wiggly[2])]

    left = edu_axes(80, 30, 380, 300)
    ymax = math.ceil(max(tests) * 1.1 / 20) * 20
    ystep = next(st for st in (10, 20, 25, 50, 100) if ymax / st <= 7)
    right = Axes(590, 30, 360, 300, xlim=(0, math.ceil(max(dfs) / 5) * 5), ylim=(0, ymax))
    fine = np.linspace(*XLIM_EDU, 500)

    def g(step, inner, until=None):
        u = f' data-until="{until}"' if until else ""
        return f'<g data-step="{step}"{u}>\n{inner}</g>\n'

    s1 = (edu_frame(left)
          + f'<path d="{left.path(income_f)}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>\n'
          + points(left, x, y)
          + right.frame("Flexibilidade (graus de liberdade)", "MSE", lowhigh=False)
          + right.ticks(xticks=range(0, int(right.xlim[1]) + 1, 5), yticks=range(0, ymax + 1, ystep)))
    body = g(1, s1)
    for k, (name, fit, colour, df, tr, te) in enumerate(fits):
        (cx, ty), (_, ry) = right.px(df, te), right.px(df, tr)
        inner = (f'<path class="fit draw" clip-path="url(#plot)" d="{left.polyline(fine, fit(fine))}" stroke="{colour}"/>\n'
                 + f'<rect x="{cx - 6:.1f}" y="{ty - 6:.1f}" width="12" height="12" fill="{colour}" stroke="{RED}" stroke-width="2"/>\n'
                 + f'<rect x="{cx - 6:.1f}" y="{ry - 6:.1f}" width="12" height="12" fill="{colour}" stroke="{POINT}" stroke-width="2"/>\n'
                 + text(left.l + left.w - 6, left.t + left.h - 52 + 20 * k, name, 14, colour, "end", 700))
        body += g(2 + k, inner)
    _, iy = right.px(0, SIGMA ** 2)
    curves = (f'<line x1="{right.l}" y1="{iy:.1f}" x2="{right.l + right.w}" y2="{iy:.1f}" stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="6 5"/>\n'
              + text(right.l + right.w - 2, iy - 7, "Var(ε)", 13, MUTED, "end")
              + f'<path class="curve draw" d="{right.polyline(dfs, tests)}" stroke="{RED}"/>\n'
              + f'<path class="curve draw" d="{right.polyline(dfs, trains)}" stroke="{POINT}"/>\n'
              + text(right.l + right.w - 2, right.px(dfs[-1], tests[-1])[1] - 10, "Test MSE", 14, RED, "end", 700)
              + text(right.px(12, 0)[0], right.px(12, float(np.interp(12, dfs, trains)))[1] - 12, "Training MSE", 14, MUTED, "middle", 700))
    body += g(5, curves)
    bx, by = right.px(best[0], best[2])
    notes = (f'<line x1="{bx:.1f}" y1="{by + 10:.1f}" x2="{bx:.1f}" y2="{right.t + right.h:.1f}" stroke="{RED}" stroke-dasharray="4 4"/>\n'
             + text(right.l + 8, right.t + 14, "← underfitting", 14, AMBER, weight=700)
             + text(right.l + right.w - 2, right.t + 14, "overfitting →", 14, TEAL, "end", 700))
    body += g(6, notes)

    defs = (f'<defs><clipPath id="plot"><rect x="{left.l}" y="{left.t}" width="{left.w}" height="{left.h}"/></clipPath></defs>\n')
    steps = [
        ("Os dados", "30 pessoas: anos de estudo (x) e renda (y). A curva tracejada é a f verdadeira, que na prática não conhecemos."),
        ("Um modelo rígido", "A reta (parametric) é simples demais para a forma de f: erra no treino e no teste. Isso é <em>underfitting</em>."),
        ("Flexibilidade na medida", "Um <em>smoothing spline</em> segue a forma de f sem perseguir o ruído. O <em>test MSE</em> é o menor dos três."),
        ("Flexível demais", "Uma curva muito flexível persegue o ruído: o <em>training MSE</em> cai ainda mais, mas o <em>test MSE</em> volta a subir. Isso é <em>overfitting</em>."),
        ("A curva em U", "Para todas as flexibilidades: o <em>training MSE</em> só cai; o <em>test MSE</em> cai e depois sobe, nunca abaixo de Var(ε)."),
        ("A lição", "Um <em>training MSE</em> pequeno não garante um <em>test MSE</em> pequeno. Queremos o fundo do U."),
    ]
    captions = "\n".join(f"      <li><h3>{t}</h3><p>{p}</p></li>" for t, p in steps)
    html = f"""<!doctype html>
<!-- GENERATED by _scripts/ch1_figures.py (overfitting_animation). Edit the script, not this file. -->
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Overfitting na prática</title>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
  <link rel="stylesheet" href="../../../assets/stepper/stepper.css">
  <style>
    .fit {{ fill: none; stroke-width: 3.5; stroke-linejoin: round; }}
    .curve {{ fill: none; stroke-width: 3.5; stroke-linejoin: round; }}
  </style>
</head>
<body>
  <div class="stepper" data-interval="3200">
    <div class="st-stage">
      <svg viewBox="0 0 1000 380" font-family="{FONT}" role="img" aria-label="Ajustes com flexibilidade crescente aos dados de renda e as curvas de training MSE e test MSE">
{defs}{body}      </svg>
    </div>
    <ol class="st-steps">
{captions}
    </ol>
  </div>
  <script src="../../../assets/stepper/stepper.js"></script>
</body>
</html>
"""
    (CHAPTER / "animations" / "overfitting.html").write_text(html)
    print(f"overfitting: best df={best[0]:.1f} lam={best[3]:.2g} test={best[2]:.1f}; "
          f"wiggly df={wiggly[0]:.1f} train={wiggly[1]:.1f} test={wiggly[2]:.1f}; linear test={tests[0]:.1f}")


# ------------------------------------------------------ two-class classification

def boundary(x1):
    return 0.5 + 0.22 * np.sin(2 * np.pi * (np.asarray(x1) - 0.1))


def p_amber(x1, x2):
    return 1 / (1 + np.exp(-(np.asarray(x2) - boundary(x1)) / 0.07))


def class_data(n=150, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    lab = (rng.random(n) < p_amber(X[:, 0], X[:, 1])).astype(int)  # 1 = amber
    return X, lab


def knn_predict(Q, X, lab, k):
    d = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k - 1, axis=1)[:, :k] if k > 1 else d.argmin(1)[:, None]
    return (lab[idx].mean(1) > 0.5).astype(int)


def class_panel(ax, X, lab, shade=None, res=None):
    l, t, w, h = ax.l, ax.t, ax.w, ax.h
    out = ""
    if shade is None:
        top = ax.path(lambda u: float(boundary(u)))
        out += f'<path d="{top} L{l + w} {t} L{l} {t} Z" fill="{AMBER}" fill-opacity="0.10"/>\n'
        out += f'<path d="{top} L{l + w} {t + h} L{l} {t + h} Z" fill="{BLUE}" fill-opacity="0.10"/>\n'
    else:
        nx, ny = res
        cw, ch = w / nx, h / ny
        for j in range(ny):
            row = shade[j]
            i = 0
            while i < nx:
                k = i
                while k < nx and row[k] == row[i]:
                    k += 1
                c = AMBER if row[i] else BLUE
                out += (f'<rect x="{l + i * cw:.1f}" y="{t + h - (j + 1) * ch:.1f}" width="{(k - i) * cw + 0.3:.1f}" '
                        f'height="{ch + 0.3:.1f}" fill="{c}" fill-opacity="0.16"/>\n')
                i = k
    for (u, v), c in zip(X, lab):
        a, b = ax.px(u, v)
        out += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{AMBER if c else BLUE}" fill-opacity="0.9"/>\n'
    out += f'<path d="{ax.path(lambda u: float(boundary(u)))}" fill="none" stroke="{PURPLE}" stroke-width="2.5" stroke-dasharray="8 6"/>\n'
    out += f'<rect x="{l}" y="{t}" width="{w}" height="{h}" fill="none" stroke="{GRID}" stroke-width="1.5"/>\n'
    return out


def class_legend(l, y):
    return (f'<line x1="{l}" y1="{y}" x2="{l + 34}" y2="{y}" stroke="{PURPLE}" stroke-width="3" stroke-dasharray="8 6"/>\n'
            + text(l + 42, y + 5, "Bayes decision boundary", 15, PURPLE, weight=600)
            + f'<circle cx="{l + 17}" cy="{y + 26}" r="5" fill="{AMBER}"/>' + text(l + 42, y + 31, "classe A", 15)
            + f'<circle cx="{l + 137}" cy="{y + 26}" r="5" fill="{BLUE}"/>' + text(l + 150, y + 31, "classe B", 15))


def bayes_classifier():
    X, lab = class_data()
    ax = Axes(20, 20, 380, 320)
    body = class_panel(ax, X, lab) + class_legend(ax.l, ax.t + ax.h + 26)
    (OUT / "bayes-classifier.svg").write_text(svg(420, 410, body))


def knn_neighbors():
    amber = [(0.2, 0.75), (0.35, 0.82), (0.15, 0.5), (0.3, 0.28), (0.55, 0.88), (0.42, 0.62)]
    blue = [(0.6, 0.45), (0.7, 0.62), (0.5, 0.35), (0.82, 0.3), (0.78, 0.78), (0.62, 0.18)]
    x0 = np.array([0.52, 0.52])
    pts = [(np.array(p), 1) for p in amber] + [(np.array(p), 0) for p in blue]
    near = sorted(pts, key=lambda pc: np.linalg.norm(pc[0] - x0))[:3]
    r = np.linalg.norm(near[-1][0] - x0) + 0.035
    ax = Axes(20, 20, 360, 330)
    cx, cy = ax.px(*x0)
    body = (f'<rect x="{ax.l}" y="{ax.t}" width="{ax.w}" height="{ax.h}" fill="none" stroke="{GRID}" stroke-width="1.5"/>\n'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * ax.w:.1f}" fill="{PURPLE}" fill-opacity="0.07" '
            f'stroke="{PURPLE}" stroke-width="2" stroke-dasharray="6 5"/>\n')
    for p, c in near:
        a, b = ax.px(*p)
        body += f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{a:.1f}" y2="{b:.1f}" stroke="{PURPLE}" stroke-width="1.5"/>\n'
    for p, c in pts:
        a, b = ax.px(*p)
        body += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="9" fill="{AMBER if c else BLUE}"/>\n'
    body += (f'<path d="M{cx - 9:.1f} {cy - 9:.1f} L{cx + 9:.1f} {cy + 9:.1f} M{cx + 9:.1f} {cy - 9:.1f} L{cx - 9:.1f} {cy + 9:.1f}" '
             f'stroke="{INK}" stroke-width="3.5" stroke-linecap="round"/>\n'
             + text(cx + 14, cy - 12, "x₀", 18, INK, weight=700))
    nb = sum(1 for _, c in near if c == 0)
    body += (text(ax.l, ax.t + ax.h + 30, f"K = 3: {nb} azuis, {3 - nb} laranja", 16, INK, weight=600)
             + text(ax.l, ax.t + ax.h + 54, f"p(azul | x₀) ≈ {nb}/3  →  classe azul", 16, PURPLE, weight=700))
    (OUT / "knn-neighbors.svg").write_text(svg(400, 420, body))


GRID_N = 70
_g = (np.arange(GRID_N) + 0.5) / GRID_N
GRID_Q = np.array([(u, v) for v in _g for u in _g])  # row-major from the bottom
P_GRID = p_amber(GRID_Q[:, 0], GRID_Q[:, 1])


def knn_k():
    X, lab = class_data()
    body = ""
    for i, k in enumerate((1, 50)):
        ax = Axes(20 + i * 410, 40, 370, 320)
        pred = knn_predict(GRID_Q, X, lab, k).reshape(GRID_N, GRID_N)
        body += text(ax.l, 26, f"K = {k}", 18, INK, weight=700)
        body += text(ax.l + ax.w, 26, "muito flexível" if k == 1 else "pouco flexível", 15, MUTED, "end", 600)
        body += class_panel(ax, X, lab, shade=pred, res=(GRID_N, GRID_N))
    body += class_legend(20, 392)
    (OUT / "knn-k.svg").write_text(svg(820, 440, body))


def knn_error():
    X, lab = class_data()
    ks = sorted(set(int(round(v)) for v in np.geomspace(1, 100, 40)))
    train, test = [], []
    for k in ks:
        train.append(float(np.mean(knn_predict(X, X, lab, k) != lab)))
        pred = knn_predict(GRID_Q, X, lab, k)
        test.append(float(np.mean(np.where(pred == 1, 1 - P_GRID, P_GRID))))
    bayes = float(np.mean(np.minimum(P_GRID, 1 - P_GRID)))
    inv = [1 / k for k in ks]
    ymax = math.ceil(max(max(train), max(test)) * 1.15 * 20) / 20
    ax = Axes(80, 30, 480, 280, xlim=(0.01, 1), ylim=(0, ymax), xlog=True)
    body = (ax.frame("1/K  (flexibilidade →)", "Error rate", lowhigh=False)
            + ax.ticks(xticks=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1),
                       yticks=[round(v, 2) for v in np.arange(0, ymax + 1e-9, 0.05)], yfmt="{:.2f}"))
    _, by = ax.px(0.01, bayes)
    body += (f'<line x1="{ax.l}" y1="{by:.1f}" x2="{ax.l + ax.w}" y2="{by:.1f}" stroke="{PURPLE}" stroke-width="2" stroke-dasharray="7 6"/>\n'
             + text(ax.l + 6, by + 18, "Bayes error rate", 14, PURPLE, weight=700))
    body += f'<path d="{ax.polyline(inv, test)}" fill="none" stroke="{RED}" stroke-width="3"/>\n'
    body += f'<path d="{ax.polyline(inv, train)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
    body += (text(ax.l + ax.w + 8, ax.px(1, test[0])[1] + 4, "Test error", 14, RED, weight=700)
             + text(ax.l + ax.w + 8, ax.px(1, train[0])[1] + 4, "Training error", 14, BLUE, weight=700))
    (OUT / "knn-error.svg").write_text(svg(700, 370, body))
    print(f"knn: bayes error={bayes:.3f}, best test={min(test):.3f} at K={ks[int(np.argmin(test))]}")


# --------------------------------------------------------------- other figures

def flexibility_interpretability():
    ax = Axes(70, 30, 500, 300)
    methods = [
        ("Subset selection", 0.08, 0.93, "start"),
        ("Lasso", 0.10, 0.82, "start"),
        ("Least squares", 0.28, 0.70, "start"),
        ("Generalized additive models", 0.47, 0.50, "start"),
        ("Trees", 0.47, 0.40, "start"),
        ("Bagging, boosting", 0.72, 0.24, "end"),
        ("Support vector machines", 0.76, 0.14, "end"),
        ("Deep learning", 0.93, 0.05, "end"),
    ]
    ax.l = 70
    body = ax.frame("Flexibilidade", "Interpretabilidade")
    for name, x, y, anchor in methods:
        px, py = ax.px(x, y)
        dx = 12 if anchor == "start" else -12
        body += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{PURPLE}"/>\n'
                 + text(px + dx, py + 5, name, 15, INK, anchor))
    (OUT / "flexibility-interpretability.svg").write_text(svg(600, 380, body))


def supervised_unsupervised():
    rng = np.random.default_rng(7)
    centres = [(0.28, 0.70, TEAL), (0.70, 0.72, BLUE), (0.50, 0.27, PURPLE)]
    pts = [(rng.normal(cx, 0.08), rng.normal(cy, 0.08), c) for cx, cy, c in centres for _ in range(22)]
    body = ""
    for i, (title, coloured) in enumerate([("Supervised: cada xᵢ tem um rótulo yᵢ", True),
                                           ("Unsupervised: só xᵢ, achar grupos", False)]):
        ax = Axes(30 + i * 420, 50, 350, 270)
        l, t, w, h = ax.l, ax.t, ax.w, ax.h
        body += (f'<rect x="{l}" y="{t}" width="{w}" height="{h}" rx="10" fill="none" stroke="{GRID}" stroke-width="1.5"/>\n'
                 + text(l, t - 16, title, 16, INK, weight=600))
        if not coloured:
            for cx, cy, _ in centres:
                px, py = ax.px(cx, cy)
                body += (f'<ellipse cx="{px:.1f}" cy="{py:.1f}" rx="62" ry="52" fill="none" '
                         f'stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="6 5"/>\n')
        for x, y, c in pts:
            px, py = ax.px(min(max(x, 0.03), 0.97), min(max(y, 0.04), 0.96))
            body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{c if coloured else POINT}" fill-opacity="0.85"/>\n'
    (OUT / "supervised-unsupervised.svg").write_text(svg(820, 340, body))


def cross_validation():
    k, bw, bh, gap = 5, 96, 36, 10
    l, t = 120, 70
    body = (text(l, 26, "Dados de treino (n observações) divididos em k = 5 <tspan font-style=\"italic\">folds</tspan>", 16, INK, weight=600)
            + f'<rect x="{l}" y="40" width="{k * bw}" height="12" rx="3" fill="{GRID}"/>\n')
    for r in range(k):
        y = t + r * (bh + gap)
        body += text(l - 14, y + bh / 2 + 5, f"Rodada {r + 1}", 14, MUTED, "end", 600)
        for c in range(k):
            val = c == r
            body += (f'<rect x="{l + c * bw + 2}" y="{y}" width="{bw - 4}" height="{bh}" rx="6" '
                     f'fill="{"#fbefd9" if val else "#dde8f7"}" stroke="{AMBER if val else BLUE}" stroke-width="1.5"/>\n')
            if val:
                body += text(l + c * bw + bw / 2, y + bh / 2 + 5, "validation", 13, "#7a5212", "middle", 700)
        body += text(l + k * bw + 18, y + bh / 2 + 5, f"MSE<tspan baseline-shift=\"sub\" font-size=\"11\">{r + 1}</tspan>", 16, INK, weight=700)
    y = t + k * (bh + gap) + 8
    body += (f'<rect x="{l}" y="{y}" width="22" height="16" rx="4" fill="#dde8f7" stroke="{BLUE}"/>'
             + text(l + 30, y + 13, "training (ajusta o modelo)", 14)
             + f'<rect x="{l + 250}" y="{y}" width="22" height="16" rx="4" fill="#fbefd9" stroke="{AMBER}"/>'
             + text(l + 280, y + 13, "validation (mede o erro)", 14))
    (OUT / "cross-validation.svg").write_text(svg(700, y + 30, body))


# ------------------------------------------- bias-variance (hand-written anim)

IRREDUCIBLE = 0.25


def bias2(x):
    return 0.9 * math.exp(-4 * x) + 0.02


def variance(x):
    return 0.03 + 0.8 * x ** 2.5


def bias_variance_paths():
    """Paths for animations/bias-variance.html: plot box 80,30 to 700,350."""
    ax = Axes(80, 30, 620, 320, ylim=(0, 1.3))
    total = lambda x: bias2(x) + variance(x) + IRREDUCIBLE
    for name, f in [("bias2", bias2), ("variance", variance), ("test", total)]:
        print(name, ax.path(f, 60)[:60], "...")


if __name__ == "__main__":
    old = OUT / "train-test-mse.svg"
    if old.exists():
        old.unlink()  # replaced by animations/overfitting.html
    income_education()
    parametric_vs_nonparametric()
    overfitting_animation()
    bayes_classifier()
    knn_neighbors()
    knn_k()
    knn_error()
    flexibility_interpretability()
    supervised_unsupervised()
    cross_validation()
    bias_variance_paths()
