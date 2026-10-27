"""Generate the SVG figures for Chapter 1 (Statistical Learning).

Run from the repo root:  python3 _scripts/ch1_figures.py
Writes into ch1/chapter-01-statistical-learning/figures/ and prints the curve
paths used by animations/bias-variance.html.
"""
import math
import random
from pathlib import Path

OUT = Path("ch1/chapter-01-statistical-learning/figures")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, AXIS = "#1b2333", "#5c6577", "#e3e7ee", "#6b7487"
TEAL, BLUE, PURPLE, AMBER, RED = "#2a8c7d", "#3d68a8", "#6c4fb0", "#b9822c", "#b4495a"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

# The error curves shared by the MSE figure and the bias-variance animation.
IRREDUCIBLE = 0.25


def bias2(x):
    return 0.9 * math.exp(-4 * x) + 0.02


def variance(x):
    return 0.03 + 0.8 * x ** 2.5


def test_mse(x):
    return bias2(x) + variance(x) + IRREDUCIBLE


def train_mse(x):
    return (bias2(x) + IRREDUCIBLE) * (1 - 0.7 * x)


def svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{FONT}">\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n{body}</svg>\n'
    )


class Axes:
    """Maps data coordinates in [x0, x1] x [y0, y1] onto a pixel box."""

    def __init__(self, left, top, width, height, xlim=(0, 1), ylim=(0, 1)):
        self.l, self.t, self.w, self.h = left, top, width, height
        self.xlim, self.ylim = xlim, ylim

    def px(self, x, y):
        (x0, x1), (y0, y1) = self.xlim, self.ylim
        return (self.l + (x - x0) / (x1 - x0) * self.w,
                self.t + self.h - (y - y0) / (y1 - y0) * self.h)

    def path(self, f, n=120):
        pts = []
        for i in range(n + 1):
            x = self.xlim[0] + (self.xlim[1] - self.xlim[0]) * i / n
            px, py = self.px(x, f(x))
            pts.append(f"{px:.1f} {py:.1f}")
        return "M" + " L".join(pts)

    def frame(self, xlabel, ylabel):
        l, t, w, h = self.l, self.t, self.w, self.h
        return (
            f'<path d="M{l} {t} V{t + h} H{l + w}" fill="none" stroke="{AXIS}" stroke-width="1.5"/>\n'
            f'<text x="{l + w / 2}" y="{t + h + 34}" text-anchor="middle" font-size="15" '
            f'font-weight="600" fill="{MUTED}">{xlabel}</text>\n'
            f'<text transform="translate({l - 22} {t + h / 2}) rotate(-90)" text-anchor="middle" '
            f'font-size="15" font-weight="600" fill="{MUTED}">{ylabel}</text>\n'
            f'<text x="{l}" y="{t + h + 18}" font-size="13" fill="{MUTED}">Baixa</text>\n'
            f'<text x="{l + w}" y="{t + h + 18}" text-anchor="end" font-size="13" fill="{MUTED}">Alta</text>\n'
        )


def flexibility_interpretability():
    ax = Axes(70, 30, 500, 300)
    methods = [
        ("Seleção de subconjuntos", 0.08, 0.93, "start"),
        ("Lasso", 0.10, 0.82, "start"),
        ("Mínimos quadrados", 0.28, 0.70, "start"),
        ("Modelos aditivos generalizados", 0.47, 0.50, "start"),
        ("Árvores", 0.47, 0.40, "start"),
        ("Bagging, boosting", 0.72, 0.24, "end"),
        ("Máquinas de vetores de suporte", 0.76, 0.14, "end"),
        ("Aprendizado profundo", 0.93, 0.05, "end"),
    ]
    body = ax.frame("Flexibilidade", "Interpretabilidade")
    for name, x, y, anchor in methods:
        px, py = ax.px(x, y)
        dx = 12 if anchor == "start" else -12
        body += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{PURPLE}"/>\n'
                 f'<text x="{px + dx:.1f}" y="{py + 5:.1f}" text-anchor="{anchor}" font-size="15" '
                 f'fill="{INK}">{name}</text>\n')
    (OUT / "flexibility-interpretability.svg").write_text(svg(600, 380, body))


def supervised_unsupervised():
    rng = random.Random(7)
    centres = [(0.28, 0.70, TEAL), (0.70, 0.72, BLUE), (0.50, 0.27, PURPLE)]
    pts = [(rng.gauss(cx, 0.08), rng.gauss(cy, 0.08), c) for cx, cy, c in centres for _ in range(22)]
    body = ""
    for i, (title, coloured) in enumerate([("Supervisionado: cada xᵢ tem um rótulo yᵢ", True),
                                           ("Não supervisionado: só xᵢ, achar grupos", False)]):
        ax = Axes(30 + i * 420, 50, 350, 270)
        l, t, w, h = ax.l, ax.t, ax.w, ax.h
        body += (f'<rect x="{l}" y="{t}" width="{w}" height="{h}" rx="10" fill="none" stroke="{GRID}" stroke-width="1.5"/>\n'
                 f'<text x="{l}" y="{t - 16}" font-size="16" font-weight="600" fill="{INK}">{title}</text>\n')
        if not coloured:
            for cx, cy, _ in centres:
                px, py = ax.px(cx, cy)
                body += (f'<ellipse cx="{px:.1f}" cy="{py:.1f}" rx="62" ry="52" fill="none" '
                         f'stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="6 5"/>\n')
        for x, y, c in pts:
            px, py = ax.px(min(max(x, 0.03), 0.97), min(max(y, 0.04), 0.96))
            fill = c if coloured else "#8d95a5"
            body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{fill}" fill-opacity="0.85"/>\n'
    (OUT / "supervised-unsupervised.svg").write_text(svg(820, 340, body))


def train_test_mse():
    ax = Axes(70, 30, 500, 290, ylim=(0, 1.35))
    body = ax.frame("Flexibilidade", "Erro quadrático médio")
    _, iy = ax.px(0, IRREDUCIBLE)
    body += (f'<line x1="{ax.l}" y1="{iy:.1f}" x2="{ax.l + ax.w}" y2="{iy:.1f}" stroke="{AXIS}" '
             f'stroke-width="1.5" stroke-dasharray="6 5"/>\n'
             f'<text x="{ax.l + ax.w - 4}" y="{iy - 8:.1f}" text-anchor="end" font-size="14" fill="{MUTED}">Var(ε)</text>\n')
    body += f'<path d="{ax.path(train_mse)}" fill="none" stroke="{BLUE}" stroke-width="3"/>\n'
    body += f'<path d="{ax.path(test_mse)}" fill="none" stroke="{RED}" stroke-width="3"/>\n'
    xmin = min((i / 400 for i in range(401)), key=test_mse)
    mx, my = ax.px(xmin, test_mse(xmin))
    body += (f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="6" fill="{RED}"/>\n'
             f'<text x="{ax.l + ax.w - 4}" y="{ax.px(1, test_mse(1))[1] - 12:.1f}" text-anchor="end" '
             f'font-size="15" font-weight="600" fill="{RED}">EQM de teste</text>\n'
             f'<text x="{ax.l + ax.w - 4}" y="{ax.px(1, train_mse(1))[1] - 12:.1f}" text-anchor="end" '
             f'font-size="15" font-weight="600" fill="{BLUE}">EQM de treino</text>\n')
    (OUT / "train-test-mse.svg").write_text(svg(600, 370, body))


def bayes_classifier():
    rng = random.Random(3)

    def boundary(x):
        return 0.5 + 0.22 * math.sin(2 * math.pi * (x - 0.1))

    ax = Axes(20, 20, 380, 320)
    body = ""
    # Shade the two Bayes decision regions.
    top = ax.path(boundary)
    l, t, w, h = ax.l, ax.t, ax.w, ax.h
    body += f'<path d="{top} L{l + w} {t} L{l} {t} Z" fill="{AMBER}" fill-opacity="0.10"/>\n'
    body += f'<path d="{top} L{l + w} {t + h} L{l} {t + h} Z" fill="{BLUE}" fill-opacity="0.10"/>\n'
    for _ in range(150):
        x, y = rng.random(), rng.random()
        p_amber = 1 / (1 + math.exp(-(y - boundary(x)) / 0.07))
        c = AMBER if rng.random() < p_amber else BLUE
        px, py = ax.px(x, y)
        body += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{c}" fill-opacity="0.9"/>\n'
    body += f'<path d="{top}" fill="none" stroke="{PURPLE}" stroke-width="3" stroke-dasharray="8 6"/>\n'
    body += f'<rect x="{l}" y="{t}" width="{w}" height="{h}" fill="none" stroke="{GRID}" stroke-width="1.5"/>\n'
    ly = t + h + 26
    body += (f'<line x1="{l}" y1="{ly}" x2="{l + 34}" y2="{ly}" stroke="{PURPLE}" stroke-width="3" stroke-dasharray="8 6"/>\n'
             f'<text x="{l + 42}" y="{ly + 5}" font-size="15" font-weight="600" fill="{PURPLE}">Fronteira de decisão de Bayes</text>\n'
             f'<circle cx="{l + 17}" cy="{ly + 26}" r="5" fill="{AMBER}"/><text x="{l + 42}" y="{ly + 31}" font-size="15" fill="{INK}">classe A</text>\n'
             f'<circle cx="{l + 137}" cy="{ly + 26}" r="5" fill="{BLUE}"/><text x="{l + 150}" y="{ly + 31}" font-size="15" fill="{INK}">classe B</text>\n')
    (OUT / "bayes-classifier.svg").write_text(svg(420, 410, body))


def bias_variance_paths():
    """Paths for the animation: plot box 80,30 to 700,350 in a 900x430 viewBox."""
    ax = Axes(80, 30, 620, 320, ylim=(0, 1.3))
    xmin = min((i / 400 for i in range(401)), key=test_mse)
    mx, my = ax.px(xmin, test_mse(xmin))
    _, iy = ax.px(0, IRREDUCIBLE)
    print("bias2   ", ax.path(bias2, 60))
    print("variance", ax.path(variance, 60))
    print("test    ", ax.path(test_mse, 60))
    print("irreducible y", round(iy, 1))
    print("optimum", round(mx, 1), round(my, 1), "x =", round(xmin, 3))
    for name, f in [("bias2", bias2), ("variance", variance), ("test", test_mse)]:
        print("end", name, [round(v, 1) for v in ax.px(1, f(1))])


if __name__ == "__main__":
    flexibility_interpretability()
    supervised_unsupervised()
    train_test_mse()
    bayes_classifier()
    bias_variance_paths()
