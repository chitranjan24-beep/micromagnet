#!/usr/bin/env python3
"""Generate all plotted figures into figures/out/. Run: python3 figures/make_figs.py [name ...]"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotlib import Axes, colormap, row

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
FIGS = {}


def fig(fn):
    FIGS[fn.__name__.replace('_', '-')] = fn
    return fn


def save(name, svg):
    open(os.path.join(OUT, name + '.svg'), 'w', encoding='utf-8').write(svg)


# ---------------------------------------------------------------- Bloch sphere helper
class Bloch:
    def __init__(self, size=300, elev=16, azim=-115, R=None, cx=None, cy=None):
        self.size = size
        self.R = R or size * 0.36
        self.cx = cx or size / 2
        self.cy = cy or size / 2
        self.e = math.radians(elev)
        self.a = math.radians(azim)
        self.parts = []

    def proj(self, x, y, z):
        a, e = self.a, self.e
        X = x * math.cos(a) - y * math.sin(a)
        Yp = x * math.sin(a) + y * math.cos(a)
        u = X
        v = z * math.cos(e) + Yp * math.sin(e)
        depth = -Yp * math.cos(e) + z * math.sin(e)  # >0 means towards viewer
        return self.cx + self.R * u, self.cy - self.R * v, depth

    def sphere(self):
        R, cx, cy = self.R, self.cx, self.cy
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" class="ln2" style="fill: var(--accent-soft); fill-opacity:.35"/>')
        # equator and meridians
        for curve in ('eq', 'xz', 'yz'):
            pts = []
            for t in np.linspace(0, 2 * math.pi, 181):
                if curve == 'eq':
                    p = (math.cos(t), math.sin(t), 0)
                elif curve == 'xz':
                    p = (math.cos(t), 0, math.sin(t))
                else:
                    p = (0, math.cos(t), math.sin(t))
                pts.append(self.proj(*p))
            self._curve(pts, 'thin')

    def _curve(self, pts, cls, front_only=False):
        seg, cur_front = [], None
        def flush():
            if len(seg) > 1:
                c = cls if cur_front else cls + ' dash'
                if not (front_only and not cur_front):
                    self.parts.append(f'<polyline class="{c}" points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in seg)}"/>')
        for u, v, d in pts:
            f = d >= -1e-9
            if cur_front is None:
                cur_front = f
            if f != cur_front:
                seg.append((u, v)); flush(); seg = [(u, v)]; cur_front = f
            else:
                seg.append((u, v))
        flush()

    def axes(self, labels=True):
        for vec, lab, off in [((1.3, 0, 0), 'x', (-12, 14)), ((0, 1.3, 0), 'y', (6, 4)), ((0, 0, 1.25), '', (0, 0))]:
            u0, v0, _ = self.proj(0, 0, 0)
            u1, v1, _ = self.proj(*vec)
            self.parts.append(f'<line class="thin" x1="{u0:.1f}" y1="{v0:.1f}" x2="{u1:.1f}" y2="{v1:.1f}"/>')
            if labels and lab:
                self.parts.append(f'<text class="sm" x="{u1 + off[0]:.1f}" y="{v1 + off[1]:.1f}">{lab}</text>')
        u0, v0, _ = self.proj(0, 0, -1.25)
        self.parts.append(f'<line class="thin dash" x1="{self.cx:.1f}" y1="{self.cy:.1f}" x2="{u0:.1f}" y2="{v0:.1f}"/>')
        if labels:
            u, v, _ = self.proj(0, 0, 1)
            self.parts.append(f'<text class="lbl" x="{u + 8:.1f}" y="{v - 10:.1f}">|0⟩</text>')
            u, v, _ = self.proj(0, 0, -1)
            self.parts.append(f'<text class="lbl" x="{u + 8:.1f}" y="{v + 18:.1f}">|1⟩</text>')

    def vector(self, x, y, z, cls='t1', label=None, loff=(6, -6), head=True):
        u0, v0, _ = self.proj(0, 0, 0)
        u1, v1, _ = self.proj(x, y, z)
        col = {'t1': 'f1', 't2': 'f2', 't3': 'f3', 't4': 'f4'}.get(cls, 'f1')
        self.parts.append(f'<line class="{cls}" style="stroke-width:2.6" x1="{u0:.1f}" y1="{v0:.1f}" x2="{u1:.1f}" y2="{v1:.1f}"/>')
        if head:
            self.parts.append(f'<circle class="{col}" cx="{u1:.1f}" cy="{v1:.1f}" r="4.2"/>')
        if label:
            self.parts.append(f'<text class="lbl {cls}t" x="{u1 + loff[0]:.1f}" y="{v1 + loff[1]:.1f}">{label}</text>')

    def path(self, pts3, cls='t2'):
        self._curve([self.proj(*p) for p in pts3], cls)

    def text(self, x, y, z, s, cls='sm', dx=0, dy=0):
        u, v, _ = self.proj(x, y, z)
        self.parts.append(f'<text class="{cls}" x="{u + dx:.1f}" y="{v + dy:.1f}">{s}</text>')

    def svg(self, w=None, h=None, title=None):
        w = w or self.size; h = h or self.size
        t = f'<text class="ttl" x="10" y="20">{title}</text>' if title else ''
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" xmlns="http://www.w3.org/2000/svg">{t}{"".join(self.parts)}</svg>'


def sph(theta, phi):
    return (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))


# ---------------------------------------------------------------- Part I figures
@fig
def bloch_basic():
    b = Bloch(size=360)
    b.sphere(); b.axes()
    th, ph = math.radians(55), math.radians(70)
    x, y, z = sph(th, ph)
    # projection lines
    u, v, _ = b.proj(x, y, 0); u0, v0, _ = b.proj(0, 0, 0); ut, vt, _ = b.proj(x, y, z)
    b.parts.append(f'<line class="thin dash" x1="{u0:.1f}" y1="{v0:.1f}" x2="{u:.1f}" y2="{v:.1f}"/>')
    b.parts.append(f'<line class="thin dash" x1="{u:.1f}" y1="{v:.1f}" x2="{ut:.1f}" y2="{vt:.1f}"/>')
    # theta arc
    arc = [sph(t, ph) for t in np.linspace(0, th, 30)]
    b.path([(0.42 * a, 0.42 * c, 0.42 * d) for a, c, d in arc], 'acc')
    b.text(*[0.42 * q for q in sph(th / 2, ph)], 'θ', 'lbl tacc', dx=4, dy=-2)
    arc2 = [(0.38 * math.cos(p), 0.38 * math.sin(p), 0) for p in np.linspace(0, ph, 30)]
    b.path(arc2, 'acc')
    b.text(0.42 * math.cos(ph / 2), 0.42 * math.sin(ph / 2), 0, 'φ', 'lbl tacc', dx=-2, dy=14)
    b.vector(x, y, z, 't1', '|ψ⟩', (8, -4))
    for lab, p, off in [('|+⟩', (1, 0, 0), (-34, 4)), ('|+i⟩', (0, 1, 0), (4, -8))]:
        b.text(*p, lab, 'sm', dx=off[0], dy=off[1])
    save('bloch-basic', b.svg())


@fig
def shots_error():
    N = np.logspace(1, 5, 200)
    ax = Axes(560, 290, xlim=(1, 5), ylim=(0, 0.16), xlabel='number of shots N (log scale)', ylabel='uncertainty of P',
              xticks=[1, 2, 3, 4, 5], xfmt=lambda v: {1: '10', 2: '100', 3: '1 000', 4: '10 000', 5: '100 000'}[int(v)])
    for p, cls, lab in [(0.5, 't1', 'P = 0.5'), (0.1, 't2', 'P = 0.1'), (0.02, 't3', 'P = 0.02')]:
        ax.line(np.log10(N), np.sqrt(p * (1 - p) / N), cls, lab)
    save('shots-error', ax.svg())


@fig
def larmor():
    b = Bloch(size=320)
    b.sphere(); b.axes()
    th = math.radians(55)
    ring = [sph(th, p) for p in np.linspace(0, 2 * math.pi, 120)]
    b.path(ring, 't2')
    for p in (0.0, 1.4, 2.8):
        x, y, z = sph(th, p)
        b.vector(x, y, z, 't1', None)
    x, y, z = sph(th, 2 * math.pi * 0.62)
    b.text(x, y, z, 'precession at f<tspan baseline-shift="sub" font-size="9">L</tspan>', 'sm t2t', dx=-60, dy=26)
    b.text(0, 0, 1.25, 'B<tspan baseline-shift="sub" font-size="9">0</tspan> along z', 'sm', dx=6, dy=-2)
    save('larmor', b.svg())


def rabi_p(t, Om, D):
    W = np.sqrt(Om ** 2 + D ** 2)
    return (Om ** 2 / W ** 2) * np.sin(W * t / 2) ** 2


@fig
def rabi_detuning():
    t = np.linspace(0, 3, 600)  # in units of 2pi/Omega -> use Om = 2pi
    Om = 2 * math.pi
    ax = Axes(600, 300, xlim=(0, 3), ylim=(0, 1.05), xlabel='drive time t (in units of 1/f<tspan baseline-shift="sub" font-size="9">Rabi</tspan>)',
              ylabel='P(|1⟩)', yticks=[0, 0.25, 0.5, 0.75, 1])
    for D, cls, lab in [(0, 't1', 'on resonance, Δ = 0'), (Om, 't2', 'Δ = Ω'), (2 * Om, 't3', 'Δ = 2Ω')]:
        ax.line(t, rabi_p(t, Om, D), cls, lab)
    save('rabi-detuning', ax.svg(legend_pos='tr'))


@fig
def rabi_chevron():
    Om = 2 * math.pi * 1.0
    t = np.linspace(0, 3, 300)
    D = np.linspace(-4, 4, 240) * Om
    TT, DD = np.meshgrid(t, D)
    P = rabi_p(TT, Om, DD)
    ax = Axes(600, 330, xlim=(0, 3), ylim=(-4, 4), xlabel='drive time t (units of 1/f<tspan baseline-shift="sub" font-size="9">Rabi</tspan>)',
              ylabel='detuning Δ/Ω', grid=False)
    ax.image(colormap(P[::-1], 'viridis'))
    save('rabi-chevron', ax.svg())


@fig
def rotating_frame():
    # resonant: rotation about x from north pole; detuned: tilted axis
    b1 = Bloch(size=300)
    b1.sphere(); b1.axes()
    path = [(0, -math.sin(a), math.cos(a)) for a in np.linspace(0, math.pi, 60)]
    b1.path(path, 't2')
    b1.vector(0, -math.sin(2.0), math.cos(2.0), 't1', None)
    b1.vector(1, 0, 0, 't3', 'drive axis (x)', (-70, 30))
    s1 = b1.svg(title='on resonance (Δ = 0)')
    b2 = Bloch(size=300)
    b2.sphere(); b2.axes()
    Om, D = 1.0, 1.0
    W = math.hypot(Om, D)
    n = np.array([Om / W, 0, D / W])
    pts = []
    s0 = np.array([0, 0, 1.0])
    for a in np.linspace(0, 2 * math.pi, 120):
        # Rodrigues rotation of s0 about n by angle a (left-handed sign to match Rabi picture)
        v = s0 * math.cos(a) + np.cross(n, s0) * math.sin(-a) * -1 + n * np.dot(n, s0) * (1 - math.cos(a))
        pts.append(tuple(v))
    b2.path(pts, 't2')
    b2.vector(*n, 't3', 'tilted axis', (-34, -10))
    s2 = b2.svg(title='detuned (Δ = Ω)')
    save('rotating-frame', row(s1, s2))


# Part I: decoherence
@fig
def t1_decay():
    t = np.linspace(0, 5, 300)
    ax = Axes(560, 280, xlim=(0, 5), ylim=(0, 1.05), xlabel='wait time τ / T<tspan baseline-shift="sub" font-size="9">1</tspan>', ylabel='P(|1⟩)')
    ax.line(t, np.exp(-t), 't1', 'ideal readout')
    ax.line(t, 0.08 + 0.80 * np.exp(-t), 't2', 'with 8% / 12% readout errors')
    ax.hline(np.exp(-1), 'thin dash', '1/e', lx=1.05)
    ax.vline(1, 'thin dash')
    save('t1-decay', ax.svg())


@fig
def ramsey():
    t = np.linspace(0, 4, 800)
    ax = Axes(600, 290, xlim=(0, 4), ylim=(0, 1.05), xlabel='free evolution time τ / T<tspan baseline-shift="sub" font-size="9">2</tspan>*', ylabel='P(|1⟩)')
    df = 3.0
    env = np.exp(-(t) ** 2)
    ax.line(t, 0.5 + 0.5 * env * np.cos(2 * math.pi * df * t), 't1', 'Ramsey fringes (detuning 3/T₂*)')
    ax.line(t, 0.5 + 0.5 * env, 't2 dash', 'Gaussian envelope exp(−(τ/T₂*)²)')
    ax.line(t, 0.5 - 0.5 * env, 't2 dash')
    save('ramsey', ax.svg(legend_pos='tr'))


@fig
def noise_psd():
    f = np.logspace(-3, 6, 300)
    lf = np.log10(f)
    ax = Axes(600, 300, xlim=(-3, 6), ylim=(-6, 4), xlabel='frequency f (Hz, log scale)', ylabel='log₁₀ S(f) (arb.)',
              xticks=list(range(-3, 7)), xfmt=lambda v: {-3: '1m', -2: '10m', -1: '0.1', 0: '1', 1: '10', 2: '100', 3: '1k', 4: '10k', 5: '100k', 6: '1M'}[int(v)])
    S = np.log10(1 / f + 1e-4)
    ax.line(lf, S, 't1', '1/f charge noise + white floor')
    ax.line(lf, np.log10(1 / f ** 2 * 1e-2 + 1e-5), 't2', 'random telegraph / drift (1/f²)')
    save('noise-psd', ax.svg(legend_pos='tr'))


@fig
def filter_functions():
    f = np.linspace(0.001, 6, 1200)  # units of 1/tau
    w = 2 * math.pi * f
    tau = 1.0
    F_ramsey = (np.sin(w * tau / 2) / (w / 2)) ** 2
    F_echo = (np.sin(w * tau / 4) ** 2 / (w / 4)) ** 2 * np.ones_like(w)
    F_echo = 16 * np.sin(w * tau / 4) ** 4 / w ** 2
    N = 4
    # CPMG-4 approximate filter function
    def cpmg(N):
        tk = (np.arange(1, N + 1) - 0.5) / N * tau
        y = 1 + (-1) ** (N + 1) * np.exp(1j * w * tau)
        for k, tt in enumerate(tk, start=1):
            y += 2 * (-1) ** k * np.exp(1j * w * tt)
        return np.abs(y) ** 2 / w ** 2
    ax = Axes(600, 300, xlim=(0, 6), ylim=(0, 1.05), xlabel='noise frequency f × τ', ylabel='filter function F(f) (norm.)')
    ax.line(f, F_ramsey / F_ramsey.max(), 't1', 'Ramsey (free evolution)')
    ax.line(f, F_echo / F_ramsey.max(), 't2', 'Hahn echo')
    c4 = cpmg(4)
    ax.line(f, c4 / F_ramsey.max(), 't3', 'CPMG, 4 π pulses')
    save('filter-functions', ax.svg(legend_pos='tr'))


@fig
def readout_hist():
    x = np.linspace(-4, 8, 600)
    s = 1.0
    g0 = np.exp(-(x - 0) ** 2 / (2 * s ** 2)) / (s * math.sqrt(2 * math.pi))
    g1 = np.exp(-(x - 4) ** 2 / (2 * s ** 2)) / (s * math.sqrt(2 * math.pi))
    ax = Axes(600, 290, xlim=(-4, 8), ylim=(0, 0.45), xlabel='integrated sensor signal (arb. units)', ylabel='probability density')
    ax.area(x, 0 * x, g0 * 0.5 / 0.5, 'fa1')
    ax.area(x, 0 * x, g1, 'fa2')
    ax.line(x, g0, 't1', 'spin down (|0⟩) shots')
    ax.line(x, g1, 't2', 'spin up (|1⟩) shots')
    ax.vline(2, 'acc', 'threshold', ly=0.42)
    save('readout-hist', ax.svg(legend_pos='tr'))


@fig
def rb_decay():
    m = np.arange(1, 401)
    p = 0.998
    F = 0.5 + 0.47 * p ** m
    rng = np.random.default_rng(3)
    ax = Axes(600, 290, xlim=(0, 400), ylim=(0.45, 1.0), xlabel='number of random Clifford gates m', ylabel='probability of correct final state')
    mm = np.arange(5, 401, 20)
    ax.scatter(mm, 0.5 + 0.47 * p ** mm + rng.normal(0, 0.008, len(mm)), 'f1', r=3.2, opacity=0.9)
    ax.line(m, F, 't2', 'fit A·pᵐ + B, p = 0.998')
    ax.hline(0.5, 'thin dash', 'fully mixed', lx=300)
    save('rb-decay', ax.svg(legend_pos='tr'))


if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
        print('made', n)
