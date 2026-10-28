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
    svg = svg.replace('id="k', f'id="{name}-k').replace('url(#k', f'url(#{name}-k')
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



# ---------------------------------------------------------------- Part II figures
def fermi(E, kT):
    return 1 / (1 + np.exp(np.clip(E / kT, -60, 60)))


@fig
def band_sige():
    # conduction band edge vs depth for Si/SiGe heterostructure under positive gate bias
    z = np.linspace(-60, 60, 1200)  # nm, 0 = top of well, surface at z=-50
    Ec = np.where(z < -50, 3000, 0.0)
    off = 160.0
    Ec = np.where((z >= -50) & (z < 0), off, Ec)
    Ec = np.where((z >= 0) & (z < 10), 0.0, Ec)
    Ec = np.where(z >= 10, off, Ec)
    field = -1.2  # meV/nm tilt from gate
    E = Ec + field * (z - 5)
    E = np.where(z < -50, np.nan, E)
    ax = Axes(600, 300, xlim=(-55, 60), ylim=(-60, 260), xlabel='depth below surface (nm, well from 0 to 10 nm)', ylabel='conduction band edge (meV)')
    ax.line(z, E, 't1', 'conduction band edge')
    ax.hline(8, 't2 dash', 'Fermi level (set by reservoir)', lx=-48)
    ax.text(-45, 225, 'SiGe spacer', 'sm'); ax.text(1, 230, 'Si well', 'sm'); ax.text(20, 210, 'SiGe buffer', 'sm')
    ax.vline(0, 'thin dash'); ax.vline(10, 'thin dash')
    save('band-sige', ax.svg(legend_pos='br'))


def ci_dot_current(Vsd, Vg, Ec=1.0, alpha=0.1, kT=0.02, n_levels=range(-3, 5)):
    I = np.zeros_like(Vsd)
    for N in n_levels:
        mu = N * Ec - alpha * Vg * 1000 * 1e-3  # all in meV, Vg in mV
        # symmetric bias, gate also couples weakly to source/drain potential via eta
        muL, muR = Vsd / 2, -Vsd / 2
        I += fermi(mu - muL, kT) - fermi(mu - muR, kT)
    return I


@fig
def coulomb_peaks():
    Vg = np.linspace(0, 40, 2000)  # mV
    kT = 0.02
    Ec, alpha = 1.0, 0.1
    G = np.zeros_like(Vg)
    for N in range(0, 6):
        mu = N * Ec + 0.5 - alpha * Vg
        G += 1 / np.cosh(mu / (2 * kT)) ** 2
    ax = Axes(600, 260, xlim=(0, 40), ylim=(0, 1.1), xlabel='plunger gate voltage V<tspan baseline-shift="sub" font-size="9">P</tspan> (mV, relative)', ylabel='conductance G (norm.)', yticks=[0, 0.5, 1])
    ax.line(Vg, G, 't1')
    for x, N in [(10, 1), (20, 2), (30, 3)]:
        ax.text(x, 0.55, f'N = {N}', 'sm', anchor='middle')
    ax.text(2.4, 0.55, 'N = 0', 'sm', anchor='middle')
    save('coulomb-peaks', ax.svg())


@fig
def coulomb_diamonds():
    vg = np.linspace(0, 40, 420)
    vsd = np.linspace(-2.5, 2.5, 300)
    VG, VSD = np.meshgrid(vg, vsd)
    I = np.zeros_like(VG)
    kT = 0.03
    Ec, alpha = 1.0, 0.1
    for N in range(-1, 7):
        mu = N * Ec + 0.5 - alpha * VG
        I += fermi(mu - VSD / 2, kT) - fermi(mu + VSD / 2, kT)
    G = np.gradient(I, vsd, axis=0)
    ax = Axes(600, 330, xlim=(0, 40), ylim=(-2.5, 2.5), xlabel='plunger gate voltage V<tspan baseline-shift="sub" font-size="9">P</tspan> (mV, relative)', ylabel='bias V<tspan baseline-shift="sub" font-size="9">SD</tspan> (mV)', grid=False)
    ax.image(colormap(np.abs(G[::-1]), 'magma', 0, np.percentile(np.abs(G), 99.5)))
    ax.text(10, 0, 'N = 1', 'lbl', anchor='middle', dy=4)
    ax.text(20, 0, 'N = 2', 'lbl', anchor='middle', dy=4)
    ax.text(30, 0, 'N = 3', 'lbl', anchor='middle', dy=4)
    for t in ax.parts[-3:]:
        pass
    save('coulomb-diamonds', ax.svg())


def dd_ground(V1, V2, U1=3.0, U2=3.0, Um=0.6, a=((0.10, 0.03), (0.03, 0.10)), kT=0.03, tc=0.0, Nmax=4):
    n1 = a[0][0] * V1 + a[0][1] * V2
    n2 = a[1][0] * V1 + a[1][1] * V2
    best = None
    states = [(i, j) for i in range(Nmax) for j in range(Nmax)]
    Es = []
    for (i, j) in states:
        E = 0.5 * U1 * (i - n1) ** 2 + 0.5 * U2 * (j - n2) ** 2 + Um * (i - n1) * (j - n2)
        Es.append(E)
    Es = np.array(Es)
    w = np.exp(-(Es - Es.min(axis=0)) / kT)
    Z = w.sum(axis=0)
    N1 = sum(w[k] * states[k][0] for k in range(len(states))) / Z
    N2 = sum(w[k] * states[k][1] for k in range(len(states))) / Z
    return N1, N2


@fig
def honeycomb():
    v1 = np.linspace(0, 32, 360); v2 = np.linspace(0, 32, 360)
    V1, V2 = np.meshgrid(v1, v2)
    N1, N2 = dd_ground(V1, V2, kT=0.05)
    sig = -(0.8 * N1 + 0.5 * N2) + 0.015 * V1 + 0.01 * V2
    d = np.hypot(np.gradient(sig, axis=1), np.gradient(sig, axis=0))
    ax = Axes(420, 400, xlim=(0, 32), ylim=(0, 32), xlabel='V<tspan baseline-shift="sub" font-size="9">P1</tspan> (mV, relative)', ylabel='V<tspan baseline-shift="sub" font-size="9">P2</tspan> (mV, relative)', grid=False, ml=56)
    ax.image(colormap(d[::-1], 'Greys', 0, np.percentile(d, 99)))
    labels = {}
    for i in range(4):
        for j in range(4):
            m = (np.round(N1) == i) & (np.round(N2) == j)
            if m.sum() > 400:
                labels[(i, j)] = (V1[m].mean(), V2[m].mean())
    for (i, j), (x, y) in labels.items():
        ax.text(x, y, f'({i},{j})', 'sm tacc', anchor='middle', dy=4)
    save('honeycomb', ax.svg())


@fig
def virtual_gates():
    v1 = np.linspace(0, 32, 300); v2 = np.linspace(0, 32, 300)
    V1, V2 = np.meshgrid(v1, v2)
    a = ((0.10, 0.05), (0.05, 0.10))
    N1, N2 = dd_ground(V1, V2, a=a, kT=0.05)
    sig = -(0.8 * N1 + 0.5 * N2)
    d = np.hypot(np.gradient(sig, axis=1), np.gradient(sig, axis=0))
    ax1 = Axes(330, 330, xlim=(0, 32), ylim=(0, 32), xlabel='V<tspan baseline-shift="sub" font-size="9">P1</tspan> (mV)', ylabel='V<tspan baseline-shift="sub" font-size="9">P2</tspan> (mV)', grid=False, title='physical gates')
    ax1.image(colormap(d[::-1], 'Greys', 0, np.percentile(d, 99)))
    # virtual: u1 = n1/a11 etc.  Map virtual grid to physical voltages
    A = np.array(a)
    u1 = np.linspace(0, 32, 300); u2 = np.linspace(0, 32, 300)
    U1, U2 = np.meshgrid(u1, u2)
    # we want n1 = 0.1*U1, n2 = 0.1*U2 -> solve A @ V = 0.1*U
    inv = np.linalg.inv(A) * 0.1
    P1 = inv[0, 0] * U1 + inv[0, 1] * U2
    P2 = inv[1, 0] * U1 + inv[1, 1] * U2
    N1v, N2v = dd_ground(P1, P2, a=a, kT=0.05)
    sigv = -(0.8 * N1v + 0.5 * N2v)
    dv = np.hypot(np.gradient(sigv, axis=1), np.gradient(sigv, axis=0))
    ax2 = Axes(330, 330, xlim=(0, 32), ylim=(0, 32), xlabel='virtual gate P1′ (mV)', ylabel='virtual gate P2′ (mV)', grid=False, title='virtual gates')
    ax2.image(colormap(dv[::-1], 'Greys', 0, np.percentile(dv, 99)))
    save('virtual-gates', row(ax1.svg(), ax2.svg()))


@fig
def interdot_anticrossing():
    eps = np.linspace(-60, 60, 400)  # ueV
    for_t = [(0, 't2', 't_c = 0'), (10, 't1', 't_c = 10 µeV'), (25, 't3', 't_c = 25 µeV')]
    ax = Axes(600, 300, xlim=(-60, 60), ylim=(-115, 60), xlabel='detuning ε (µeV)', ylabel='energy (µeV)')
    for tc, cls, lab in for_t:
        E = 0.5 * np.sqrt(eps ** 2 + 4 * tc ** 2)
        ax.line(eps, E, cls, lab); ax.line(eps, -E, cls)
    ax.text(-58, 48, '(1,0): electron on the left', 'sm'); ax.text(12, 48, '(0,1): electron on the right', 'sm')
    save('interdot-anticrossing', ax.svg(legend_pos='br'))


def gamma(f, L, Cp, R, Z0=50.0, Cc=None):
    w = 2 * np.pi * f
    Zdev = 1 / (1j * w * Cp + 1 / R)
    Z = 1j * w * L + Zdev
    return (Z - Z0) / (Z + Z0)


@fig
def reflecto_resonance():
    f = np.linspace(150e6, 350e6, 2000)
    L, Cp = 820e-9, 0.5e-12
    ax = Axes(330, 300, xlim=(150, 350), ylim=(0, 1.02), xlabel='frequency (MHz)', ylabel='|Γ| (reflection magnitude)', title='magnitude', mr=12)
    ax2 = Axes(330, 300, xlim=(150, 350), ylim=(-180, 180), xlabel='frequency (MHz)', ylabel='phase of Γ (deg)', title='phase', yticks=[-180, -90, 0, 90, 180], mr=12)
    for R, cls, lab in [(10e3, 't2', 'R = 10 kΩ'), (33e3, 't1', 'R = 33 kΩ (matched)'), (150e3, 't3', 'R = 150 kΩ'), (1e9, 't4', 'R → ∞ (blockade)')]:
        g = gamma(f, L, Cp, R)
        ax.line(f / 1e6, np.abs(g), cls, lab)
        ax2.line(f / 1e6, np.degrees(np.angle(g)), cls)
    save('reflecto-resonance', row(ax.svg(legend_pos='bl'), ax2.svg()))


@fig
def reflecto_sensitivity():
    R = np.logspace(3.5, 6.5, 400)
    L, Cp = 820e-9, 0.5e-12
    f0 = 1 / (2 * np.pi * np.sqrt(L * Cp))
    g = gamma(f0, L, Cp, R)
    dG = np.abs(np.gradient(g, np.log(R)))
    ax = Axes(600, 280, xlim=(3.5, 6.5), ylim=(0, 1.05), xlabel='sensor resistance R (log scale)', ylabel='value', xticks=[4, 5, 6], xfmt=lambda v: {4: '10 kΩ', 5: '100 kΩ', 6: '1 MΩ'}[int(v)])
    ax.line(np.log10(R), np.abs(g), 't1', '|Γ| at f₀')
    ax.line(np.log10(R), dG / dG.max(), 't2', 'sensitivity |dΓ/d ln R| (norm.)')
    ax.vline(np.log10(L / (50 * Cp)), 'thin dash', 'R_match = L/(Z₀C_p)', ly=0.2)
    save('reflecto-sensitivity', ax.svg(legend_pos='tr'))


@fig
def sensor_peak():
    V = np.linspace(-3, 3, 800)
    G0 = 1 / np.cosh(V / 0.6) ** 2
    G1 = 1 / np.cosh((V + 0.45) / 0.6) ** 2
    ax = Axes(600, 290, xlim=(-3, 3), ylim=(0, 1.1), xlabel='sensor gate voltage (mV, relative)', ylabel='sensor conductance (norm.)')
    ax.line(V, G0, 't1', 'qubit dot holds N electrons')
    ax.line(V, G1, 't2', 'one extra electron nearby: peak shifts')
    ax.vline(0.55, 'acc', 'operating point (steepest slope)', ly=1.02)
    ax.text(0.62, 0.48, 'ΔG', 'lbl tacc')
    ax.parts.append(f'<line class="acc" x1="{float(ax.X(0.58)):.1f}" x2="{float(ax.X(0.58)):.1f}" y1="{float(ax.Y(1/np.cosh(0.55/0.6)**2)):.1f}" y2="{float(ax.Y(1/np.cosh(1.0/0.6)**2)):.1f}"/>')
    save('sensor-peak', ax.svg(legend_pos='tl'))


# ---------------------------------------------------------------- Part III figures
@fig
def zeeman_valley():
    B = np.linspace(0, 1.2, 600)
    Ez = 2 * 57.88 * B  # ueV
    Evs = 70.0
    dsv = 3.0
    # basis energies
    e1 = -Ez / 2              # v-, down
    e2 = Ez / 2               # v-, up
    e3 = Evs - Ez / 2         # v+, down
    e4 = Evs + Ez / 2         # v+, up
    # anticrossing e2 <-> e3
    m = (e2 + e3) / 2; d = np.sqrt(((e2 - e3) / 2) ** 2 + dsv ** 2)
    ax = Axes(330, 310, xlim=(0, 1.2), ylim=(-80, 230), xlabel='magnetic field B (T)', ylabel='energy (µeV)', title='spin–valley levels', mr=12)
    ax.line(B, e1, 't1', 'v₋ ↓'); ax.line(B, m - d, 't2', 'v₋ ↑ / v₊ ↓ (mixed)'); ax.line(B, m + d, 't3'); ax.line(B, e4, 't4', 'v₊ ↑')
    Bh = Evs / (2 * 57.88)
    ax.vline(Bh, 'thin dash', 'hot spot', ly=-60)
    # relaxation rate
    rate = (B ** 5) * 1.0 + 40 * (dsv ** 2) / (((Ez - Evs) / 4) ** 2 + dsv ** 2) * (B > 0.05)
    ax2 = Axes(330, 310, xlim=(0, 1.2), ylim=(-1, 3.2), xlabel='magnetic field B (T)', ylabel='log₁₀ (1/T₁) (arb.)', title='relaxation rate', mr=12, yticks=[-1, 0, 1, 2, 3])
    ax2.line(B[30:], np.log10(rate[30:] + 0.1), 't2')
    ax2.vline(Bh, 'thin dash', 'E_Z = E_VS', ly=2.9)
    save('zeeman-valley', row(ax.svg(legend_pos='tl'), ax2.svg()))


@fig
def elzerman_trace():
    t = np.linspace(0, 3.0, 3000)  # ms
    V = np.where(t < 0.5, 1.0, np.where(t < 1.5, 0.0, np.where(t < 2.5, -1.0, 1.0)))
    rng = np.random.default_rng(7)
    base = np.where(t < 0.5, 0.2, np.where(t < 1.5, 0.0, np.where(t < 2.5, 0.0, 0.2)))
    # spin up: tunnels out at 1.62 ms, spin-down electron tunnels back in at 1.86
    up = base.copy(); up[(t > 1.62) & (t < 1.86)] += 0.6
    # dot empty during "empty" phase -> sensor level shift
    down = base.copy()
    for arr in (up, down):
        arr[(t >= 2.5) & (t < 2.52)] += 0.0
    up = up + rng.normal(0, 0.05, t.size); down = down + rng.normal(0, 0.05, t.size) - 1.2
    up = up + 0.0
    ax = Axes(640, 360, xlim=(0, 3), ylim=(-2.8, 2.6), xlabel='time (ms)', ylabel='signal (arb.)', yticks=[], grid=False)
    ax.line(t, 1.6 + 0.6 * V, 't4', 'plunger pulse: load / read / empty')
    ax.line(t, up, 't2', 'sensor trace, spin up: blip')
    ax.line(t, down, 't1', 'sensor trace, spin down: no blip')
    for x, lab in [(0.25, 'empty'), (1.0, 'load'), (2.0, 'read'), (2.75, 'empty')]:
        ax.text(x, 2.45, lab, 'sm', anchor='middle')
    save('elzerman-trace', ax.svg(legend_pos='bl'))


@fig
def st_energies():
    eps = np.linspace(-200, 200, 800)  # ueV, positive favours (2,0)
    tc, Ez = 25.0, 35.0
    # singlet hybridisation between S(1,1) at 0 and S(2,0) at -eps
    a, b = 0.0, -eps
    mid = (a + b) / 2; half = np.sqrt(((a - b) / 2) ** 2 + 2 * tc ** 2)
    ax = Axes(620, 330, xlim=(-200, 200), ylim=(-230, 90), xlabel='detuning ε (µeV)  ← (1,1) favoured      (2,0) favoured →', ylabel='energy (µeV)')
    ax.line(eps, mid - half, 't1', 'S (hybridised S(1,1)–S(2,0))')
    ax.line(eps, mid + half, 't1')
    ax.line(eps, 0 * eps, 't3', 'T₀(1,1)')
    ax.line(eps, Ez + 0 * eps, 't2', 'T₊(1,1), T₋(1,1)')
    ax.line(eps, -Ez + 0 * eps, 't2')
    ax.vline(120, 'acc')
    ax.text(114, 70, 'readout point: S → (2,0), T blocked', 'sm tacc', anchor='end')
    ax.text(-190, -40, 'J(ε) = gap between S and T₀', 'sm')
    save('st-energies', ax.svg(legend_pos='bl'))


@fig
def crot_spectrum():
    f = np.linspace(-30, 30, 1500)  # MHz relative to qubit-1 centre ... use two qubits
    f1, f2, J = -12.0, 12.0, 6.0
    def lor(x, x0, w=0.6):
        return w ** 2 / ((x - x0) ** 2 + w ** 2)
    ax = Axes(620, 280, xlim=(-25, 25), ylim=(0, 1.15), xlabel='drive frequency − 10 GHz (MHz)', ylabel='excitation probability', yticks=[0, 0.5, 1])
    y = 0.9 * (lor(f, f1 - J / 2) + lor(f, f1 + J / 2) + lor(f, f2 - J / 2) + lor(f, f2 + J / 2))
    ax.line(f, y, 't1')
    for x0, lab in [(f1 - J / 2, 'Q1 | Q2=↓'), (f1 + J / 2, 'Q1 | Q2=↑'), (f2 - J / 2, 'Q2 | Q1=↓'), (f2 + J / 2, 'Q2 | Q1=↑')]:
        ax.text(x0, 1.02, lab, 'sm', anchor='middle')
    ax.text(f1, 0.35, '← J →', 'sm tacc', anchor='middle')
    save('crot-spectrum', ax.svg())


@fig
def st_bloch():
    b = Bloch(size=320, cx=240)
    b.sphere(); b.axes(labels=False)
    u, v, _ = b.proj(0, 0, 1); b.parts.append(f'<text class="lbl" x="{u + 8:.1f}" y="{v - 10:.1f}">|S⟩</text>')
    u, v, _ = b.proj(0, 0, -1); b.parts.append(f'<text class="lbl" x="{u + 8:.1f}" y="{v + 18:.1f}">|T₀⟩</text>')
    b.vector(0, 0, 1.0, 't1', 'J (exchange)', (8, 22))
    b.vector(1.0, 0, 0, 't2', 'ΔB<tspan baseline-shift="sub" font-size="9">z</tspan> (field difference)', (-130, 22))
    th = math.atan2(1.0, 0.6)
    n = (math.sin(th), 0, math.cos(th))
    b.vector(*n, 't3', 'both on', (6, -4))
    u, v, _ = b.proj(1, 0, 0); b.parts.append(f'<text class="sm" x="{u - 120:.1f}" y="{v + 42:.1f}">|↑↓⟩ and |↓↑⟩ lie on the ±x axis</text>')
    save('st-bloch', b.svg(w=480))


@fig
def eo_axes():
    b = Bloch(size=320, cx=250)
    b.sphere(); b.axes(labels=False)
    u, v, _ = b.proj(0, 0, 1); b.parts.append(f'<text class="lbl" x="{u + 8:.1f}" y="{v - 10:.1f}">|0<tspan baseline-shift="sub" font-size="9">L</tspan>⟩</text>')
    u, v, _ = b.proj(0, 0, -1); b.parts.append(f'<text class="lbl" x="{u + 8:.1f}" y="{v + 18:.1f}">|1<tspan baseline-shift="sub" font-size="9">L</tspan>⟩</text>')
    b.vector(0, 0, 1.0, 't1', 'J₁₂ axis (z)', (8, 26))
    a = math.radians(120)
    b.vector(math.sin(a), 0, math.cos(a), 't2', 'J₂₃ axis (n), 120° from z', (-170, 30))
    a2 = math.radians(-120)
    b.vector(math.sin(a2), 0, math.cos(a2), 't3', 'J₁₃ axis (m), triangular devices', (6, 34))
    save('eo-axes', b.svg(w=520))


if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
        print('made', n)
