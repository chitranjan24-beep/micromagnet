#!/usr/bin/env python3
"""Generate all plotted figures for the experiment guide into figures/out/.
Run: python3 figures/make_figs.py [name ...]   (no names = all figures)
All simulated data use the book's running-example device numbers (see chapter 2)."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotlib import Axes, colormap, row, column

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
FIGS = {}
RNG = np.random.default_rng(7)

# physical constants in convenient units
KB_UEV = 86.17          # ueV per K
H_UEV_GHZ = 4.1357      # ueV per GHz
MUB_UEV = 57.88         # ueV per T


def fig(fn):
    FIGS[fn.__name__.replace('_', '-')] = fn
    return fn


def save(name, svg):
    svg = svg.replace('id="k', f'id="{name}-k').replace('url(#k', f'url(#{name}-k')
    open(os.path.join(OUT, name + '.svg'), 'w', encoding='utf-8').write(svg)


def pow10(v):
    """Tick label for a log10 axis value."""
    e = int(round(v))
    if abs(v - e) > 1e-6:
        return f'{10 ** v:.3g}'
    return {-3: '0.001', -2: '0.01', -1: '0.1', 0: '1', 1: '10', 2: '100', 3: '1000', 4: '10⁴', 5: '10⁵', 6: '10⁶'}.get(e, f'10^{e}')


def fermi(E, kT):
    return 1.0 / (1.0 + np.exp(np.clip(E / kT, -60, 60)))


# ================================================================ Part A
@fig
def energy_ladder():
    items = [
        (0.017, 'Rabi frequency 4 MHz', 't4'),
        (0.04, 'exchange J ≈ 10 MHz (on)', 't4'),
        (4.3, 'thermal energy kT at 50 mK', 't2'),
        (40, 'interdot tunnel coupling ≈ 10 GHz', 't3'),
        (64, 'Zeeman energy at 0.55 T (15.4 GHz)', 't1'),
        (150, 'valley splitting (100–300 µeV)', 't3'),
        (3000, 'orbital spacing ≈ 3 meV', 't3'),
        (6500, 'charging energy ≈ 6.5 meV', 't3'),
    ]
    ax = Axes(w=640, h=330, xlim=(-2.3, 4.2), ylim=(-0.5, len(items) - 0.5), xlabel='energy (µeV, logarithmic scale)',
              xticks=[-2, -1, 0, 1, 2, 3, 4], yticks=[], xfmt=pow10, ml=20, mr=20, mt=16, grid=True)
    for i, (E, lab, cls) in enumerate(items):
        x = math.log10(E)
        ax.scatter([x], [i], cls=cls.replace('t', 'f'), r=5, opacity=0.95)
        anchor = 'end' if x > 1.2 else 'start'
        ax.text(x, i, lab, anchor=anchor, dx=(-9 if anchor == 'end' else 9), dy=4)
    save('energy-ladder', ax.svg())


@fig
def time_ladder():
    items = [  # (lo, hi) in seconds
        (60e-9, 130e-9, 'single-qubit gate (π/2 … π)', 't1'),
        (100e-9, 300e-9, 'two-qubit gate', 't1'),
        (1e-6, 10e-6, 'Pauli-blockade readout', 't2'),
        (100e-6, 1e-3, 'Elzerman readout window', 't2'),
        (2e-6, 20e-6, 'T2* (28Si)', 't3'),
        (20e-6, 200e-6, 'Hahn echo T2', 't3'),
        (0.2e-3, 3e-3, 'T2 with dynamical decoupling', 't3'),
        (5e-3, 1.0, 'T1 (field dependent)', 't3'),
        (60, 3 * 3600, 'drift that forces recalibration', 't4'),
        (1 * 86400, 3 * 86400, 'one cooldown', 't4'),
    ]
    ax = Axes(w=640, h=360, xlim=(-7.5, 5.8), ylim=(-0.6, len(items) - 0.4), xlabel='time (seconds, logarithmic scale)',
              xticks=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
              xfmt=lambda v: {-7: '100 ns', -6: '1 µs', -5: '', -4: '100 µs', -3: '1 ms', -2: '', -1: '0.1 s', 0: '1 s', 1: '', 2: '', 3: '1000 s', 4: '', 5: '10⁵ s'}.get(int(round(v)), ''),
              yticks=[], ml=20, mr=20, mt=16)
    for i, (lo, hi, lab, cls) in enumerate(items):
        a, b = math.log10(lo), math.log10(hi)
        ax.line([a, b], [i, i], cls=cls, extra='style="stroke-width:7;stroke-linecap:round"')
        if b < 1.0:
            ax.text(b, i, lab, dx=10, dy=4)
        else:
            ax.text(a, i, lab, anchor='end', dx=-10, dy=4)
    save('time-ladder', ax.svg())


# ---------------------------------------------------------------- chapter 2
@fig
def band_profile():
    """Conduction-band edge versus depth for a gate below and above the accumulation threshold."""
    z = np.linspace(-12, 70, 1200)               # nm; gate oxide from -12 to 0, SiGe spacer 0-30, Si well 30-38
    dEc = 0.15                                   # Si/Si0.7Ge0.3 conduction-band offset (eV), approximate
    base = np.where((z >= 30) & (z <= 38), 0.0, dEc)
    base = np.where(z < 0, 2.0, base)            # Al2O3 barrier (drawn clipped)
    ax = Axes(w=600, h=320, xlim=(-12, 70), ylim=(-0.25, 0.45), xlabel='depth below the surface (nm)',
              ylabel='conduction-band edge (eV)', mt=18)
    for Vg, cls, lab in [(0.0, 't2', 'gate below threshold'), (0.55, 't1', 'gate above threshold')]:
        # potential drops linearly from the gate into the substrate over ~60 nm (screened deeper by the electrons)
        phi = -Vg * np.clip(1 - (z + 12) / 90, 0, 1) * 0.55
        Ec = base + phi + 0.18
        ax.line(z[z >= 0], Ec[z >= 0], cls=cls, label=lab)
    ax.hline(0.0, cls='thin dash', label='Fermi level of the ohmic contacts', lx=40)
    ax.area(np.array([30, 38]), np.array([-0.25, -0.25]), np.array([0.45, 0.45]), cls='fa3')
    ax.text(34, 0.40, 'Si well', anchor='middle')
    ax.text(15, 0.40, 'SiGe spacer', anchor='middle')
    ax.text(54, 0.40, 'SiGe buffer', anchor='middle')
    ax.area(np.array([-12, 0]), np.array([-0.25, -0.25]), np.array([0.45, 0.45]), cls='fa2')
    ax.text(-6, 0.40, 'oxide', anchor='middle')
    save('band-profile', ax.svg(legend_pos='br'))


def magnet_field(xs, z0, mags, Ms=1.8, n=(40, 20, 40)):
    """Stray field (tesla) of uniformly magnetised boxes (magnetisation along +y, mu0*Ms in T) at points (x, 0, -z0).
    Each magnet: (x0, x1, y0, y1, zb, zt) in nm with z up. Numerical dipole sum."""
    Bx = np.zeros_like(xs); By = np.zeros_like(xs); Bz = np.zeros_like(xs)
    for (x0, x1, y0, y1, zb, zt) in mags:
        gx = np.linspace(x0, x1, n[0]); gy = np.linspace(y0, y1, n[1] * 6); gz = np.linspace(zb, zt, n[2] // 4)
        dv = (x1 - x0) / n[0] * (y1 - y0) / (n[1] * 6) * (zt - zb) / (n[2] // 4)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing='ij')
        X = X.ravel(); Y = Y.ravel(); Z = Z.ravel()
        for i, xp in enumerate(xs):
            rx = xp - X; ry = 0 - Y; rz = -z0 - Z
            r2 = rx * rx + ry * ry + rz * rz; r = np.sqrt(r2)
            # dipole m = Ms dv along y: B = mu0/(4pi) [3 (m.r) r / r^5 - m / r^3]
            mdotr = ry
            fac = Ms * dv / (4 * np.pi)
            Bx[i] += np.sum(fac * 3 * mdotr * rx / r ** 5)
            By[i] += np.sum(fac * (3 * mdotr * ry / r ** 5 - 1 / r ** 3))
            Bz[i] += np.sum(fac * 3 * mdotr * rz / r ** 5)
    return Bx, By, Bz


@fig
def micromagnet_field():
    xs = np.linspace(-300, 300, 61)
    # two Co magnets (200 nm thick) whose pole faces look at each other across a 300 nm gap (along y, the field
    # direction); the lower magnet starts at x = 0, which breaks the symmetry and creates gradients along the dot array.
    mags = [(-1000, 1000, 150, 1500, 0, 200), (0, 1000, -1500, -150, 0, 200)]
    Bx, By, Bz = magnet_field(xs, 120, mags)
    ax = Axes(w=600, h=320, xlim=(-300, 300), ylim=(-0.05, 0.25), xlabel='position along the dot array, x (nm)',
              ylabel='stray field at the dots (T)', mt=18)
    ax.line(xs, By, cls='t1', label='B∥: along the applied field')
    ax.line(xs, Bz, cls='t2', label='B⊥: out of plane')
    ax.vline(-45, cls='thin dash', label='dot 1', ly=0.0)
    ax.vline(45, cls='thin dash', label='dot 2', ly=0.0)
    gpar = np.gradient(By, xs) * 1e3; gperp = np.gradient(Bz, xs) * 1e3
    for xd in (-45, 45):
        print('x=%d: B_par %.3f T, dB_par/dx %.2f mT/nm, B_perp %.3f T, dB_perp/dx %.2f mT/nm' % (
            xd, np.interp(xd, xs, By), np.interp(xd, xs, gpar), np.interp(xd, xs, Bz), np.interp(xd, xs, gperp)))
    save('micromagnet-field', ax.svg(legend_pos='tr'))


# ---------------------------------------------------------------- chapter 4
@fig
def fermi_readout():
    """Occupation of reservoir states near the Fermi level, compared with the two spin levels during readout."""
    E = np.linspace(-90, 90, 600)       # ueV relative to EF
    ax = Axes(w=600, h=320, xlim=(-90, 90), ylim=(-0.02, 1.05), xlabel='energy relative to the reservoir Fermi level (µeV)',
              ylabel='probability a reservoir state is EMPTY', mt=18)
    for Te, cls in [(0.05, 't1'), (0.1, 't3'), (0.2, 't2')]:
        kT = KB_UEV * Te
        ax.line(E, 1 - fermi(E, kT), cls=cls, label=f'Tₑ = {int(Te * 1000)} mK')
    ez = 63.8
    ax.vline(-ez / 2, cls='thin dash', label='spin-↓ level', ly=0.92)
    ax.vline(ez / 2, cls='thin dash', label='spin-↑ level', ly=0.8)
    for Te in (0.05, 0.2):
        kT = KB_UEV * Te
        print('Te', Te, 'P(empty at spin-down level) =', 1 - fermi(-ez / 2, kT), ' P(empty at spin-up) =', 1 - fermi(ez / 2, kT))
    save('fermi-readout', ax.svg(legend_pos='tl'))


def cascade(stages, att_db):
    """Noise temperature leaving each stage for a line with attenuators att_db[i] at temperature stages[i] (K)."""
    T = 300.0
    out = [T]
    for Ts, a in zip(stages, att_db):
        A = 10 ** (a / 10)
        T = T / A + Ts * (1 - 1 / A)
        out.append(T)
    return out


@fig
def line_noise():
    stages = [50, 4, 0.8, 0.1, 0.012]
    names = ['300 K', '50 K', '4 K', 'still', 'cold plate', 'MXC']
    cfgs = [([0, 20, 10, 10, 10], 't1', 'drive line: 20+10+10+10 dB'),
            ([0, 20, 0, 0, 0], 't2', 'only 20 dB at 4 K'),
            ([0, 0, 0, 0, 0], 't4', 'no attenuators')]
    ax = Axes(w=600, h=320, xlim=(-0.3, 5.3), ylim=(-2.2, 2.7), xlabel='stage',
              ylabel='noise temperature (K, log scale)', xticks=list(range(6)), xfmt=lambda v: names[int(round(v))] if 0 <= round(v) < 6 else '',
              yticks=[-2, -1, 0, 1, 2], yfmt=pow10, mt=18)
    for att, cls, lab in cfgs:
        T = cascade(stages, att)
        ax.line(np.arange(6), np.log10(T), cls=cls, label=lab)
        ax.scatter(np.arange(6), np.log10(T), cls=cls.replace('t', 'f'), r=3, opacity=0.9)
        f = 15.4e9
        hfk = 6.626e-34 * f / 1.381e-23
        print(lab, ['%.3g' % t for t in T], 'n(15.4GHz) at MXC = %.3g' % (1 / (math.exp(hfk / T[-1]) - 1)))
    save('line-noise', ax.svg(legend_pos='tr'))


# ================================================================ Part C
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


@fig
def turn_on():
    V = np.linspace(0, 0.8, 500)
    def I(V, Vth):  # nA, a channel that turns on and then rises roughly linearly (lead resistance limited)
        x = (V - Vth) / 0.012
        return 2.0 * np.log1p(np.exp(np.clip(x, -50, 50))) * 0.012 / 0.1
    up, dn = I(V, 0.36), I(V, 0.335)
    ax = Axes(330, 300, xlim=(0, 0.8), ylim=(-0.2, 10.5), xlabel='lead gates LL = LR (V)', ylabel='current O1 → O2 (nA)',
              title='accumulation (turn-on)', mr=12)
    ax.line(V, up + RNG.normal(0, 0.03, V.size), 't1', 'sweep up')
    ax.line(V, dn + RNG.normal(0, 0.03, V.size), 't2', 'sweep down')
    ax.vline(0.36, 'thin dash', 'threshold', ly=8.8)
    Vg = np.linspace(0, 2.2, 400)
    Ig = 0.3 * RNG.normal(0, 1, Vg.size) + 1e3 * np.exp((Vg - 2.2) / 0.11)   # pA
    ax2 = Axes(330, 300, xlim=(0, 2.2), ylim=(-5, 120), xlabel='voltage on one gate (V)', ylabel='current drawn by the gate (pA)',
               title='leakage check', mr=12)
    ax2.line(Vg, Ig, 't3')
    ax2.hline(10, 'thin dash', 'tolerable leakage', lx=0.05)
    ax2.vline(1.55, 'thin dash', 'onset', ly=100)
    save('turn-on', row(ax.svg(legend_pos='tl'), ax2.svg()))


@fig
def pinch_off():
    V = np.linspace(-0.1, 0.8, 600)
    ax = Axes(600, 310, xlim=(-0.1, 0.8), ylim=(-0.2, 6.5), xlabel='barrier voltage (V), all other gates open',
              ylabel='conductance (e²/h)', mt=18)
    for Vpo, cls, lab, w in [(0.18, 't1', 'B1', 0.035), (0.26, 't3', 'B2', 0.03), (0.21, 't2', 'B3', 0.04)]:
        G = 5.2 * sigmoid((V - Vpo - 0.06) / w)
        # weak resonances near pinch-off: unintended small dots / disorder
        G += 0.35 * np.exp(-((V - Vpo - 0.02) / 0.006) ** 2) + 0.2 * np.exp(-((V - Vpo - 0.045) / 0.007) ** 2)
        G += RNG.normal(0, 0.02, V.size)
        ax.line(V, G, cls, f'{lab}: pinch-off ≈ {Vpo:.2f} V')
    save('pinch-off', ax.svg(legend_pos='br'))


@fig
def barrier_map():
    """Current through dot 1 versus its two barriers B1 (x) and B2 (y), plunger P1 fixed (log colour scale)."""
    b1 = np.linspace(0.10, 0.50, 400); b2 = np.linspace(0.15, 0.55, 400)
    B1, B2 = np.meshgrid(b1, b2)
    T1 = sigmoid((B1 - 0.27) / 0.03); T2 = sigmoid((B2 - 0.34) / 0.03)
    Gs = 1 / (1 / (T1 + 1e-12) + 1 / (T2 + 1e-12))          # two barriers in series (classical)
    mu = -(B1 + B2) / 0.022                                 # dot level falls as both barriers rise (units of peak spacing)
    frac = mu - np.floor(mu)
    peaks = np.exp(-((frac - 0.5) / 0.07) ** 2)
    closed = sigmoid((0.6 - (T1 + T2)) / 0.12)              # Coulomb blockade only when the barriers are fairly closed
    G = Gs * ((1 - closed) + closed * (0.02 + peaks))
    Z = np.log10(G + 1e-5)
    ax = Axes(430, 400, xlim=(0.10, 0.50), ylim=(0.15, 0.55), xlabel='B1 (V)', ylabel='B2 (V)', grid=False, ml=56, mt=18)
    ax.image(colormap(Z[::-1], 'viridis', -3.2, 0))
    ax.text(0.40, 0.49, 'open channel', 'sm', anchor='middle')
    ax.text(0.12, 0.17, 'pinched off', 'sm', anchor='start', style='fill:#fff')
    ax.text(0.30, 0.235, '← a dot forms here:', 'sm', anchor='start', style='fill:#fff')
    ax.text(0.30, 0.235, '   diagonal Coulomb peaks', 'sm', anchor='start', style='fill:#fff', dy=14)
    save('barrier-map', ax.svg())


def ci_current(Vsd, Vg, Ec, alpha, kT, Vg0, Nrange, eo=None, eta=0.5):
    """Sequential-tunnelling current (arb.) through a single dot in the constant-interaction model.
    Energies in meV, voltages in mV. eo: optional excited-state energy (meV) opening an extra channel."""
    I = np.zeros_like(Vsd)
    for N in Nrange:
        mu = N * Ec - alpha * (Vg - Vg0)
        muS, muD = eta * Vsd, -(1 - eta) * Vsd
        I += fermi(mu - muS, kT) - fermi(mu - muD, kT)
        if eo is not None:
            mue = mu + eo
            # excited channel adds current only inside the bias window where the ground channel also conducts
            win = (fermi(mu - muS, kT) - fermi(mu - muD, kT))
            I += 0.6 * (fermi(mue - muS, kT) - fermi(mue - muD, kT)) * np.sign(win) * (np.abs(win) > 0.5)
    return I


@fig
def diamonds():
    vg = np.linspace(400, 640, 520)          # mV on P1
    vsd = np.linspace(-9, 9, 360)            # mV bias
    VG, VSD = np.meshgrid(vg, vsd)
    Ec, alpha = 6.5, 0.10
    I = ci_current(VSD, VG, Ec, alpha, kT=0.06, Vg0=450, Nrange=range(0, 4), eo=3.0)
    G = np.abs(np.gradient(I, vsd, axis=0))
    ax = Axes(600, 360, xlim=(400, 640), ylim=(-9, 9), xlabel='plunger P1 (mV)', ylabel='bias V_SD (mV)', grid=False, mt=18)
    ax.image(colormap(G[::-1], 'magma', 0, np.percentile(G, 99.3)))
    for x, lab in [(420, 'N = 0'), (482.5, 'N = 1'), (547.5, 'N = 2'), (612, 'N = 3')]:
        ax.text(x, 0, lab, 'lbl', anchor='middle', dy=4, style='fill:#fff')
    save('diamonds', ax.svg())


@fig
def te_fit():
    V = np.linspace(-0.6, 0.6, 400)   # mV around a peak
    alpha = 0.10
    ax = Axes(330, 300, xlim=(-0.6, 0.6), ylim=(-0.03, 1.1), xlabel='plunger offset (mV)', ylabel='conductance (norm.)',
              title='peak shape', mr=12)
    for Te, cls in [(0.05, 't1'), (0.2, 't2')]:
        kT = KB_UEV * Te / 1000      # meV
        G = 1 / np.cosh(alpha * V / (2 * kT)) ** 2
        ax.line(V, G + RNG.normal(0, 0.01, V.size), cls, f'Tₑ = {int(Te * 1000)} mK')
    Tm = np.linspace(0.01, 0.5, 100)
    Te = np.sqrt(Tm ** 2 + 0.048 ** 2)
    fwhm = 3.525 * KB_UEV * Te / alpha / 1000   # mV
    ax2 = Axes(330, 300, xlim=(0, 0.5), ylim=(0, 1.6), xlabel='mixing-chamber temperature (K)', ylabel='peak FWHM (mV)',
               title='width versus temperature', mr=12)
    ax2.line(Tm, 3.525 * KB_UEV * Tm / alpha / 1000, 'thin dash', 'Tₑ = T_MXC (ideal)')
    ax2.line(Tm, fwhm, 't1', 'measured: saturates')
    pts = np.array([0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ax2.scatter(pts, 3.525 * KB_UEV * np.sqrt(pts ** 2 + 0.048 ** 2) / alpha / 1000 * (1 + RNG.normal(0, 0.03, pts.size)), 'f1', r=3.5, opacity=0.9)
    save('te-fit', row(ax.svg(legend_pos='tr'), ax2.svg(legend_pos='tl')))


@fig
def sensor_operating():
    V = np.linspace(-1.5, 1.5, 700)    # mV on SP relative to a peak
    w = 0.32
    G0 = 0.45 / np.cosh(V / w) ** 2
    shift = 0.16                       # mV equivalent shift when one electron enters D2
    G1 = 0.45 / np.cosh((V + shift) / w) ** 2
    slope = np.abs(np.gradient(G0, V))
    ax = Axes(600, 310, xlim=(-1.5, 1.5), ylim=(-0.02, 0.52), xlabel='sensor plunger SP (mV, relative to peak centre)',
              ylabel='sensor conductance (e²/h)', mt=18)
    ax.line(V, G0, 't1', 'D2 holds N electrons')
    ax.line(V, G1, 't2', 'D2 holds N+1 electrons')
    ax.line(V, slope / slope.max() * 0.45, 'thin dash', 'slope |dG/dV| (scaled)')
    Vop = V[np.argmax(slope * (V > 0))]
    ax.vline(Vop, 'acc', 'operating point', ly=0.49)
    print('operating point %.3f mV, dG = %.3f e2/h' % (Vop, 0.45 / np.cosh(Vop / w) ** 2 - 0.45 / np.cosh((Vop + shift) / w) ** 2))
    save('sensor-operating', ax.svg(legend_pos='tl'))


@fig
def sensor_compensation():
    VP = np.linspace(300, 520, 1500)                # mV on P2
    trans = [345, 410, 468]                          # D2 charge transitions (0->1, 1->2, 2->3)
    nD2 = sum(sigmoid((VP - t) / 0.6) for t in trans)
    w = 0.32
    cross = 0.012                                    # SP-equivalent shift per mV of P2 (cross-capacitance)
    shift = 0.16
    Vop = 0.28
    def G(Veff):
        return 0.45 / np.cosh(Veff / w) ** 2
    # uncompensated: P2 sweeps the sensor along its own peak and through neighbouring peaks (period 1.6 mV in SP units)
    per = 50.0                                       # sensor peak spacing on SP: E_C/alpha = 4 meV / 0.08 eV/V
    Veff_u = Vop - cross * (VP - 300) + shift * nD2
    Gu = sum(G(Veff_u + k * per) for k in range(-6, 6))
    Veff_c = Vop + shift * nD2 - 0.0                  # compensated: SP moved by -cross*dVP2 automatically
    Gc = G(Veff_c)
    ax = Axes(600, 320, xlim=(300, 520), ylim=(-0.02, 1.02), xlabel='plunger P2 (mV)', ylabel='sensor signal (e²/h, offset)', mt=18, yticks=[])
    ax.line(VP, Gu * 0.9 + 0.5 + RNG.normal(0, 0.004, VP.size), 't2', 'without compensation')
    ax.line(VP, Gc * 1.0 + 0.02 + RNG.normal(0, 0.004, VP.size), 't1', 'with SP compensated')
    for t, lab in zip(trans, ['0→1', '1→2', '2→3']):
        ax.vline(t, 'thin dash', lab, ly=0.45)
    save('sensor-compensation', ax.svg(legend_pos='tr'))


# ================================================================ Part D
L_TANK, CP_TANK = 1.2e-6, 0.5e-12


def gamma(f, R, L=L_TANK, Cp=CP_TANK, Z0=50.0):
    w = 2 * np.pi * f
    Zdev = 1 / (1j * w * Cp + 1 / R)
    Z = 1j * w * L + Zdev
    return (Z - Z0) / (Z + Z0)


@fig
def reflecto_resonance():
    f = np.linspace(150e6, 260e6, 2000)
    ax = Axes(330, 300, xlim=(150, 260), ylim=(0, 1.02), xlabel='frequency (MHz)', ylabel='|Γ|', title='reflection magnitude', mr=12)
    ax2 = Axes(330, 300, xlim=(150, 260), ylim=(-180, 180), xlabel='frequency (MHz)', ylabel='phase of Γ (°)', title='reflection phase',
               yticks=[-180, -90, 0, 90, 180], mr=12)
    for R, cls, lab in [(12e3, 't2', 'R = 12 kΩ'), (48e3, 't1', 'R = 48 kΩ (matched)'), (200e3, 't3', 'R = 200 kΩ'), (1e10, 't4', 'R → ∞ (blockade)')]:
        g = gamma(f, R)
        ax.line(f / 1e6, np.abs(g), cls, lab)
        ax2.line(f / 1e6, np.degrees(np.angle(g)), cls)
    print('f0 = %.1f MHz, Rmatch = %.1f kOhm' % (1 / (2 * np.pi * np.sqrt(L_TANK * CP_TANK)) / 1e6, L_TANK / (50 * CP_TANK) / 1e3))
    save('reflecto-resonance', row(ax.svg(legend_pos='bl'), ax2.svg()))


@fig
def reflecto_match():
    R = np.logspace(3.5, 6.5, 500)
    f0 = 1 / (2 * np.pi * np.sqrt(L_TANK * CP_TANK))
    g = gamma(f0, R)
    dG = np.abs(np.gradient(g, np.log(R)))
    ax = Axes(600, 300, xlim=(3.5, 6.5), ylim=(0, 1.4), xlabel='sensor resistance R (log scale)', ylabel='value at f₀', yticks=[0, 0.25, 0.5, 0.75, 1],
              xticks=[4, 5, 6], xfmt=lambda v: {4: '10 kΩ', 5: '100 kΩ', 6: '1 MΩ'}.get(int(round(v)), ''), mt=18)
    ax.line(np.log10(R), np.abs(g), 't1', '|Γ| at f₀')
    ax.line(np.log10(R), dG / dG.max(), 't2', 'response to a change in R, |dΓ/d ln R| (normalised)')
    ax.vline(np.log10(L_TANK / (50 * CP_TANK)), 'thin dash', 'R = L/(Z₀C_p) = 48 kΩ', ly=0.15)
    save('reflecto-match', ax.svg(legend_pos='tr'))


@fig
def snr_time():
    tmin = 0.055      # us, integration time for SNR = 1
    rng = np.random.default_rng(3)
    axs = []
    for t, title in [(0.2, '0.2 µs integration'), (2.0, '2 µs integration')]:
        snr = math.sqrt(t / tmin)
        sig = 1.0 / snr
        a = rng.normal(0, sig, 4000); b = rng.normal(1, sig, 4000)
        bins = np.linspace(-1.5, 2.5, 90)
        ha, _ = np.histogram(a, bins); hb, _ = np.histogram(b, bins)
        c = (bins[1:] + bins[:-1]) / 2
        ax = Axes(330, 280, xlim=(-1.5, 2.5), ylim=(0, max(ha.max(), hb.max()) * 1.45), xlabel='demodulated signal (norm.)', ylabel='counts',
                  title=f'{title}: SNR ≈ {snr:.1f}', mr=12, yticks=[])
        ax.area(c, np.zeros_like(c), ha, 'fa1'); ax.line(c, ha, 't1', 'N electrons')
        ax.area(c, np.zeros_like(c), hb, 'fa2'); ax.line(c, hb, 't2', 'N + 1')
        axs.append(ax.svg(legend_pos='tr'))
    save('snr-time', row(*axs))


@fig
def rf_power():
    P = np.linspace(-125, -80, 300)    # dBm at the device
    amp = 10 ** (P / 20)                # carrier amplitude (arb.)
    Pb = -100.0                         # power at which the RF excitation broadens the sensor peak
    broad = np.sqrt(1 + (10 ** ((P - Pb) / 20) / 1.0) ** 2)
    signal = amp / broad ** 1.6         # response per electron: grows with amplitude, drops once the peak broadens
    snr = signal / signal.max()
    ax = Axes(600, 290, xlim=(-125, -80), ylim=(0, 1.08), xlabel='carrier power at the tank (dBm)', ylabel='SNR per electron (normalised)', mt=18)
    ax.line(P, snr, 't1', 'measured SNR in a fixed integration time')
    ax.vline(P[np.argmax(snr)], 'thin dash', 'optimum', ly=1.02)
    ax.text(-124, 0.36, 'too weak: amplifier noise wins', 'sm')
    ax.text(-95, 0.38, 'too strong: the carrier', 'sm')
    ax.text(-95, 0.38, 'broadens the sensor peak', 'sm', dy=14)
    print('optimum power %.1f dBm' % P[np.argmax(snr)])
    save('rf-power', ax.svg(legend_pos='tl'))


# ---------------------------------------------------------------- chapter 11: double dot
ALPHA = np.array([[0.10, 0.03], [0.03, 0.10]])      # eV/V: rows = dots, columns = gates P1, P2


def dd_occupation(V1, V2, U=(6.5, 6.5), Um=0.9, A=ALPHA, V0=(0.450, 0.460), kT=0.004, Nmax=4):
    """Thermal charge occupation of a double dot in the constant-interaction model.
    V in volts, energies in meV. Returns <N1>, <N2>."""
    n1 = (A[0, 0] * (V1 - V0[0]) + A[0, 1] * (V2 - V0[1])) * 1000 / U[0]   # induced charge (in electrons) offsets
    n2 = (A[1, 0] * (V1 - V0[0]) + A[1, 1] * (V2 - V0[1])) * 1000 / U[1]
    states = [(i, j) for i in range(Nmax) for j in range(Nmax)]
    Es = []
    for (i, j) in states:
        # E = U1/2 * (i - 0.5 - n1)^2 ... shifted so that the 0->1 transition sits at n = 0
        E = U[0] * (0.5 * i * i - i * (0.5 + n1)) + U[1] * (0.5 * j * j - j * (0.5 + n2)) + Um * i * j
        Es.append(E)
    Es = np.array(Es)
    Emin = Es.min(axis=0)
    w = np.exp(-(Es - Emin) / kT)
    Z = w.sum(axis=0)
    N1 = sum(w[k] * states[k][0] for k in range(len(states))) / Z
    N2 = sum(w[k] * states[k][1] for k in range(len(states))) / Z
    return N1, N2


def csd_image(v1, v2, A=ALPHA, Um=0.9, sens=(0.10, 0.16), noise=0.0, kT=0.004):
    V1, V2 = np.meshgrid(v1, v2)
    N1, N2 = dd_occupation(V1, V2, A=A, Um=Um, kT=kT)
    sig = -(sens[0] * N1 + sens[1] * N2)
    if noise:
        sig = sig + RNG.normal(0, noise, sig.shape)
    d = np.hypot(np.gradient(sig, axis=1), np.gradient(sig, axis=0))
    return V1, V2, N1, N2, d


@fig
def csd_dqd():
    v1 = np.linspace(0.40, 0.66, 420); v2 = np.linspace(0.40, 0.66, 420)
    V1, V2, N1, N2, d = csd_image(v1, v2, noise=0.0)
    ax = Axes(460, 430, xlim=(400, 660), ylim=(400, 660), xlabel='P1 (mV)', ylabel='P2 (mV)', grid=False, ml=56, mt=18)
    ax.image(colormap(d[::-1], 'Greys', 0, np.percentile(d, 99.2)))
    for i in range(3):
        for j in range(3):
            m = (np.round(N1) == i) & (np.round(N2) == j)
            if m.sum() > 300:
                ax.text(V1[m].mean() * 1000, V2[m].mean() * 1000, f'({i},{j})', 'sm tacc', anchor='middle', dy=4)
    save('csd-dqd', ax.svg())


@fig
def virtual_gates():
    v = np.linspace(0.40, 0.62, 330)
    V1, V2, N1, N2, d = csd_image(v, v)
    ax1 = Axes(330, 330, xlim=(400, 620), ylim=(400, 620), xlabel='P1 (mV)', ylabel='P2 (mV)', grid=False, title='physical gates', mr=12)
    ax1.image(colormap(d[::-1], 'Greys', 0, np.percentile(d, 99)))
    # virtual gates vP1, vP2 defined so that each moves only its own dot: V = A^-1 * diag(a11, a22) * u
    Ainv = np.linalg.inv(ALPHA)
    M = Ainv @ np.diag([ALPHA[0, 0], ALPHA[1, 1]])
    u = np.linspace(0.40, 0.62, 330)
    U1, U2 = np.meshgrid(u, u)
    base = np.array([0.45, 0.46])
    P1 = base[0] + M[0, 0] * (U1 - base[0]) + M[0, 1] * (U2 - base[1])
    P2 = base[1] + M[1, 0] * (U1 - base[0]) + M[1, 1] * (U2 - base[1])
    N1v, N2v = dd_occupation(P1, P2)
    sig = -(0.10 * N1v + 0.16 * N2v)
    dv = np.hypot(np.gradient(sig, axis=1), np.gradient(sig, axis=0))
    ax2 = Axes(330, 330, xlim=(400, 620), ylim=(400, 620), xlabel='virtual gate vP1 (mV)', ylabel='virtual gate vP2 (mV)', grid=False,
               title='virtual gates', mr=12)
    ax2.image(colormap(dv[::-1], 'Greys', 0, np.percentile(dv, 99)))
    print('virtual gate matrix (physical = M @ virtual):', M)
    save('virtual-gates', row(ax1.svg(), ax2.svg()))


@fig
def last_electron():
    VP = np.linspace(200, 600, 2400)    # mV on P1 (virtual), other dot empty
    trans = [452, 517, 612]
    N = sum(sigmoid((VP - t) / 0.5) for t in trans)
    sig = -0.10 * N + RNG.normal(0, 0.004, VP.size)
    ax = Axes(600, 280, xlim=(200, 600), ylim=(-0.25, 0.03), xlabel='virtual plunger vP1 (mV)', ylabel='sensor signal (arb.)', yticks=[], mt=18)
    ax.line(VP, sig, 't1')
    ax.text(300, -0.03, 'N₁ = 0: no transitions over 250 mV', 'sm', anchor='middle')
    ax.text(485, -0.14, 'N₁ = 1', 'sm', anchor='middle')
    ax.text(560, -0.24, 'N₁ = 2', 'sm', anchor='middle', dy=0)
    ax.vline(452, 'thin dash', '0→1', ly=-0.2)
    ax.vline(517, 'thin dash', '1→2', ly=-0.2)
    save('last-electron', ax.svg())


# ---------------------------------------------------------------- chapter 12: tunnelling
@fig
def interdot_polarization():
    eps = np.linspace(-150, 150, 600)    # ueV detuning
    kT = KB_UEV * 0.05
    ax = Axes(600, 300, xlim=(-150, 150), ylim=(-0.03, 1.05), xlabel='detuning ε (µeV)', ylabel='probability electron is in dot 2', mt=18)
    for tc, cls in [(2, 't2'), (20, 't1'), (50, 't3')]:
        Om = np.sqrt(eps ** 2 + 4 * tc ** 2)
        P = 0.5 * (1 + eps / Om * np.tanh(Om / (2 * kT)))
        ax.line(eps, P, cls, f't_c = {tc} µeV')
    ax.text(-140, 0.3, '(1,0): electron in dot 1', 'sm')
    ax.text(55, 0.45, '(0,1): electron in dot 2', 'sm')
    save('interdot-polarization', ax.svg(legend_pos='tl'))


@fig
def pat_map():
    eps = np.linspace(-160, 160, 420)    # ueV
    f = np.linspace(1, 40, 300)          # GHz
    E, F = np.meshgrid(eps, f)
    tc = 20.0
    Om = np.sqrt(E ** 2 + 4 * tc ** 2) / H_UEV_GHZ          # GHz
    gam = 0.8
    resp = np.sign(E) * (gam ** 2 / ((F - Om) ** 2 + gam ** 2))
    base = np.tanh(E / 10.0) * 0.15
    Z = base + 0.85 * resp
    ax = Axes(460, 380, xlim=(-160, 160), ylim=(1, 40), xlabel='detuning ε (µeV)', ylabel='microwave frequency (GHz)', grid=False, ml=56, mt=18)
    ax.image(colormap(Z[::-1], 'RdBu', -1, 1))
    ax.line(eps, np.sqrt(eps ** 2 + 4 * tc ** 2) / H_UEV_GHZ, 'thin dash')
    ax.text(0, 2 * tc / H_UEV_GHZ - 2.5, '2t_c/h ≈ 9.7 GHz', 'sm', anchor='middle')
    save('pat-map', ax.svg())


@fig
def tunnel_rates():
    V = np.linspace(-40, 50, 300)       # mV of virtual barrier relative to a reference
    rate = 10 ** (3 + V / 29.5)         # one decade per ~30 mV (exp(V/12.8 mV))
    ax = Axes(330, 300, xlim=(-40, 50), ylim=(0, 7), xlabel='virtual barrier vB3 (mV, relative)', ylabel='log₁₀ tunnel rate (Hz)',
              title='dot–reservoir rate', mr=12, yticks=[0, 1, 2, 3, 4, 5, 6, 7])
    ax.area(np.array([-40, 50]), np.array([4, 4]), np.array([5, 5]), 'fa3')
    ax.text(-38, 4.35, 'Elzerman readout window', 'sm')
    ax.area(np.array([-40, 50]), np.array([0, 0]), np.array([2.0, 2.0]), 'fa2')
    ax.text(-38, 1.0, 'slower than a fast map: lines smear', 'sm')
    meas = np.arange(-35, 45, 7.0)
    ax.scatter(meas, 3 + meas / 29.5 + RNG.normal(0, 0.06, meas.size), 'f1', r=3.5, opacity=0.9)
    ax.line(V, np.log10(rate), 't1')
    V2 = np.linspace(-30, 30, 300)
    tc = 10 ** (1.0 + V2 / 23.3)   # ueV, e-fold per 10 mV
    ax2 = Axes(330, 300, xlim=(-30, 30), ylim=(-0.5, 2.5), xlabel='virtual barrier vB2 (mV, relative)', ylabel='log₁₀ t_c (µeV)',
               title='interdot tunnel coupling', mr=12, yticks=[0, 1, 2])
    ax2.line(V2, np.log10(tc), 't3')
    mt = np.arange(-26, 28, 6.0)
    ax2.scatter(mt, 1.0 + mt / 23.3 + RNG.normal(0, 0.05, mt.size), 'f3', r=3.5, opacity=0.9)
    save('tunnel-rates', row(ax.svg(), ax2.svg()))


@fig
def load_unload():
    t = np.linspace(0, 400, 800)       # us
    rng = np.random.default_rng(11)
    Gin, Gout = 1 / 45.0, 1 / 30.0     # per us
    P_load = 1 - np.exp(-t * Gin)
    P_unl = np.exp(-t * Gout)
    ax = Axes(600, 300, xlim=(0, 400), ylim=(-0.05, 1.45), xlabel='time after the pulse edge (µs)', ylabel='average occupation of dot 2', mt=18, yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.line(t, P_load + rng.normal(0, 0.012, t.size), 't1', 'after pulsing the level below E_F (loading)')
    ax.line(t, P_unl + rng.normal(0, 0.012, t.size), 't2', 'after pulsing it above E_F (unloading)')
    ax.text(150, 0.62, 'fit: Γ_in = 1/45 µs ≈ 22 kHz', 'sm')
    ax.text(150, 0.25, 'fit: Γ_out = 1/30 µs ≈ 33 kHz', 'sm')
    save('load-unload', ax.svg(legend_pos='tr'))



# ---------------------------------------------------------------- chapter 13: pulses
def _highpass(x, dt, tau):
    """First-order RC high-pass (bias tee seen from the device side)."""
    a = tau / (tau + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = a * (y[i - 1] + x[i] - x[i - 1])
    return y


@fig
def bias_tee_droop():
    dt = 0.05                     # ms
    t = np.arange(0, 1000, dt)    # ms
    ph = t % 10.0
    # 10 ms cycle: load 4 ms (+10 mV), manipulate 1 ms (+20 mV), read 4 ms (+5 mV), empty 1 ms (0)
    levels = [(0.0, 4.0, 10.0, 'load'), (4.0, 5.0, 20.0, 'manipulate'), (5.0, 9.0, 5.0, 'read'), (9.0, 10.0, 0.0, 'empty')]
    uni = np.select([ph < 4, ph < 5, ph < 9], [10.0, 20.0, 5.0], 0.0)
    comp = uni - 7.5                                    # same pulses, shifted so the cycle averages to zero
    tau = 200.0                   # ms (2 MOhm x 100 nF)
    cls = ['t1', 't3', 't2', 't4']
    out = []
    for seq, title in [(uni, 'unipolar sequence'), (comp, 'zero-average sequence')]:
        y = _highpass(np.concatenate([[0.0], seq]), dt, tau)[1:]
        ax = Axes(330, 300, xlim=(0, 1000), ylim=(-12, 22), xlabel='time (ms)', ylabel='level at the gate (mV)',
                  title=title, mr=12, xticks=[0, 250, 500, 750, 1000])
        for (a0, a1, v, name), c in zip(levels, cls):
            idx = np.where((ph > a0 + 0.5 * (a1 - a0) - dt / 2) & (ph <= a0 + 0.5 * (a1 - a0) + dt / 2))[0]
            ax.line(t[idx], y[idx], c, name if seq is uni else None)
        ax.hline(0, 'thin dash')
        out.append(ax)
    out[0].text(330, -10.5, 'all levels sink by 7.5 mV', 'sm')
    out[1].text(300, -10.5, 'levels stay where they were set', 'sm')
    save('bias-tee-droop', row(out[0].svg(legend_pos='tr'), out[1].svg()))


def _line_response(t):
    """Step response of a realistic baseband line (ns): fast edge, a 4% 25-ns undershoot, 3% 600-ns droop."""
    r = 1 - np.exp(-t / 0.8)
    r += 0.04 * np.exp(-t / 25.0) * (t > 0) * (1 - np.exp(-t / 2.0))
    r -= 0.03 * (1 - np.exp(-t / 600.0))
    return np.where(t > 0, r, 0)


@fig
def step_response():
    t = np.linspace(-20, 1000, 4000)
    raw = _line_response(t)
    fixed = np.where(t > 0, 1 - np.exp(-t / 0.8), 0) + RNG.normal(0, 0.0015, t.size) * (t > 0)
    ax = Axes(330, 300, xlim=(-20, 1000), ylim=(0.9, 1.06), xlabel='time after the step (ns)', ylabel='step at the gate (normalized)',
              title='what the gate actually sees', mr=12, yticks=[0.9, 0.95, 1.0, 1.05])
    ax.hline(1, 'thin dash')
    ax.line(t, raw, 't2', 'without correction')
    ax.line(t, fixed, 't1', 'pre-distorted')
    # in-situ map: charge transition position after a step of 2 mV on vP1
    tau = np.linspace(4, 1000, 160)
    off = np.linspace(-0.25, 0.25, 140)   # mV of DC offset relative to the nominal transition
    T, O = np.meshgrid(tau, off)
    shift = 2.0 * (_line_response(T + 1e-9) - 1)  # mV the gate is short of the target
    occ = sigmoid((O + shift) / 0.012)
    ax2 = Axes(330, 300, xlim=(0, 1000), ylim=(-0.25, 0.25), xlabel='delay after the step (ns)', ylabel='DC offset (mV)',
               title='measured in situ at a charge line', mr=12, grid=False)
    ax2.image(colormap(occ[::-1], 'viridis', 0, 1))
    ax2.line(tau, -2.0 * (_line_response(tau + 1e-9) - 1), 'thin dash', extra='style="stroke:#fff"')
    ax2.text(120, 0.17, 'charge line follows the settling step', 'sm', style='fill:#fff')
    save('step-response', row(ax.svg(legend_pos='br'), ax2.svg()))


@fig
def pulse_amplitude_map():
    V = np.linspace(440, 480, 400)     # DC on vP1 (mV)
    A = np.linspace(0, 30, 240)        # programmed pulse amplitude at the AWG (mV)
    VV, AA = np.meshgrid(V, A)
    att = 0.5                          # fraction arriving at the gate (line + divider)
    v0 = 470.0
    w = 0.25
    # square wave: half the time at V, half at V + att*A; electron follows (fast tunnelling)
    occ = 0.5 * sigmoid((VV - v0) / w) + 0.5 * sigmoid((VV + att * AA - v0) / w)
    sig = np.gradient(occ, axis=1)
    ax = Axes(560, 330, xlim=(440, 480), ylim=(0, 30), xlabel='DC on virtual plunger vP1 (mV)', ylabel='programmed pulse amplitude (mV)',
              grid=False, mt=18)
    ax.image(colormap(sig[::-1], 'magma', 0, sig.max()))
    ax.text(462, 26, 'slope: 0.5 of the programmed', 'sm', style='fill:#fff')
    ax.text(462, 23.5, 'amplitude reaches the gate', 'sm', style='fill:#fff')
    ax.text(471.5, 6, 'DC level at', 'sm', style='fill:#fff')
    ax.text(471.5, 3.5, 'the transition', 'sm', style='fill:#fff')
    ax.text(441, 26, 'pulse top at the', 'sm', style='fill:#fff')
    ax.text(441, 23.5, 'transition', 'sm', style='fill:#fff')
    save('pulse-amplitude-map', ax.svg())



# ---------------------------------------------------------------- chapter 14: spins in a field
GMUB = 2 * MUB_UEV    # ueV/T for g = 2 (115.8)


@fig
def spin_valley_levels():
    B = np.linspace(0, 1.5, 600)
    Evs, D = 120.0, 2.5        # valley splitting of dot 2, spin-valley coupling (exaggerated)
    Ez = GMUB * B
    a = -Ez / 2                # v-, down
    b0 = Ez / 2                # v-, up
    c0 = Evs - Ez / 2          # v+, down
    d = Evs + Ez / 2           # v+, up
    mid = (b0 + c0) / 2
    half = np.sqrt(((b0 - c0) / 2) ** 2 + D ** 2)
    lo, hi = mid - half, mid + half
    ax = Axes(600, 340, xlim=(0, 1.5), ylim=(-100, 220), xlabel='total magnetic field at the dot (T)', ylabel='energy (µeV)', mt=18)
    ax.line(B, a, 't1')
    ax.line(B, lo, 't2')
    ax.line(B, hi, 't3')
    ax.line(B, d, 't4')
    ax.vline(0.56, 'thin dash', 'operating point', ly=200)
    ax.vline(Evs / GMUB, 'thin dash', 'hotspot E_Z = E_VS', ly=200)
    ax.text(0.03, -50, 'valley −, spin ↓ (ground state)', 'sm')
    ax.text(0.03, 42, 'valley −, spin ↑ (qubit)', 'sm')
    ax.text(0.03, 72, 'valley +, spin ↓', 'sm')
    ax.text(0.03, 160, 'valley +, spin ↑', 'sm')
    ax.text(1.12, 30, 'states swap', 'sm')
    ax.text(1.12, 16, 'character here', 'sm')
    save('spin-valley-levels', ax.svg())


@fig
def t1_field():
    B = np.linspace(0.15, 2.0, 800)
    Ez = GMUB * B
    Evs, D = 120.0, 0.5
    dlt = Ez - Evs
    mix = 0.5 * (1 - np.abs(dlt) / np.sqrt(dlt ** 2 + D ** 2))
    rate = 1.0 + 560 * B ** 5 + 20 * B ** 7 + 1e6 * mix * (Ez / Evs) ** 5
    ax = Axes(600, 320, xlim=(0.15, 2.0), ylim=(-5, 1), xlabel='total magnetic field at the dot (T)', ylabel='log₁₀ T₁ (s)', mt=18,
              yticks=[-5, -4, -3, -2, -1, 0, 1])
    ax.line(B, -np.log10(rate), 't1', 'model: phonons + spin–valley mixing')
    Bm = np.array([0.25, 0.35, 0.45, 0.56, 0.7, 0.85, 0.95, 1.0, 1.02, 1.04, 1.06, 1.1, 1.2, 1.4, 1.6, 1.8])
    Ezm = GMUB * Bm
    dm = Ezm - Evs
    mm = 0.5 * (1 - np.abs(dm) / np.sqrt(dm ** 2 + D ** 2))
    rm = 1.0 + 560 * Bm ** 5 + 20 * Bm ** 7 + 1e6 * mm * (Ezm / Evs) ** 5
    ax.scatter(Bm, -np.log10(rm) + RNG.normal(0, 0.08, Bm.size), 'f1', r=3.5, opacity=0.9, label='measured')
    ax.vline(0.56, 'thin dash', 'operating point: 30 ms', ly=0.6)
    ax.text(1.1, -4.6, 'hotspot (E_Z = E_VS)', 'sm')
    ax.text(1.45, -2.7, '∝ B⁻⁵ … B⁻⁷', 'sm')
    save('t1-field', ax.svg(legend_pos='tr'))


@fig
def field_sweep():
    Bx = np.linspace(0, 1.0, 400)
    gmu = 27.99      # GHz/T for g = 2
    out = []
    for Bm, cls, name in [(0.1509, 't1', 'dot 1'), (0.1616, 't2', 'dot 2')]:
        Bmm = Bm * np.tanh(Bx / 0.08)        # magnet saturates above ~0.2 T
        out.append((gmu * (Bx + Bmm), cls, name, Bm))
    ax = Axes(600, 320, xlim=(0, 1.0), ylim=(0, 35), xlabel='external field B_ext (T)', ylabel='qubit frequency (GHz)', mt=18)
    for f, cls, name, Bm in out:
        ax.line(Bx, f, cls, name)
        ax.line(Bx, gmu * (Bx + Bm), 'thin dash')
    bpts = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.scatter(bpts, gmu * (bpts + 0.1509), 'f1', r=3.5, opacity=0.9)
    ax.scatter(bpts, gmu * (bpts + 0.1616), 'f2', r=3.5, opacity=0.9)
    ax.text(0.55, 9.5, 'intercepts: magnet adds', 'sm')
    ax.text(0.55, 7.3, '151 mT (dot 1), 162 mT (dot 2)', 'sm')
    ax.text(0.55, 3.5, 'slope 27.99 GHz/T → g = 2.000', 'sm')
    ax.text(0.1, 1.5, '← magnet not yet saturated', 'sm')
    save('field-sweep', ax.svg(legend_pos='tl'))



# ---------------------------------------------------------------- chapter 15: Elzerman readout
EZ_UEV, KT_UEV, G0_RO = 64.0, KB_UEV * 0.05, 30e3     # Zeeman energy, kT at 50 mK, bare tunnel rate (Hz)


def _ro_fid(D, tR, tmin=0.5e-6, T1=30e-3, Ez=EZ_UEV, kT=KT_UEV, G0=G0_RO):
    f = lambda E: 1 / (1 + np.exp(E / kT))
    Gu = G0 * (1 - f(Ez - D))
    Gin = G0 * f(-D)
    Fu = Gu / (Gu + 1 / T1) * (1 - np.exp(-(Gu + 1 / T1) * tR)) * np.exp(-Gin * tmin)
    Fd = np.exp(-G0 * tR * f(D))
    return Fu, Fd


@fig
def elzerman_map():
    from scipy.linalg import expm
    t = np.linspace(0, 300e-6, 200)
    Ds = np.linspace(-30, 100, 160)
    f = lambda E: 1 / (1 + np.exp(E / KT_UEV))
    Z = np.zeros((Ds.size, t.size))
    for i, D in enumerate(Ds):
        Eu, Ed = EZ_UEV - D, -D
        # states: down, up, empty
        Q = np.zeros((3, 3))
        Q[2, 0] = G0_RO * (1 - f(Ed)); Q[0, 2] = G0_RO * f(Ed)
        Q[2, 1] = G0_RO * (1 - f(Eu)); Q[1, 2] = G0_RO * f(Eu)
        Q -= np.diag(Q.sum(axis=0))
        p0 = np.array([0.5, 0.5, 0.0])
        dt = t[1] - t[0]
        M = expm(Q * dt)
        p = p0.copy()
        for j in range(t.size):
            Z[i, j] = p[2]
            p = M @ p
    ax = Axes(560, 360, xlim=(0, 300), ylim=(-30, 100), xlabel='time in the read stage (µs)', ylabel='spin-down level below E_F, Δ (µeV)',
              grid=False, mt=18, ml=62)
    ax.image(colormap(Z[::-1], 'magma', 0, 0.35))
    ax.hline(0, 'thin dash')
    ax.hline(EZ_UEV, 'thin dash')
    ax.text(60, -18, 'both levels above E_F: the dot empties and stays empty', 'sm', style='fill:#000')
    ax.text(60, 85, 'both levels below E_F: nothing leaves', 'sm', style='fill:#fff')
    ax.text(60, 36, 'spin-up electrons leave, a spin-down one enters', 'sm', style='fill:#fff')
    ax.text(60, 26, '→ a transient "spin signal" only in this window', 'sm', style='fill:#fff')
    save('elzerman-map', ax.svg())


@fig
def elzerman_traces():
    rng = np.random.default_rng(5)
    t = np.arange(0, 600, 2.0)       # us, 2 us integration per point (SNR 6)
    sig = 1 / 6.0
    def trace(blips):
        y = np.zeros_like(t)
        for a, b in blips:
            y[(t >= a) & (t < b)] = 1.0
        return y + rng.normal(0, sig, t.size)
    rows = [([(34, 82)], 't2', 'spin up: tunnels out at 34 µs, a spin-down electron returns at 82 µs'),
            ([], 't1', 'spin down: no event'),
            ([(262, 276)], 't2', 'spin up, late and short event: still above threshold'),
            ([(120, 121.6)], 't3', 'spin up, event too short for the bandwidth: read as down')]
    out = []
    for k, (bl, cls, title) in enumerate(rows):
        y = trace(bl)
        if k == 3:
            y = np.zeros_like(t) + rng.normal(0, sig, t.size)
            y[np.argmin(abs(t - 120))] += 0.45
        ax = Axes(860, 136 if k == 3 else 120, xlim=(0, 600), ylim=(-0.6, 1.6), xlabel='time in the read stage (µs)' if k == 3 else '', ylabel='', title=title,
                  yticks=[0, 1], mt=22, mb=48 if k == 3 else 14, ml=58, xticks=[0, 100, 200, 300, 400, 500, 600] if k == 3 else [])
        ax.line(t, y, cls)
        ax.hline(0.5, 'thin dash', 'threshold', lx=560)
        out.append(ax.svg())

    save('elzerman-traces', column(*out))


@fig
def readout_tradeoff():
    D = np.linspace(10, 70, 300)
    Fu, Fd = _ro_fid(D, 500e-6)
    ax = Axes(330, 300, xlim=(10, 70), ylim=(80, 100), xlabel='spin-down level below E_F, Δ (µeV)', ylabel='fidelity (%)',
              title='choosing the read level', mr=12)
    ax.line(D, 100 * Fd, 't1', 'F↓')
    ax.line(D, 100 * Fu, 't2', 'F↑')
    ax.line(D, 100 * (Fu + Fd - 1), 't3', 'visibility')
    ax.vline(40, 'thin dash', 'Δ* = 40 µeV', ly=91)
    tR = np.linspace(20e-6, 3e-3, 300)
    Fu2, Fd2 = _ro_fid(40, tR)
    ax2 = Axes(330, 300, xlim=(0, 3000), ylim=(80, 100), xlabel='read window t_R (µs)', ylabel='fidelity (%)',
               title='choosing the read window', mr=12, xticks=[0, 1000, 2000])
    ax2.line(tR * 1e6, 100 * Fd2, 't1', 'F↓')
    ax2.line(tR * 1e6, 100 * Fu2, 't2', 'F↑')
    ax2.line(tR * 1e6, 100 * (Fu2 + Fd2 - 1), 't3', 'visibility')
    ax2.vline(500, 'thin dash', '500 µs', ly=82)
    save('readout-tradeoff', row(ax.svg(legend_pos='br'), ax2.svg(legend_pos='br')))


@fig
def t1_decay():
    tw = np.array([0.1, 1, 2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 130])   # ms
    V, a = 0.986, 0.003
    T1 = 30.0
    rng = np.random.default_rng(3)
    n = 2000
    P = a + 0.5 * V * np.exp(-tw / T1)
    Pm = rng.binomial(n, P) / n
    tt = np.linspace(0, 140, 300)
    ax = Axes(600, 300, xlim=(0, 140), ylim=(0, 0.55), xlabel='wait time before readout (ms)', ylabel='spin-up fraction', mt=18)
    ax.line(tt, a + 0.5 * V * np.exp(-tt / T1), 't1', 'fit: T₁ = 30 ± 2 ms')
    ax.scatter(tw, Pm, 'f1', r=3.5, opacity=0.9)
    ax.text(70, 0.30, 'random spin loaded (½ up), then wait, then read', 'sm')
    save('t1-decay', ax.svg(legend_pos='tr'))



# ---------------------------------------------------------------- chapter 16: Pauli spin blockade
@fig
def psb_levels():
    eps = np.linspace(-150, 250, 800)     # ueV, positive towards (0,2)
    tc, Ez, Est = 20.0, 64.0, 120.0
    # singlet branch: S(1,1) at 0, S(0,2) at -eps, coupled by sqrt(2) tc
    mid = -eps / 2
    half = np.sqrt((eps / 2) ** 2 + 2 * tc ** 2)
    Slo, Shi = mid - half, mid + half
    ax = Axes(600, 340, xlim=(-150, 250), ylim=(-260, 160), xlabel='detuning ε (µeV), towards (0,2) →', ylabel='energy (µeV)', mt=18)
    ax.area(np.array([0, Est]), np.array([-260, -260]), np.array([160, 160]), 'fa3')
    ax.line(eps, Slo, 't1', 'singlet branch')
    ax.line(eps, Shi, 'thin dash')
    for E, lab in [(Ez, 'T₊(1,1) = ↑↑'), (0.0, 'T₀(1,1)'), (-Ez, 'T₋(1,1) = ↓↓')]:
        ax.line(eps, np.full_like(eps, E), 't2')
        ax.text(-145, E + 6, lab, 'sm')
    ax.line(eps, -eps + Est, 't4', 'triplets (0,2)')
    ax.text(8, 146, 'readout window', 'sm')
    ax.text(8, 124, '0 < ε < E_ST', 'sm')
    ax.text(150, -215, 'S(0,2)', 'sm')
    ax.text(-145, -20, 'S(1,1)', 'sm')
    save('psb-levels', ax.svg(legend_pos='bl'))


@fig
def psb_csd():
    x = np.linspace(-6, 14, 400)   # vP1 (mV, relative)
    y = np.linspace(-6, 16, 440)   # vP2
    X, Y = np.meshgrid(x, y)
    s, E, w = 9.0, 1.2, 0.12
    sg = lambda z: 1 / (1 + np.exp(-z / w))
    eps = Y - X                                   # >0 towards (0,2)
    inner = sg(eps) * sg(Y) * sg(s - X)           # in the (0,2) region next to the segment
    # charge-state map (approximate): base levels for each region
    n02 = sg(eps) * sg(Y) * sg(s - X)
    n01 = sg(-Y) * sg(-X) * sg(Y - X + 50)
    n12 = sg(X - s) * sg(Y - s)
    n11 = 1 - np.clip(n02 + n01 + n12, 0, 1)
    sig_S = 0.8 * n02 + 0.3 * n01 + 1.0 * n12 + 0.5 * n11
    strip = n02 * sg(E - eps)
    sig_T = sig_S - 0.3 * strip                   # blockaded triplets keep the (1,1) signal in the strip
    ax = Axes(330, 340, xlim=(-6, 14), ylim=(-6, 16), xlabel='vP1 (mV, relative)', ylabel='vP2 (mV, relative)', title='after preparing a singlet', grid=False, mr=12)
    ax.image(colormap(sig_S[::-1], 'viridis', 0.2, 1.05))
    ax.text(-5, 12, '(0,2)', 'sm', style='fill:#fff'); ax.text(8, -4, '(1,1)', 'sm', style='fill:#fff')
    ax.text(-5, -4, '(0,1)', 'sm', style='fill:#fff'); ax.text(10.5, 13, '(1,2)', 'sm', style='fill:#000')
    ax2 = Axes(330, 340, xlim=(-6, 14), ylim=(-6, 16), xlabel='vP1 (mV, relative)', ylabel='vP2 (mV, relative)', title='triplet minus singlet', grid=False, mr=12)
    ax2.image(colormap((sig_S - sig_T)[::-1], 'magma', 0, 0.32))
    for xs, ys in [([-6, 0], [0, 0]), ([0, 0], [-6, 0]), ([0, s], [0, s]), ([s, s], [s, 16]), ([s, 14], [s, s])]:
        ax2.line(np.array(xs), np.array(ys), 'thin', extra='style="stroke:#999"')
    ax2.text(-5, 12, '(0,2)', 'sm', style='fill:#fff'); ax2.text(8, -4, '(1,1)', 'sm', style='fill:#fff')
    ax2.text(-5, -4, '(0,1)', 'sm', style='fill:#fff'); ax2.text(10.5, 13, '(1,2)', 'sm', style='fill:#fff')
    ax2.text(-3.5, 6.5, 'blockade strip,', 'sm', style='fill:#fff')
    ax2.text(-3.5, 5.2, '1.2 mV wide', 'sm', style='fill:#fff')
    save('psb-csd', row(ax.svg(), ax2.svg()))


@fig
def psb_decay():
    t = np.logspace(-8, -1, 500)    # s
    ax = Axes(600, 320, xlim=(-8, -1), ylim=(-0.03, 1.05), xlabel='time at the readout point (log₁₀ s)', ylabel='probability still blockaded in (1,1)', mt=18,
              xticks=[-8, -7, -6, -5, -4, -3, -2, -1])
    lt = np.log10(t)
    ax.area(np.log10(np.array([3e-6, 30e-6])), np.array([-0.03, -0.03]), np.array([1.05, 1.05]), 'fa3')
    ax.line(lt, np.exp(-t / 3e-3), 't2', '↑↑ and ↓↓ (T₊, T₋): spin flip needed, τ ≈ 3 ms')
    ax.line(lt, np.exp(-t / 1e-6), 't1', 'antiparallel, T₀-like, ΔE_Z/h = 300 MHz: τ ≈ 1 µs')
    ax.line(lt, np.exp(-t / 2e-4), 't4', 'antiparallel, T₀-like, ΔE_Z/h = 20 MHz: τ ≈ 200 µs')
    ax.line(lt, np.exp(-t / 2e-8), 't3', 'singlet: tunnels to (0,2) within ~10 ns')
    ax.text(-5.45, 0.52, 'integration', 'sm')
    ax.text(-5.45, 0.45, 'window', 'sm')
    save('psb-decay', ax.svg(legend_pos='bl'))



# ---------------------------------------------------------------- chapter 17: readout fidelity
def _psb_shots(n, t_int, rng, tau_even=3e-3, tau_odd=1e-6, snr2=2.3):
    """Integrated blockade signal (1 = blocked) for n even and n odd shots."""
    sig = 1.0 / (snr2 * np.sqrt(t_int / 2e-6))
    out = []
    for tau in (tau_even, tau_odd):
        td = rng.exponential(tau, n)
        frac = np.minimum(td, t_int) / t_int
        out.append(frac + rng.normal(0, sig, n))
    return out


def _best_threshold(ev, od):
    th = np.linspace(-0.5, 1.5, 801)
    Fe = (ev[None, :] > th[:, None]).mean(axis=1)
    Fo = (od[None, :] <= th[:, None]).mean(axis=1)
    k = np.argmax(Fe + Fo)
    return th[k], Fe[k], Fo[k]


@fig
def readout_histograms():
    rng = np.random.default_rng(21)
    svgs = []
    for t_int, title in [(2e-6, '2 µs: noise dominates'), (20e-6, '20 µs: our choice'), (1e-3, '1 ms: decay tail grows')]:
        ev, od = _psb_shots(40000, t_int, rng)
        bins = np.linspace(-0.8, 1.8, 90)
        he, _ = np.histogram(ev, bins, density=True)
        ho, _ = np.histogram(od, bins, density=True)
        c = 0.5 * (bins[1:] + bins[:-1])
        th, Fe, Fo = _best_threshold(ev, od)
        ymax = max(he.max(), ho.max()) * 1.15
        ax = Axes(280, 260, xlim=(-0.8, 1.8), ylim=(0, ymax), xlabel='integrated signal', ylabel='probability density' if t_int < 3e-6 else '',
                  title=title, mr=10, ml=46 if t_int < 3e-6 else 22, yticks=[], xticks=[0, 1])
        ax.area(c, np.zeros_like(c), ho, 'fa1')
        ax.line(c, ho, 't1', 'odd (not blocked)' if t_int > 5e-4 else None)
        ax.area(c, np.zeros_like(c), he, 'fa2')
        ax.line(c, he, 't2', 'even (blocked)' if t_int > 5e-4 else None)
        ax.vline(th, 'thin dash')
        ax.text(1.75, ymax * 0.9, f'F = {100 * (Fe + Fo) / 2:.1f}%', 'sm', anchor='end')
        svgs.append(ax.svg(legend_pos='tl' if t_int > 5e-4 else 'tr'))
    save('readout-histograms', row(*svgs, gap=4))


@fig
def readout_fid_time():
    rng = np.random.default_rng(22)
    ts = np.logspace(np.log10(0.5e-6), np.log10(1e-3), 26)
    Fm, Fe_l, Fo_l = [], [], []
    for t in ts:
        ev, od = _psb_shots(30000, t, rng)
        th, Fe, Fo = _best_threshold(ev, od)
        Fm.append((Fe + Fo) / 2); Fe_l.append(Fe); Fo_l.append(Fo)
    lt = np.log10(ts * 1e6)
    ax = Axes(600, 320, xlim=(np.log10(0.5), 3), ylim=(80, 100), xlabel='integration time (log₁₀ µs)', ylabel='fidelity (%)', mt=18,
              xticks=[0, 1, 2, 3], xfmt=lambda v: {0: '1', 1: '10', 2: '100', 3: '1000'}.get(round(v), ''))
    ax.line(lt, 100 * np.array(Fo_l), 't1', 'odd (antiparallel) states')
    ax.line(lt, 100 * np.array(Fe_l), 't2', 'even (parallel) states')
    ax.line(lt, 100 * np.array(Fm), 't3', 'average')
    ax.vline(np.log10(20), 'thin dash', '20 µs', ly=81.5)
    ax.text(0.0, 97.3, 'noise-limited', 'sm')
    ax.text(2.2, 97.3, 'decay-limited', 'sm')
    save('readout-fid-time', ax.svg(legend_pos='br'))



# ---------------------------------------------------------------- chapter 18: finding the resonance
F1_GHZ, F2_GHZ, LO_GHZ = 15.420, 15.720, 15.500


def _burst_resp(f, f0, fr):
    """Time-averaged spin flip probability after a long burst (MHz units for fr, GHz for f)."""
    d = (f - f0) * 1e3
    return 0.5 * fr ** 2 / (fr ** 2 + d ** 2)


@fig
def resonance_search():
    rng = np.random.default_rng(31)
    f = np.arange(15.20, 15.90, 0.002)
    P = _burst_resp(f, F1_GHZ, 5.0) + _burst_resp(f, F2_GHZ, 4.0)
    P += _burst_resp(f, 2 * LO_GHZ - F1_GHZ, 0.18) + _burst_resp(f, 2 * LO_GHZ - F2_GHZ, 0.14)
    Pm = 0.003 + 0.973 * np.clip(P, 0, 0.5)
    Pm = rng.binomial(200, Pm) / 200
    ax = Axes(600, 320, xlim=(15.2, 15.9), ylim=(-0.02, 0.6), xlabel='programmed microwave frequency (GHz)', ylabel='spin-up fraction after a 1 ms burst', mt=18)
    ax.line(f, Pm, 't1')
    ax.text(F1_GHZ - 0.07, 0.53, 'qubit 1', 'sm'); ax.text(F2_GHZ + 0.012, 0.53, 'qubit 2', 'sm')
    ax.text(2 * LO_GHZ - F1_GHZ + 0.008, 0.2, 'image of qubit 1', 'sm')
    ax.text(2 * LO_GHZ - F2_GHZ + 0.008, 0.2, 'image of qubit 2', 'sm')
    ax.vline(LO_GHZ, 'thin dash', 'LO', ly=0.57)
    save('resonance-search', ax.svg())


@fig
def line_power():
    rng = np.random.default_rng(32)
    d = np.linspace(-1.5, 1.5, 301)     # MHz detuning
    ax = Axes(600, 300, xlim=(-1.5, 1.5), ylim=(-0.02, 0.8), xlabel='detuning from qubit 1 (MHz)', ylabel='spin-up fraction', mt=18)
    sf = np.sqrt(2) / (2 * np.pi * 6.0)   # MHz, Gaussian sigma for T2* = 6 us
    for fr, cls, lab in [(1.0, 't2', 'f_R = 1 MHz: power broadened'), (0.25, 't3', 'f_R = 250 kHz'), (0.02, 't1', 'f_R = 20 kHz, 5 ms burst: T₂*-limited')]:
        # average over quasi-static detuning noise
        dd = d[:, None] + rng.normal(0, sf, (1, 4000))
        P = (0.5 * fr ** 2 / (fr ** 2 + dd ** 2)).mean(axis=1)
        if fr < 0.05:
            P = P / P.max() * 0.5
        Pm = 0.003 + 0.973 * P
        Pm = rng.binomial(400, Pm) / 400
        ax.line(d, Pm, cls, lab)
    save('line-power', ax.svg(legend_pos='tr'))


@fig
def line_field():
    B = np.linspace(0.390, 0.410, 50)
    gmu = 27.99
    q1 = F1_GHZ + gmu * (B - 0.4)
    q2 = F2_GHZ + gmu * (B - 0.4)
    ax = Axes(600, 300, xlim=(0.390, 0.410), ylim=(15.0, 16.0), xlabel='external field (T)', ylabel='programmed frequency of the line (GHz)', mt=18,
              xticks=[0.390, 0.395, 0.400, 0.405, 0.410])
    ax.line(B, q1, 't1', 'qubit 1: +28 GHz/T')
    ax.line(B, q2, 't2', 'qubit 2: +28 GHz/T')
    ax.line(B, 2 * LO_GHZ - q1, 't1', None, extra='stroke-dasharray="5 4"')
    ax.line(B, 2 * LO_GHZ - q2, 't2', None, extra='stroke-dasharray="5 4"')
    ax.text(0.3905, 15.95, 'dashed, images: move the wrong way (−28 GHz/T)', 'sm')
    ax.hline(LO_GHZ, 'thin dash', 'LO (fixed)', lx=0.3955)
    save('line-field', ax.svg(legend_pos='br'))



# ---------------------------------------------------------------- chapter 19: Rabi
VIS_E, OFF_E = 0.973, 0.003      # Elzerman visibility and offset (chapter 17)


@fig
def rabi_chevron():
    rng = np.random.default_rng(41)
    d = np.linspace(-12, 12, 121)          # MHz
    t = np.linspace(0, 1.5, 151)           # us
    D, T = np.meshgrid(d, t)
    fr = 4.0
    Om = np.sqrt(fr ** 2 + D ** 2)
    P = fr ** 2 / Om ** 2 * np.sin(np.pi * Om * T) ** 2
    Pm = rng.binomial(150, OFF_E + VIS_E * P) / 150
    ax = Axes(600, 360, xlim=(-12, 12), ylim=(0, 1.5), xlabel='detuning from 15.420 GHz (MHz)', ylabel='burst duration (µs)', grid=False, mt=18)
    ax.image(colormap(Pm[::-1], 'viridis', 0, 1))
    ax.text(-11.5, 0.06, 'π pulse: 125 ns at f_R = 4 MHz', 'sm', style='fill:#fff')
    ax2 = Axes(600, 220, xlim=(0, 1.5), ylim=(-0.03, 1.03), xlabel='burst duration (µs)', ylabel='spin-up fraction', mt=18, yticks=[0, 0.5, 1])
    tt = np.linspace(0, 1.5, 400)
    ax2.line(tt, OFF_E + VIS_E * np.sin(np.pi * fr * tt) ** 2, 't1', 'on resonance')
    ax2.line(tt, OFF_E + VIS_E * (fr ** 2 / (fr ** 2 + 36)) * np.sin(np.pi * np.sqrt(fr ** 2 + 36) * tt) ** 2, 't2', '6 MHz off resonance')
    save('rabi-chevron', column(ax.svg(), ax2.svg(legend_pos='tr')))


@fig
def rabi_decay():
    rng = np.random.default_rng(42)
    fr, tau = 4.0, 40.0
    out = []
    for t0, title in [(0, 'first 2 µs'), (30, 'after 30 µs of driving')]:
        t = np.linspace(t0, t0 + 2, 201)
        P = 0.5 - 0.5 * np.exp(-t / tau) * np.cos(2 * np.pi * fr * t)
        Pm = rng.binomial(200, OFF_E + VIS_E * P) / 200
        ax = Axes(330, 260, xlim=(t0, t0 + 2), ylim=(-0.03, 1.03), xlabel='burst duration (µs)', ylabel='spin-up fraction', title=title, mr=12, yticks=[0, 0.5, 1])
        ax.scatter(t, Pm, 'f1', r=2.2, opacity=0.8)
        tt = np.linspace(t0, t0 + 2, 600)
        ax.line(tt, OFF_E + VIS_E * (0.5 - 0.5 * np.exp(-tt / tau) * np.cos(2 * np.pi * fr * tt)), 't1')
        out.append(ax.svg())
    save('rabi-decay', row(*out))


@fig
def rabi_amplitude():
    rng = np.random.default_rng(43)
    A = np.linspace(0, 1.0, 11)              # normalized drive amplitude (fraction of max)
    lin = 16.0 * A                           # MHz if perfectly linear
    meas = lin * (1 - 0.18 * np.clip(A - 0.6, 0, None) ** 2 / 0.16)
    meas = meas + rng.normal(0, 0.12, A.size)
    meas[0] = 0
    ax = Axes(600, 300, xlim=(0, 1.0), ylim=(0, 18), xlabel='microwave amplitude at the AWG (fraction of full scale)', ylabel='Rabi frequency f_R (MHz)', mt=18)
    ax.line(A, lin, 'thin dash')
    ax.scatter(A, meas, 'f1', r=4, opacity=0.95)
    ax.line(np.linspace(0, 1, 100), 16 * np.linspace(0, 1, 100) * (1 - 0.18 * np.clip(np.linspace(0, 1, 100) - 0.6, 0, None) ** 2 / 0.16), 't1')
    ax.vline(0.25, 'thin dash', 'operating point: 4 MHz', ly=16.5)
    ax.text(0.62, 6.5, 'linear up to ≈ 10 MHz,', 'sm')
    ax.text(0.62, 5.2, 'then falls below the line', 'sm')
    ax.text(0.62, 15.0, 'dashed: linear extrapolation', 'sm')
    save('rabi-amplitude', ax.svg())



# ---------------------------------------------------------------- chapter 20: Ramsey and drift
A_1F = 87.3      # kHz^2 : 1/f frequency-noise amplitude, S(f) = A/f  (gives T2* = 6 us for 100 s averaging)


@fig
def ramsey_fringes():
    rng = np.random.default_rng(51)
    tau = np.linspace(0, 15, 301)      # us
    det, T2 = 1.0, 6.0                 # MHz, us
    P = 0.5 + 0.5 * np.cos(2 * np.pi * det * tau) * np.exp(-(tau / T2) ** 2)
    Pm = rng.binomial(200, OFF_E + VIS_E * P) / 200
    tt = np.linspace(0, 15, 1500)
    ax = Axes(860, 280, xlim=(0, 15), ylim=(-0.03, 1.03), xlabel='free evolution time τ (µs)', ylabel='spin-up fraction', mt=18, yticks=[0, 0.5, 1])
    ax.scatter(tau, Pm, 'f1', r=2.2, opacity=0.8)
    ax.line(tt, OFF_E + VIS_E * (0.5 + 0.5 * np.cos(2 * np.pi * det * tt) * np.exp(-(tt / T2) ** 2)), 't1')
    ax.line(tt, OFF_E + VIS_E * (0.5 + 0.5 * np.exp(-(tt / T2) ** 2)), 'thin dash')
    ax.line(tt, OFF_E + VIS_E * (0.5 - 0.5 * np.exp(-(tt / T2) ** 2)), 'thin dash')
    ax.text(8.5, 0.9, 'fit: detuning 1.000 MHz, T₂* = 6.0 µs, Gaussian envelope', 'sm')
    save('ramsey-fringes', ax.svg())


@fig
def t2star_averaging():
    Tm = np.logspace(-2, 4.6, 200)       # s
    fh = 1e5
    sig = np.sqrt(A_1F * np.log(fh * Tm))            # kHz
    T2 = np.sqrt(2) / (2 * np.pi * sig * 1e3) * 1e6  # us
    ax = Axes(600, 300, xlim=(-2, 4.6), ylim=(4, 10), xlabel='total measurement time (log₁₀ s)', ylabel='T₂* (µs)', mt=18,
              xticks=[-2, -1, 0, 1, 2, 3, 4], xfmt=lambda v: {-2: '10 ms', -1: '0.1 s', 0: '1 s', 1: '10 s', 2: '100 s', 3: '17 min', 4: '3 h'}.get(round(v), ''))
    ax.line(np.log10(Tm), T2, 't1')
    ax.vline(2, 'thin dash', 'our 6 µs: 100 s', ly=9.6)
    ax.text(-1.8, 4.5, 'the slower the measurement, the more slow noise it averages over', 'sm')
    save('t2star-averaging', ax.svg())


@fig
def frequency_drift():
    rng = np.random.default_rng(52)
    n, dt = 3600, 2.0                    # 2 h of Ramsey points every 2 s
    f = np.fft.rfftfreq(n, dt)
    amp = np.zeros_like(f)
    amp[1:] = np.sqrt(A_1F / f[1:] / (2 * dt * n)) * n
    ph = rng.uniform(0, 2 * np.pi, f.size)
    x = np.fft.irfft(amp * np.exp(1j * ph), n)
    t = np.arange(n) * dt / 60.0         # minutes
    x = x - x.mean() - 1.5 * t / 60.0    # magnet drift, -1.5 kHz per hour
    jump = np.where(t > 74, 45.0, 0.0)   # a two-level fluctuator switching once
    x = x + jump
    meas = x + rng.normal(0, 4.0, n)     # estimation noise per point
    ax = Axes(380, 300, xlim=(0, 120), ylim=(-120, 140), xlabel='time (min)', ylabel='qubit-1 frequency offset (kHz)', mt=18, title='Ramsey-tracked frequency', mr=12)
    ax.line(t, meas, 't1')
    ax.text(40, 120, 'jump at 74 min: a charge trap switched', 'sm')
    S = np.abs(np.fft.rfft(meas - meas.mean())) ** 2 * 2 * dt / n
    ax2 = Axes(300, 300, xlim=(-3.6, -0.6), ylim=(0, 6), xlabel='log₁₀ frequency (Hz)', ylabel='log₁₀ S (kHz²/Hz)', mt=18, title='its spectrum', mr=12,
               xticks=[-3, -2, -1])
    fb = f[1:]
    # bin logarithmically
    edges = np.logspace(np.log10(fb[0]), np.log10(fb[-1]), 22)
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (fb >= a) & (fb < b)
        if m.any():
            cx.append(np.log10(np.sqrt(a * b))); cy.append(np.log10(S[1:][m].mean()))
    ax2.scatter(np.array(cx), np.array(cy), 'f1', r=3.5, opacity=0.9)
    ax2.line(np.array([-3.6, -0.6]), np.log10(A_1F) - np.array([-3.6, -0.6]), 'thin dash')
    ax2.text(-2.0, 4.6, '∝ 1/f', 'sm')
    save('frequency-drift', row(ax.svg(), ax2.svg()))



# ---------------------------------------------------------------- chapter 21: echo, CPMG, noise spectroscopy
def _chi_cpmg(t, N, A=A_1F * 1e6, fl=1e-2):
    """Phase variance for CPMG-N (N=0: Ramsey) of total length t under S(f) = A/f (Hz^2/Hz).
    Uses the closed-form CPMG filter function (Cywinski et al. 2008): |Y|^2 = 2F/w^2."""
    grids = [np.logspace(np.log10(fl), 8, 20000)]
    if N > 0:
        f0 = N / (2 * t)
        grids.append(np.linspace(max(f0 - 60 / t, fl), f0 + 60 / t, 6000))
    f = np.unique(np.concatenate(grids))
    w = 2 * np.pi * f
    x = w * t
    if N == 0:
        Y2 = 4 * np.sin(x / 2) ** 2 / w ** 2
    else:
        sN = np.sin(x / 2) ** 2 if N % 2 == 0 else np.cos(x / 2) ** 2
        Y2 = 16 * np.sin(x / (4 * N)) ** 4 * sN / (np.cos(x / (2 * N)) ** 2 + 1e-30) / w ** 2
    return (2 * np.pi) ** 2 * np.trapezoid((A / f) * Y2, f)


def _cpmg_curve(N, ts):
    return np.exp(-np.array([_chi_cpmg(t, N) for t in ts]) / 2)


@fig
def echo_decays():
    rng = np.random.default_rng(61)
    ts = np.logspace(-6, -2, 70)
    ax = Axes(600, 340, xlim=(0, 4), ylim=(-0.03, 1.5), xlabel='total evolution time (log₁₀ µs)', ylabel='remaining coherence', mt=18, yticks=[0, 0.25, 0.5, 0.75, 1],
              xticks=[0, 1, 2, 3, 4], xfmt=lambda v: {0: '1', 1: '10', 2: '100', 3: '1000', 4: '10⁴'}.get(round(v), ''))
    for N, cls, lab in [(1, 't1', 'Hahn echo (N = 1): T₂ = 30 µs'), (16, 't3', 'CPMG N = 16: 105 µs'), (256, 't2', 'CPMG N = 256: 420 µs'), (1024, 't4', 'CPMG N = 1024: 840 µs')]:
        W = _cpmg_curve(N, ts)
        ax.line(np.log10(ts * 1e6), W, cls, lab)
        pts = ts[::4]
        Wm = _cpmg_curve(N, pts) + rng.normal(0, 0.02, pts.size)
        ax.scatter(np.log10(pts * 1e6), Wm, 'f' + cls[1], r=2.6, opacity=0.85)
    tt = np.logspace(-6, -4.5, 60)
    ax.line(np.log10(tt * 1e6), np.exp(-(tt / 6e-6) ** 2), 'thin dash')
    ax.text(0.05, 0.12, 'Ramsey: 6 µs', 'sm')
    save('echo-decays', ax.svg(legend_pos='tr'))


@fig
def t2_vs_n():
    Ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    T2s = []
    for N in Ns:
        ts = np.logspace(-5.5, -2, 50)
        W = _cpmg_curve(N, ts)
        T2s.append(np.interp(np.exp(-1), W[::-1], ts[::-1]))
    T2s = np.array(T2s) * 1e6
    p = np.polyfit(np.log10(Ns), np.log10(T2s), 1)
    rng = np.random.default_rng(62)
    ax = Axes(600, 300, xlim=(-0.1, 3.1), ylim=(1.2, 3.2), xlabel='number of π pulses N (log₁₀)', ylabel='T₂ (log₁₀ µs)', mt=18,
              xticks=[0, 1, 2, 3], yticks=[1.5, 2, 2.5, 3], xfmt=lambda v: {0: '1', 1: '10', 2: '100', 3: '1000'}.get(round(v), ''),
              yfmt=lambda v: {1.5: '30', 2: '100', 2.5: '300', 3: '1000'}.get(v, ''))
    ax.scatter(np.log10(Ns), np.log10(T2s) + rng.normal(0, 0.02, Ns.size), 'f1', r=4, opacity=0.95)
    xx = np.linspace(-0.1, 3.1, 10)
    ax.line(xx, np.polyval(p, xx), 't1', f'fit: T₂ ∝ N^{p[0]:.2f}')
    ax.text(1.6, 1.5, 'for 1/f noise, T₂ ∝ N^(1/2) (approximately)', 'sm')
    save('t2-vs-n', ax.svg(legend_pos='tl'))


@fig
def noise_spectrum():
    rng = np.random.default_rng(63)
    A = A_1F * 1e6     # Hz^2
    # CPMG reconstruction: S(f0) = -ln W / (8 t), f0 = N/(2t) (one-sided S in Hz^2/Hz), using points with W ~ 0.25-0.8
    xs, ys = [], []
    for N in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for t in np.logspace(-5.5, -2.5, 14):
            W = np.exp(-_chi_cpmg(t, N) / 2)
            if 0.25 < W < 0.8:
                f0 = N / (2 * t)
                ys.append(np.log10(-np.log(W) / (8 * t)) + rng.normal(0, 0.04))
                xs.append(np.log10(f0))
    # Ramsey-tracking spectrum points
    fl = np.logspace(-3.3, -0.8, 12)
    yl = np.log10(A / fl) + rng.normal(0, 0.08, fl.size)
    ax = Axes(600, 320, xlim=(-3.5, 6.5), ylim=(1, 11), xlabel='frequency (log₁₀ Hz)', ylabel='log₁₀ S_f (Hz²/Hz)', mt=18,
              xticks=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    xx = np.linspace(-3.5, 6.5, 10)
    ax.line(xx, np.log10(A) - xx, 'thin dash')
    ax.scatter(np.log10(fl), yl, 'f2', r=4, opacity=0.95, label='Ramsey tracking (Chapter 20)')
    ax.scatter(np.array(xs), np.array(ys), 'f1', r=3, opacity=0.8, label='CPMG filter reconstruction')
    ax.text(-0.3, 9.6, 'S_f = A/f, A = 87 kHz²: nine decades of the same noise', 'sm')
    ax.text(-0.5, 4.6, 'gap: neither method reaches 1 Hz–10 kHz here', 'sm')
    save('noise-spectrum', ax.svg(legend_pos='bl'))



# ---------------------------------------------------------------- chapter 22: calibration
def _rot(axis_phase, theta, det_frac=0.0):
    """SU(2) rotation by theta about an equatorial axis at angle axis_phase, with detuning det_frac (Delta/f_R)."""
    nx, ny, nz = np.cos(axis_phase), np.sin(axis_phase), det_frac
    nrm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    th = theta * nrm
    nx, ny, nz = nx / nrm, ny / nrm, nz / nrm
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c - 1j * nz * s, (-1j * nx - ny) * s], [(-1j * nx + ny) * s, c + 1j * nz * s]])


_ALLXY = [('I', 'I'), ('X', 'X'), ('Y', 'Y'), ('X', 'Y'), ('Y', 'X'),
          ('x', 'I'), ('y', 'I'), ('x', 'y'), ('y', 'x'), ('x', 'Y'), ('y', 'X'), ('X', 'y'), ('Y', 'x'), ('x', 'X'), ('X', 'x'), ('y', 'Y'), ('Y', 'y'),
          ('X', 'I'), ('Y', 'I'), ('x', 'x'), ('y', 'y')]


def _allxy(amp=1.0, det=0.0, yphase=np.pi / 2):
    out = []
    for a, b in _ALLXY:
        psi = np.array([1, 0], complex)
        for g in (a, b):
            if g == 'I':
                continue
            th = (np.pi if g.isupper() else np.pi / 2) * amp
            ph = 0.0 if g.lower() == 'x' else yphase
            psi = _rot(ph, th, det) @ psi
        out.append(abs(psi[1]) ** 2)
    return np.array(out)


@fig
def allxy():
    ideal = _allxy()
    ax = Axes(860, 300, xlim=(-0.7, 20.7), ylim=(-0.05, 1.1), xlabel='', ylabel='spin-up probability', mt=18, mb=62,
              xticks=list(range(21)), xfmt=lambda v: '', yticks=[0, 0.5, 1])
    xs = np.arange(21)
    for i in range(21):
        ax.line(np.array([i - 0.45, i + 0.45]), np.array([ideal[i], ideal[i]]), 'thin', extra='style="stroke:var(--ink-3);stroke-width:2"')
    ax.scatter(xs - 0.15, _allxy(amp=1.05), 'f2', r=4, opacity=0.95, label='amplitude 5% too high')
    ax.scatter(xs + 0.15, _allxy(det=0.08), 'f1', r=4, opacity=0.95, label='detuned by 0.08 f_R')
    svg = ax.svg(legend_pos='tl')
    # pulse-pair labels under the axis
    labs = ''.join(f'<text class="sm" x="{float(ax.X(i)):.1f}" y="{ax.h - 40}" text-anchor="middle" style="font-size:9.5px">{a}{b}</text>' for i, (a, b) in enumerate(_ALLXY))
    svg = svg.replace('</svg>', labs + f'<text class="sm" x="{ax.ml}" y="{ax.h - 14}">pulse pairs (X, Y: π; x, y: π/2; I: idle); grey bars: ideal 0, ½, 1</text></svg>')
    save('allxy', svg)


@fig
def error_amp():
    amps = np.linspace(0.94, 1.06, 241)
    ax = Axes(600, 300, xlim=(0.94, 1.06), ylim=(-0.03, 1.05), xlabel='X90 amplitude relative to the Rabi estimate', ylabel='spin-up probability', mt=18)
    for n, cls in [(2, 't3'), (10, 't1'), (42, 't2')]:
        P = np.array([abs((np.linalg.matrix_power(_rot(0, np.pi / 2 * a), n) @ np.array([1, 0]))[1]) ** 2 for a in amps * 0.992])
        ax.line(amps, OFF_E + VIS_E * P, cls, f'{n} × X90')
    ax.vline(0.992 ** -1, 'thin dash', 'true optimum', ly=0.08)
    ax.text(0.942, 0.62, 'more repetitions → sharper peak', 'sm')
    ax.text(0.942, 0.55, '→ more precise amplitude', 'sm')
    save('error-amp', ax.svg(legend_pos='tr'))


@fig
def heating_shift():
    t = np.linspace(0, 2000, 400)      # us since the burst train started
    shift = -120 * (1 - np.exp(-t / 300))     # kHz
    rng = np.random.default_rng(71)
    tm = np.arange(20, 2000, 80)
    ax = Axes(600, 300, xlim=(0, 2000), ylim=(-150, 20), xlabel='time since the bursts started (µs)', ylabel='frequency shift of Q2 (kHz)', mt=18)
    ax.hline(0, 'thin dash')
    ax.line(t, shift, 't2', 'without prepulse')
    ax.scatter(tm, -120 * (1 - np.exp(-tm / 300)) + rng.normal(0, 4, tm.size), 'f2', r=3.5, opacity=0.9)
    ax.line(t, np.full_like(t, -120), 't1', 'after a 1 ms off-resonant prepulse')
    ax.text(700, -95, 'settles in ~ 300 µs to −120 kHz', 'sm')
    save('heating-shift', ax.svg(legend_pos='tr'))



# ---------------------------------------------------------------- chapter 23: benchmarking
@fig
def rb_decay():
    rng = np.random.default_rng(81)
    ms = np.array([1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 300, 400, 600, 800, 1000])
    p_ref, p_int = 0.9970, 0.99556
    A, B = 0.47, 0.50
    ax = Axes(600, 320, xlim=(0, 1000), ylim=(0.45, 1.0), xlabel='number of Cliffords m', ylabel='probability of the expected outcome', mt=18)
    for p, cls, fcls, lab in [(p_ref, 't1', 'f1', 'reference: F_C = 99.85%'), (p_int, 't2', 'f2', 'interleaved with X90')]:
        for m in ms:
            seqs = B + A * p ** m + rng.normal(0, 0.004 + 0.012 * (1 - p ** m), 20)
            ax.scatter(np.full(20, m), np.clip(seqs, 0, 1), fcls, r=1.8, opacity=0.35)
        mm = np.linspace(0, 1000, 300)
        ax.line(mm, B + A * p ** mm, cls, lab)
    ax.hline(0.5, 'thin dash', 'fully randomized', lx=420)
    ax.text(20, 0.465, 'fit A p^m + B: A and B absorb SPAM, p measures the gates', 'sm')
    save('rb-decay', ax.svg(legend_pos='tr'))


@fig
def gst_budget():
    names = ['preparation', 'measurement', 'idle 125 ns', 'X90', 'Y90']
    coh = np.array([0.02, 0.10, 0.02, 0.025, 0.03])      # % (coherent part)
    inc = np.array([0.13, 1.30, 0.03, 0.045, 0.05])      # % (incoherent part)
    ax = Axes(600, 300, xlim=(-0.6, 4.6), ylim=(-2.7, 0.5), xlabel='', ylabel='log₁₀ error (%)', mt=18, mb=40, xticks=[0, 1, 2, 3, 4],
              xfmt=lambda v: names[int(round(v))] if 0 <= round(v) < 5 else '', yticks=[-2, -1, 0],
              yfmt=lambda v: {-2: '0.01%', -1: '0.1%', 0: '1%'}.get(v, ''))
    out = []
    for i in range(5):
        x0 = i - 0.3
        tot = coh[i] + inc[i]
        ax.area(np.array([x0, x0 + 0.28]), np.array([-2.7, -2.7]), np.array([np.log10(inc[i]), np.log10(inc[i])]), 'fa1')
        ax.area(np.array([x0 + 0.32, x0 + 0.6]), np.array([-2.7, -2.7]), np.array([np.log10(coh[i]), np.log10(coh[i])]), 'fa2')
        for xa, ya, c in [(x0, np.log10(inc[i]), 't1'), (x0 + 0.32, np.log10(coh[i]), 't2')]:
            ax.line(np.array([xa, xa, xa + 0.28, xa + 0.28]), np.array([-2.7, ya, ya, -2.7]), c)
        ax.text(i, np.log10(tot) + 0.12, f'{100 - tot:.2f}%', 'sm', anchor='middle')
    ax.line(np.array([-1, -1]), np.array([0, 0]), 't1', 'incoherent (decoherence)')
    ax.line(np.array([-1, -1]), np.array([0, 0]), 't2', 'coherent (calibration)')
    save('gst-budget', ax.svg(legend_pos='tr'))


# ================================================================ main
if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
        print('made', n)
