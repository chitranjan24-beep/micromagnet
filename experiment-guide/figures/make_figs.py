#!/usr/bin/env python3
"""Generate all plotted figures for the experiment guide into figures/out/.
Run: python3 figures/make_figs.py [name ...]   (no names = all figures)
All simulated data use the book's running-example device numbers (see chapter 2)."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotlib import Axes, colormap, row

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


# ================================================================ main
if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
        print('made', n)
