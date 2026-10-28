"""Tiny SVG plotting helpers whose colours follow the page theme (CSS classes from style.css)."""
import base64, io, math
import numpy as np


def _nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    t = start
    while t <= hi + 1e-9 * span:
        ticks.append(round(t, 12))
        t += step
    return ticks


def _fmt(v):
    if abs(v) < 1e-12:
        return '0'
    if abs(v) >= 1000 or abs(v) < 0.01:
        s = f'{v:.3g}'
    else:
        s = f'{v:.3f}'.rstrip('0').rstrip('.')
    return s.replace('-', '−')


class Axes:
    def __init__(self, w=560, h=300, xlim=(0, 1), ylim=(0, 1), xlabel='', ylabel='', title='',
                 ml=58, mr=28, mt=26, mb=46, xticks=None, yticks=None, grid=True, xfmt=None, yfmt=None):
        self.w, self.h = w, h
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.xlim, self.ylim = xlim, ylim
        self.parts = []
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
        self.xticks = xticks if xticks is not None else _nice_ticks(*xlim)
        self.yticks = yticks if yticks is not None else _nice_ticks(*ylim)
        self.grid = grid
        self.xfmt = xfmt or _fmt
        self.yfmt = yfmt or _fmt
        self.legend_items = []

    def X(self, x):
        x0, x1 = self.xlim
        return self.ml + (np.asarray(x, float) - x0) / (x1 - x0) * (self.w - self.ml - self.mr)

    def Y(self, y):
        y0, y1 = self.ylim
        return self.h - self.mb - (np.asarray(y, float) - y0) / (y1 - y0) * (self.h - self.mt - self.mb)

    def line(self, x, y, cls='t1', label=None, extra=''):
        X, Y = self.X(x), self.Y(y)
        pts = []
        for a, b in zip(X, Y):
            if np.isfinite(a) and np.isfinite(b):
                pts.append(f'{a:.1f},{b:.1f}')
        self.parts.append(f'<polyline class="{cls}" points="{" ".join(pts)}" {extra}/>')
        if label:
            self.legend_items.append((cls, label))

    def area(self, x, y0, y1, cls='fa1'):
        X = self.X(x)
        A = self.Y(y1); B = self.Y(y0)
        pts = [f'{a:.1f},{b:.1f}' for a, b in zip(X, A)] + [f'{a:.1f},{b:.1f}' for a, b in zip(X[::-1], B[::-1])]
        self.parts.append(f'<polygon class="{cls}" points="{" ".join(pts)}"/>')

    def scatter(self, x, y, cls='f1', r=1.6, opacity=0.5, label=None):
        X, Y = self.X(x), self.Y(y)
        s = ''.join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{r}"/>' for a, b in zip(X, Y))
        self.parts.append(f'<g class="{cls}" fill-opacity="{opacity}">{s}</g>')
        if label:
            self.legend_items.append((cls.replace('f', 't'), label))

    def vline(self, x, cls='thin dash', label=None, ly=None):
        X = float(self.X(x))
        self.parts.append(f'<line class="{cls}" x1="{X:.1f}" x2="{X:.1f}" y1="{self.mt}" y2="{self.h - self.mb}"/>')
        if label:
            yy = self.mt + 12 if ly is None else float(self.Y(ly))
            self.parts.append(f'<text class="sm" x="{X + 4:.1f}" y="{yy:.1f}">{label}</text>')

    def hline(self, y, cls='thin dash', label=None, lx=None):
        Y = float(self.Y(y))
        self.parts.append(f'<line class="{cls}" x1="{self.ml}" x2="{self.w - self.mr}" y1="{Y:.1f}" y2="{Y:.1f}"/>')
        if label:
            xx = self.w - self.mr - 4 if lx is None else float(self.X(lx))
            anchor = 'end' if lx is None else 'start'
            self.parts.append(f'<text class="sm" text-anchor="{anchor}" x="{xx:.1f}" y="{Y - 4:.1f}">{label}</text>')

    def text(self, x, y, s, cls='sm', anchor='start', dx=0, dy=0):
        self.parts.append(f'<text class="{cls}" text-anchor="{anchor}" x="{float(self.X(x)) + dx:.1f}" y="{float(self.Y(y)) + dy:.1f}">{s}</text>')

    def image(self, rgba, extent=None):
        """rgba: HxWx4 uint8 array covering xlim x ylim (row 0 = top)."""
        from PIL import Image
        im = Image.fromarray(rgba)
        buf = io.BytesIO(); im.save(buf, format='PNG', optimize=True)
        b = base64.b64encode(buf.getvalue()).decode()
        x0, x1 = (extent or (self.xlim + self.ylim))[:2]
        y0, y1 = (extent or (self.xlim + self.ylim))[2:]
        X0, X1 = float(self.X(x0)), float(self.X(x1))
        Y0, Y1 = float(self.Y(y1)), float(self.Y(y0))
        self.parts.append(f'<image x="{X0:.1f}" y="{Y0:.1f}" width="{X1 - X0:.1f}" height="{Y1 - Y0:.1f}" preserveAspectRatio="none" href="data:image/png;base64,{b}"/>')

    def svg(self, legend_pos='tr'):
        w, h, ml, mr, mt, mb = self.w, self.h, self.ml, self.mr, self.mt, self.mb
        out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" xmlns="http://www.w3.org/2000/svg">']
        if self.grid:
            for t in self.xticks:
                X = float(self.X(t)); out.append(f'<line class="grid" x1="{X:.1f}" x2="{X:.1f}" y1="{mt}" y2="{h - mb}"/>')
            for t in self.yticks:
                Y = float(self.Y(t)); out.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{Y:.1f}" y2="{Y:.1f}"/>')
        out.append(f'<defs><clipPath id="c{id(self)}"><rect x="{ml}" y="{mt}" width="{w - ml - mr}" height="{h - mt - mb}"/></clipPath></defs>')
        out.append(f'<g clip-path="url(#c{id(self)})">' + ''.join(self.parts) + '</g>')
        out.append(f'<rect class="thin" x="{ml}" y="{mt}" width="{w - ml - mr}" height="{h - mt - mb}" style="stroke: var(--ink-3)"/>')
        for t in self.xticks:
            X = float(self.X(t))
            out.append(f'<line class="thin" x1="{X:.1f}" x2="{X:.1f}" y1="{h - mb}" y2="{h - mb + 4}"/>')
            out.append(f'<text class="sm" text-anchor="middle" x="{X:.1f}" y="{h - mb + 17}">{self.xfmt(t)}</text>')
        for t in self.yticks:
            Y = float(self.Y(t))
            out.append(f'<line class="thin" x1="{ml - 4}" x2="{ml}" y1="{Y:.1f}" y2="{Y:.1f}"/>')
            out.append(f'<text class="sm" text-anchor="end" x="{ml - 7}" y="{Y + 4:.1f}">{self.yfmt(t)}</text>')
        if self.xlabel:
            out.append(f'<text class="lbl" text-anchor="middle" x="{(ml + w - mr) / 2:.1f}" y="{h - 8}">{self.xlabel}</text>')
        if self.ylabel:
            out.append(f'<text class="lbl" text-anchor="middle" transform="translate(14 {(mt + h - mb) / 2:.1f}) rotate(-90)">{self.ylabel}</text>')
        if self.title:
            out.append(f'<text class="ttl" x="{ml}" y="{mt - 9}">{self.title}</text>')
        if self.legend_items:
            n = len(self.legend_items)
            lw = max(len(l) for _, l in self.legend_items) * 6.6 + 34
            lx = w - mr - lw - 6 if 'r' in legend_pos else ml + 6
            ly = mt + 6 if 't' in legend_pos else h - mb - 6 - n * 17
            out.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{n * 17 + 6}" rx="3" style="fill: var(--paper); fill-opacity: .9; stroke: var(--rule)"/>')
            for i, (cls, lab) in enumerate(self.legend_items):
                yy = ly + 14 + i * 17
                out.append(f'<line class="{cls}" x1="{lx + 6:.1f}" x2="{lx + 24:.1f}" y1="{yy - 4:.1f}" y2="{yy - 4:.1f}"/>')
                out.append(f'<text class="sm" x="{lx + 29:.1f}" y="{yy:.1f}">{lab}</text>')
        out.append('</svg>')
        return ''.join(out)


def colormap(z, cmap='viridis', vmin=None, vmax=None):
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    z = np.asarray(z, float)
    vmin = np.nanmin(z) if vmin is None else vmin
    vmax = np.nanmax(z) if vmax is None else vmax
    n = np.clip((z - vmin) / (vmax - vmin + 1e-30), 0, 1)
    return (cm(n) * 255).astype(np.uint8)


def row(*svgs, gap=10):
    """Place several complete <svg> strings side by side in one flex row (wraps on phones)."""
    inner = ''.join(f'<div style="flex:1 1 260px;min-width:0">{s}</div>' for s in svgs)
    return f'<div style="display:flex;flex-wrap:wrap;gap:{gap}px;align-items:flex-start">{inner}</div>'
