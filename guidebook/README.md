# RFSoC 4x2 Spin-Qubit Guidebook

A from-zero guidebook to semiconductor spin-qubit experiments with the RFSoC 4x2 board, PYNQ, QICK and spinQICK.

- `chapters/` — one HTML fragment per chapter (meta comment on the first line sets id, part and title)
- `refs.json` — bibliography; cite with `<cite data-r="key"></cite>`
- `figures/` — `plotlib.py` (theme-aware SVG plots) and `make_figs.py` (generates `figures/out/`)
- `template/` — page shell, styles and navigation script
- `build.py` — assembles everything into `dist/index.html`

Build:

```
python3 figures/make_figs.py   # regenerate plots (needs numpy, matplotlib, scipy)
python3 build.py               # writes dist/index.html
python3 build.py --preview     # also writes dist/preview.html for local viewing
```
