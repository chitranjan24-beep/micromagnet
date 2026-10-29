#!/usr/bin/env python3
"""Assemble the experiment guide into one HTML page.

Chapters live in chapters/NN-name.html. Each starts with a meta comment:
    <!--meta {"id": "bloch", "part": "Part I · Qubit physics", "title": "...", "numbered": true}-->

Markup conventions inside chapters:
    <cite data-r="key1,key2"></cite>     numbered citation, keys from refs.json
    <dfn data-def="one-line definition">Term</dfn>   collected into the glossary
    <!--plot:name-->                      inlines figures/out/name.svg (or .png as <img>)
    <figure> ... <figcaption>             auto-numbered "Figure N.k"
    <h2>, <h3>                            h2 auto-numbered N.M; ids added if missing
Special placeholders: <!--CONTENTS-->, <!--REFERENCES-->, <!--GLOSSARY-->

Usage: python3 build.py [--preview]   (--preview also writes dist/preview.html with a full document skeleton)
"""
import base64, glob, html, json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def slug(s):
    s = re.sub(r'<[^>]+>', '', s)
    s = html.unescape(s).lower()
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    return s[:60] or 'x'


def load_chapters():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'chapters', '*.html'))):
        src = open(path, encoding='utf-8').read()
        m = re.match(r'\s*<!--meta\s+(\{.*?\})\s*-->', src, re.S)
        if not m:
            sys.exit(f'missing meta in {path}')
        meta = json.loads(m.group(1))
        meta.setdefault('numbered', True)
        meta['body'] = src[m.end():]
        meta['path'] = os.path.basename(path)
        out.append(meta)
    return out


def main():
    preview = '--preview' in sys.argv
    refs = json.load(open(os.path.join(ROOT, 'refs.json'), encoding='utf-8'))
    chapters = load_chapters()
    n = 0
    for ch in chapters:
        if ch['numbered']:
            n += 1
            ch['num'] = n
            ch['label'] = f"Chapter {n} · {ch['title']}"
        else:
            ch['num'] = None
            ch['label'] = ch['title']

    cite_order = []
    gloss = {}
    errors = []

    for ch in chapters:
        body = ch['body']
        cid = ch['id']

        # plots
        def plot(m):
            name = m.group(1)
            svg = os.path.join(ROOT, 'figures', 'out', name + '.svg')
            png = os.path.join(ROOT, 'figures', 'out', name + '.png')
            if os.path.exists(svg):
                return open(svg, encoding='utf-8').read()
            if os.path.exists(png):
                b = base64.b64encode(open(png, 'rb').read()).decode()
                return f'<img alt="{name}" src="data:image/png;base64,{b}">'
            errors.append(f'{ch["path"]}: missing plot {name}')
            return f'<p class="status">[missing figure {name}]</p>'
        body = re.sub(r'<!--plot:([\w-]+)-->', plot, body)

        # section numbering and ids
        sec = 0
        used = set()

        def head(m):
            nonlocal sec
            tag, attrs, inner = m.group(1), m.group(2) or '', m.group(3)
            if 'id=' not in attrs:
                base = f'{cid}-{slug(inner)}'
                i, hid = 2, base
                while hid in used:
                    hid = f'{base}-{i}'; i += 1
                used.add(hid)
                attrs += f' id="{hid}"'
            if tag == 'h2':
                sec += 1
                if ch['num'] and 'data-nonum' not in attrs:
                    inner = f'<span class="sn">{ch["num"]}.{sec}</span>{inner}'
            return f'<{tag}{attrs}>{inner}</{tag}>'
        body = re.sub(r'<(h2|h3)(\s[^>]*)?>(.*?)</\1>', head, body, flags=re.S)

        # figure numbering
        fig = 0

        def figcap(m):
            nonlocal fig
            fig += 1
            label = f'Figure {ch["num"]}.{fig}' if ch['num'] else f'Figure {fig}'
            return f'<figcaption><b>{label}.</b> '
        body = re.sub(r'<figcaption>\s*', figcap, body)

        # citations
        def cite(m):
            keys = [k.strip() for k in m.group(1).split(',') if k.strip()]
            parts = []
            for k in keys:
                if k not in refs:
                    errors.append(f'{ch["path"]}: unknown ref {k}')
                    continue
                if k not in cite_order:
                    cite_order.append(k)
                i = cite_order.index(k) + 1
                parts.append(f'<a href="#ref-{k}" title="{html.escape(refs[k]["short"] if "short" in refs[k] else refs[k]["text"][:80])}">{i}</a>')
            return '<cite class="r">[' + ', '.join(parts) + ']</cite>'
        body = re.sub(r'<cite data-r="([^"]+)"\s*>\s*</cite>', cite, body)

        # glossary terms
        def dfn(m):
            d, term = m.group(1), m.group(2)
            plain = html.unescape(re.sub(r'<[^>]+>', '', term)).strip()
            key = plain.lower()
            gid = 'g-' + slug(plain)
            if key in gloss:
                return f'<dfn title="{d}">{term}</dfn>'
            gloss[key] = {'term': plain, 'def': html.unescape(d), 'id': gid, 'where': ch['label'], 'chid': cid}
            return f'<dfn id="{gid}" title="{d}">{term}</dfn>'
        body = re.sub(r'<dfn data-def="([^"]*)">(.*?)</dfn>', dfn, body, flags=re.S)
        ch['body'] = body

    # rail + contents
    rail, contents = [], []
    parts = []
    for ch in chapters:
        if not parts or parts[-1][0] != ch['part']:
            parts.append((ch['part'], []))
        parts[-1][1].append(ch)
    for pname, chs in parts:
        rail.append(f'    <div class="part">{html.escape(pname)}</div>')
        items = []
        for ch in chs:
            num = ch['num'] if ch['num'] else ''
            rail.append(f'    <a class="toc" href="#{ch["id"]}"><span class="n">{num}</span><span>{html.escape(ch["title"])}</span></a>')
            items.append(f'<li><span class="n">{num}</span><a href="#{ch["id"]}">{html.escape(ch["title"])}</a></li>')
        contents.append(f'<section><h3>{html.escape(pname)}</h3><ol>{"".join(items)}</ol></section>')
    contents_html = '<div class="contents-grid">' + ''.join(contents) + '</div>'

    # references
    reflis = []
    allkeys = cite_order + sorted(k for k in refs if k not in cite_order)
    for k in allkeys:
        r = refs[k]
        link = f' <a href="{html.escape(r["url"])}">{html.escape(r.get("linktext", "link"))}</a>' if r.get('url') else ''
        note = f' <span class="status">{html.escape(r["note"])}</span>' if r.get('note') else ''
        reflis.append(f'<li id="ref-{k}">{r["text"]}{link}{note}</li>')
    refs_html = '<ol class="refs">' + '\n'.join(reflis) + '</ol>'

    gl = sorted(gloss.values(), key=lambda g: g['term'].lower())
    gloss_html = '<dl class="gloss">' + ''.join(
        f'<dt><a href="#{g["id"]}">{html.escape(g["term"])}</a></dt><dd>{html.escape(g["def"])} <span class="status">({html.escape(g["where"])})</span></dd>'
        for g in gl) + '</dl>'

    out_ch = []
    for i, ch in enumerate(chapters):
        body = ch['body'].replace('<!--CONTENTS-->', contents_html).replace('<!--REFERENCES-->', refs_html).replace('<!--GLOSSARY-->', gloss_html)
        kicker = f'<p class="kicker">{html.escape(ch["part"])}' + (f' · Chapter {ch["num"]}' if ch['num'] else '') + '</p>'
        title = f'<h1>{ch["title"]}</h1>' if not ch.get('notitle') else ''
        prev_ = chapters[i - 1] if i > 0 else None
        next_ = chapters[i + 1] if i + 1 < len(chapters) else None
        pager = '<nav class="pager">'
        if prev_:
            pager += f'<a class="prev" href="#{prev_["id"]}"><small>Previous</small>{html.escape(prev_["label"])}</a>'
        if next_:
            pager += f'<a class="next" href="#{next_["id"]}"><small>Next</small>{html.escape(next_["label"])}</a>'
        pager += '</nav>'
        hidden = '' if i == 0 else ' hidden'
        out_ch.append(f'<article class="chapter" id="{ch["id"]}" data-label="{html.escape(ch["label"])}"{hidden}>\n{kicker}\n{title}\n{body}\n{pager}\n</article>')

    shell = open(os.path.join(ROOT, 'template', 'shell.html'), encoding='utf-8').read()
    css = open(os.path.join(ROOT, 'template', 'style.css'), encoding='utf-8').read()
    js = open(os.path.join(ROOT, 'template', 'app.js'), encoding='utf-8').read()
    page = (shell.replace('/*STYLE*/', css).replace('/*APP*/', js)
            .replace('<!--RAIL-->', '\n'.join(rail)).replace('<!--CHAPTERS-->', '\n'.join(out_ch)))
    os.makedirs(os.path.join(ROOT, 'dist'), exist_ok=True)
    open(os.path.join(ROOT, 'dist', 'index.html'), 'w', encoding='utf-8').write(page)
    if preview:
        doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>' + page + '</body></html>')
        open(os.path.join(ROOT, 'dist', 'preview.html'), 'w', encoding='utf-8').write(doc)
    size = len(page.encode()) / 1e6
    print(f'{len(chapters)} chapters, {len(cite_order)} cited refs, {len(gloss)} glossary terms, {size:.2f} MB')
    if errors:
        print('ERRORS:'); print('\n'.join(errors)); sys.exit(1)


if __name__ == '__main__':
    main()
