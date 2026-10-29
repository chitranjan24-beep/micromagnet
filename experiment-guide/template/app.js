(function () {
  var chapters = Array.prototype.slice.call(document.querySelectorAll('.chapter'));
  var byId = {};
  chapters.forEach(function (c) { byId[c.id] = c; });
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.rail a.toc'));
  var where = document.getElementById('where');
  var typeset = {};

  function chapterFor(hash) {
    if (!hash) return chapters[0];
    if (byId[hash]) return byId[hash];
    var el = document.getElementById(hash);
    if (el) { var c = el.closest('.chapter'); if (c) return c; }
    return chapters[0];
  }

  function mathFor(ch) {
    if (typeset[ch.id]) return;
    if (window.MathJax && MathJax.typesetPromise) {
      typeset[ch.id] = true;
      MathJax.typesetPromise([ch]).catch(function (e) { typeset[ch.id] = false; console.warn(e); });
    }
  }

  function show() {
    var hash = decodeURIComponent((location.hash || '').slice(1));
    var ch = chapterFor(hash);
    chapters.forEach(function (c) { c.hidden = c !== ch; });
    tocLinks.forEach(function (a) { a.classList.toggle('on', a.getAttribute('href') === '#' + ch.id); });
    if (where) where.textContent = ch.getAttribute('data-label') || '';
    document.body.classList.remove('menu-open');
    mathFor(ch);
    var target = hash && hash !== ch.id ? document.getElementById(hash) : null;
    if (target) { setTimeout(function () { target.scrollIntoView(); }, 30); }
    else { window.scrollTo(0, 0); }
    var on = document.querySelector('.rail a.toc.on');
    if (on && on.scrollIntoView) { try { on.scrollIntoView({ block: 'nearest' }); } catch (e) {} }
    try { localStorage.setItem('chip-to-qubit-last', ch.id); } catch (e) {}
  }

  window.addEventListener('hashchange', show);
  window.addEventListener('mathjax-ready', function () {
    chapters.forEach(function (c) { if (!c.hidden) mathFor(c); });
  });

  var menuBtn = document.getElementById('menuBtn');
  if (menuBtn) menuBtn.addEventListener('click', function () { document.body.classList.toggle('menu-open'); });

  var themeBtn = document.getElementById('themeBtn');
  if (themeBtn) themeBtn.addEventListener('click', function () {
    var root = document.documentElement;
    var cur = root.getAttribute('data-theme');
    var dark = cur ? cur === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme', dark ? 'light' : 'dark');
  });

  // Search over section headings and glossary terms
  var q = document.getElementById('q');
  var hits = document.getElementById('hits');
  var index = [];
  chapters.forEach(function (c) {
    var label = c.getAttribute('data-label') || '';
    c.querySelectorAll('h2[id], h3[id], dfn[id]').forEach(function (h) {
      index.push({ id: h.id, text: h.textContent.replace(/\s+/g, ' ').trim(), where: label });
    });
  });
  if (q) q.addEventListener('input', function () {
    var s = q.value.trim().toLowerCase();
    hits.innerHTML = '';
    if (s.length < 2) return;
    var n = 0;
    for (var i = 0; i < index.length && n < 30; i++) {
      if (index[i].text.toLowerCase().indexOf(s) >= 0) {
        var a = document.createElement('a');
        a.href = '#' + index[i].id;
        a.innerHTML = '';
        a.appendChild(document.createTextNode(index[i].text));
        var sm = document.createElement('small');
        sm.textContent = '  ' + index[i].where;
        a.appendChild(document.createElement('br'));
        a.appendChild(sm);
        hits.appendChild(a);
        n++;
      }
    }
    if (!n) hits.textContent = 'No matching headings or terms.';
  });

  show();
})();
