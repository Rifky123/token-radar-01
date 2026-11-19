// Prelaunch Atlas — a static dashboard over window.RADAR (written beside dashboard/ by main.ts).
// No network, no framework: every byte on the page comes from the sealed snapshot's own figures.
(function () {
  'use strict';
  var D = window.RADAR;
  if (!D) return;
  var app = document.getElementById('app');
  var state = { tab: 'board', q: '', cat: 'all', onlyWatch: false, sort: 'rank', dir: 1, shown: 40, sel: null };

  // Deep links let a reader (or a screenshot) open a view directly: #detail=<slug>,
  // #filter=<category>, #watchlist and #live. No network, no history rewriting.
  (function readHash() {
    var h;
    try { h = decodeURIComponent((window.location.hash || '').replace(/^#/, '')); } catch (e) { h = ''; }
    if (h.indexOf('detail=') === 0) state.sel = h.slice(7);
    else if (h.indexOf('filter=') === 0) state.cat = h.slice(7);
    else if (h === 'watchlist') state.onlyWatch = true;
    else if (h === 'live') state.tab = 'live';
  })();

  function usd(n) {
    if (n == null || !isFinite(n)) return 'n/a';
    if (Math.abs(n) >= 1e12) return '$' + (n / 1e12).toFixed(2) + 'T';
    if (Math.abs(n) >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return '$' + Math.round(n);
  }
  function pct(n) { return n == null || !isFinite(n) ? 'n/a' : (n > 0 ? '+' : '') + n.toFixed(2) + '%'; }
  function age(d) { return d == null ? 'unknown' : d >= 365 ? (d / 365).toFixed(1) + ' years' : Math.round(d) + ' days'; }

  function node(tag, attrs) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) n.setAttribute(k, v);
    });
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c == null) continue;
      if (Array.isArray(c)) c.forEach(function (x) { if (x) n.appendChild(x); });
      else if (typeof c === 'string') n.appendChild(document.createTextNode(c));
      else n.appendChild(c);
    }
    return n;
  }

  document.getElementById('asof').textContent = 'Sealed snapshot ' + D.as_of + ' · ' + D.board_size + ' ranked · ' + D.live_size + ' market anchors';

  function renderKpis() {
    var f = D.fit;
    var k = node('div', { class: 'kpis' },
      node('div', { class: 'kpi' }, node('div', { class: 'v', text: String(D.board_size) }), node('div', { class: 'l', text: 'protocols ranked' })),
      node('div', { class: 'kpi teal' }, node('div', { class: 'v', text: String(D.watchlist_size) }), node('div', { class: 'l', text: 'watchlist — all covered' })),
      node('div', { class: 'kpi' }, node('div', { class: 'v', text: String(D.early_size) }), node('div', { class: 'l', text: 'early finds below the floor' })),
      node('div', { class: 'kpi teal' }, node('div', { class: 'v', text: String(f.trainedOn) }), node('div', { class: 'l', text: 'traded comps fitted' })),
      node('div', { class: 'kpi' }, node('div', { class: 'v', text: f.cv.spearman.toFixed(2) }), node('div', { class: 'l', text: 'held-out rank correlation' }))
    );
    return k;
  }

  function renderMethod() {
    var f = D.fit;
    var left = node('div', {},
      node('h2', { text: 'How the value is derived' }),
      node('p', { html: 'The <b>' + f.trainedOn + ' protocols that already trade</b> are the comparables. A ridge fit in log space relates their market cap to TVL, seven-day TVL growth, time listed and chain count; the same function scores every pre-token row. No name is special-cased.' }),
      node('p', { html: '<code>value = 10^(a + b·log10(TVL) + c·growth + d·logTime + e·listed + f·logChains)</code>' }),
      node('p', { html: 'Five-fold cross-validation holds the <b>' + f.cv.folds + '-fold</b> rank correlation at <b>' + f.cv.spearman.toFixed(3) + '</b> and median absolute error at <b>' + f.cv.medianAbsLogError.toFixed(2) + ' log units</b>, against <b>' + f.cv.baselineMedianAbsLogError.toFixed(2) + '</b> for a flat TVL multiple. Rank bands come from 24 seeded bootstrap refits.' })
    );
    var right = node('div', { class: 'stats' },
      node('div', { class: 'stat' }, node('div', { class: 'v', text: f.tvlCoef.toFixed(3) }), node('div', { class: 'l', text: 'TVL elasticity (log)' })),
      node('div', { class: 'stat' }, node('div', { class: 'v', text: f.growthCoef.toFixed(2) }), node('div', { class: 'l', text: 'growth coefficient' })),
      node('div', { class: 'stat' }, node('div', { class: 'v', text: f.ageCoef.toFixed(2) }), node('div', { class: 'l', text: 'age coefficient' })),
      node('div', { class: 'stat' }, node('div', { class: 'v', text: f.cv.rmse.toFixed(2) }), node('div', { class: 'l', text: 'held-out RMSE (log)' }))
    );
    return node('section', { class: 'method' }, left, right);
  }

  function tabs() {
    var t = node('div', { class: 'tabs', role: 'tablist' });
    [['board', 'Pre-token board'], ['live', 'Already trading'], ['method', 'Model & method']].forEach(function (x) {
      t.appendChild(node('button', {
        class: 'tab', role: 'tab', text: x[1], 'aria-selected': state.tab === x[0] ? 'true' : 'false',
        onclick: function () { state.tab = x[0]; state.sel = null; render(); }
      }));
    });
    return t;
  }

  function filtered() {
    var q = state.q.toLowerCase().trim();
    var cats = new Set();
    D.rows.forEach(function (r) { cats.add(r.category || 'Other'); });
    var list = D.rows.filter(function (r) {
      if (state.onlyWatch && !r.watchlist) return false;
      if (state.cat !== 'all' && (r.category || 'Other') !== state.cat) return false;
      if (q && (r.name + ' ' + r.slug + ' ' + (r.category || '') + ' ' + r.chains.join(' ')).toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
    var key = state.sort;
    var dir = state.dir;
    list.sort(function (a, b) {
      var x = a[key], y = b[key];
      if (typeof x === 'string') return dir * (x < y ? -1 : x > y ? 1 : 0);
      if (x == null) x = -Infinity; if (y == null) y = -Infinity;
      return dir * (x - y) || a.rank - b.rank;
    });
    return { list: list, cats: Array.from(cats).sort() };
  }

  function th(label, key, cls) {
    return node('th', {
      class: cls || '', text: label,
      'aria-sort': state.sort === key ? (state.dir === 1 ? 'ascending' : 'descending') : 'none',
      onclick: function () { if (state.sort === key) state.dir = -state.dir; else { state.sort = key; state.dir = key === 'rank' ? 1 : -1; } render(); }
    });
  }

  function boardPanel() {
    var f = filtered();
    var list = f.list;
    var shown = list.slice(0, state.shown);
    var search = node('input', { type: 'search', placeholder: 'Search name, chain or category…', value: state.q, oninput: function (e) { state.q = e.target.value; state.shown = 40; render(); } });
    var sel = node('select', { onchange: function (e) { state.cat = e.target.value; state.shown = 40; render(); } }, node('option', { value: 'all', text: 'All categories' }));
    f.cats.forEach(function (c) { sel.appendChild(node('option', { value: c, text: c, selected: state.cat === c ? 'selected' : null })); });
    var watch = node('button', {
      class: 'tab', text: state.onlyWatch ? 'Watchlist only ✓' : 'Watchlist only',
      onclick: function () { state.onlyWatch = !state.onlyWatch; state.shown = 40; render(); }
    });
    var toolbar = node('div', { class: 'toolbar' }, search, sel, watch,
      node('span', { class: 'count', text: list.length + ' of ' + D.board_size + ' rows' }));

    var tbody = node('tbody');
    shown.forEach(function (r) {
      var chg = r.change_7d;
      tbody.appendChild(node('tr', { onclick: function () { state.sel = r.slug; render(); } },
        node('td', { class: 'rank', text: String(r.rank) }),
        node('td', { class: 'name' + (r.watchlist ? ' watch' : ''), text: r.name }),
        node('td', {}, (r.chains || []).slice(0, 2).map(function (c) { return node('span', { class: 'tag', text: c }); })),
        node('td', { text: r.category || 'Other' }),
        node('td', { class: 'num', text: usd(r.tvl_usd) }),
        node('td', { class: 'num ' + (chg == null ? '' : chg >= 0 ? 'up' : 'down'), text: pct(chg) }),
        node('td', { class: 'num value', text: usd(r.implied_value_usd) }),
        node('td', { class: 'num band', text: '#' + r.rank_band[0] + '–' + r.rank_band[1] })
      ));
    });
    var table = node('table', {},
      node('thead', {}, node('tr', {},
        th('#', 'rank'), th('Protocol', 'name'), th('Chain', 'category'), th('Category', 'category'),
        th('TVL', 'tvl_usd', 'num'), th('7d', 'change_7d', 'num'), th('Implied value', 'implied_value_usd', 'num'), th('Band', 'rank_band', 'num'))),
      tbody);
    var panel = node('div', { class: 'panel' }, toolbar, table);
    if (shown.length < list.length) panel.appendChild(node('div', { class: 'more' },
      node('button', { text: 'Show 40 more of ' + (list.length - shown.length), onclick: function () { state.shown += 40; render(); } })));
    return panel;
  }

  function livePanel() {
    var tbody = node('tbody');
    D.live.forEach(function (r) {
      tbody.appendChild(node('tr', {},
        node('td', { class: 'name', text: r.name }),
        node('td', { text: r.category || 'Other' }),
        node('td', { class: 'num', text: usd(r.tvl_usd) }),
        node('td', { class: 'num value', text: usd(r.mcap_usd) }),
        node('td', { class: 'num', text: r.mcap_tvl.toFixed(2) + '×' }),
        node('td', { class: 'num', text: pct(r.change_7d) })
      ));
    });
    return node('div', { class: 'panel' },
      node('p', { class: 'explain', html: 'These protocols <b>already record a token</b>. They are the market anchors the value fit learns from — never mixed into the pre-token ranking. Market cap is DefiLlama\'s own record, not this model\'s claim.' }),
      node('table', {}, node('thead', {}, node('tr', {}, node('th', { text: 'Protocol' }), node('th', { text: 'Category' }),
        node('th', { class: 'num', text: 'TVL' }), node('th', { class: 'num', text: 'Market cap' }),
        node('th', { class: 'num', text: 'Mcap / TVL' }), node('th', { class: 'num', text: '7d' }))), tbody));
  }

  function detailPanel() {
    var r = null;
    for (var i = 0; i < D.rows.length; i++) if (D.rows[i].slug === state.sel) { r = D.rows[i]; break; }
    if (!r) return null;
    var plan = node('ol', { class: 'plan' });
    r.farm_steps.forEach(function (s) {
      var m = /Cost: (.+?)\. Time: (.+?)\.$/.exec(s);
      var action = m ? s.slice(0, m.index) : s;
      var cost = m ? m[1] + ' · ' + m[2] : '';
      plan.appendChild(node('li', {}, action, node('span', { class: 'cost', text: cost })));
    });
    var facts = node('div', { class: 'facts' },
      node('div', { class: 'fact' }, node('div', { class: 'l', text: 'TVL' }), node('div', { class: 'v', text: usd(r.tvl_usd) })),
      node('div', { class: 'fact' }, node('div', { class: 'l', text: '7-day TVL change' }), node('div', { class: 'v', text: pct(r.change_7d) })),
      node('div', { class: 'fact' }, node('div', { class: 'l', text: 'Listed since' }), node('div', { class: 'v', text: age(r.age_days) })),
      node('div', { class: 'fact' }, node('div', { class: 'l', text: 'Bootstrap rank band' }), node('div', { class: 'v', text: '#' + r.rank_band[0] + '–' + r.rank_band[1] })),
      node('div', { class: 'fact' }, node('div', { class: 'l', text: 'Growth multiplier' }), node('div', { class: 'v', text: r.growth_mult.toFixed(3) + '×' })),
      node('div', { class: 'fact' }, node('div', { class: 'l', text: 'Maturity multiplier' }), node('div', { class: 'v', text: r.maturity_mult.toFixed(3) + '×' }))
    );
    var left = node('div', {},
      node('button', { class: 'back', text: '← Back to board', onclick: function () { state.sel = null; render(); } }),
      node('h2', { text: r.name }),
      node('div', { class: 'crumb', text: 'Rank #' + r.rank + ' · ' + (r.category || 'Other') + ' · ' + (r.chains || []).join(', ') + (r.watchlist ? ' · on the watchlist' : ' · early find below the floor') }),
      node('div', { class: 'big', text: usd(r.implied_value_usd) }, node('small', { text: 'implied value if a token launched against today\'s record' })),
      facts,
      node('div', { class: 'formula', html: '<b>Reading the model:</b> a ' + usd(r.tvl_usd) + ' TVL anchor, ' + pct(r.change_7d) + ' seven-day growth and ' + age(r.age_days) + ' on record produce this value under the fitted comps curve. DefiLlama records no token for this protocol; that is a record, not proof none exists.' })
    );
    var right = node('div', {}, node('h2', { text: 'How to farm it' }),
      node('p', { class: 'explain', text: 'Seven steps, each with its cash cost and elapsed time, derived from this protocol\'s own category, chains, TVL and age.' }), plan);
    return node('div', { class: 'detail' }, left, right);
  }

  function render() {
    app.innerHTML = '';
    app.appendChild(renderKpis());
    app.appendChild(renderMethod());
    app.appendChild(tabs());
    var d = state.sel ? detailPanel() : state.tab === 'board' ? boardPanel() : state.tab === 'live' ? livePanel() : node('div', { class: 'panel' }, renderMethod());
    app.appendChild(d || boardPanel());
    document.getElementById('foot').textContent =
      'Prelaunch Atlas · built from the sealed ' + D.as_of + ' DefiLlama snapshot · ' +
      D.fit.version + ' · values are this radar\'s own claim, not a prediction of any token event.';
  }
  render();
})();
