// ASL Sign Classifier — front end for app.py. Plain DOM, no build step.
const $ = (id) => document.getElementById(id);
const SVGNS = 'http://www.w3.org/2000/svg';
const S = { info: null, metric: 'acc', active: null };
// sequential blue ramp for the confusion matrix; zero recedes to the surface (dark mode)
const RAMP = ['#104281', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#b7d3f6'];

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function svg(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
const pct = (v) => `${(v * 100).toFixed(1)}%`;

// ------------------------------------------------------------------ info + clips
async function init() {
  S.info = await (await fetch('/api/info')).json();
  const i = S.info;
  const pills = $('pills');
  const pill = (label, value) => { const p = el('span', { class: 'pill' }); p.append(`${label} `); p.append(el('b', {}, value)); pills.append(p); };
  pill('backbone', i.backbone);
  pill('signs', String(i.labels.length));
  pill('input', `${i.num_frames} × ${i.frame_size.join('×')}`);
  if (i.best_val_acc != null) pill('best val', pct(i.best_val_acc));
  pill('params', `${(i.parameters / 1e6).toFixed(1)}M`);
  pill('device', i.device);

  const box = $('clips');
  for (const c of i.clips) {
    const b = el('button', { class: 'clip', 'data-path': c.path });
    b.append(el('span', {}, c.label));
    b.append(el('small', {}, c.path.split('/').pop().replace('.mp4', '')));
    b.addEventListener('click', () => classifyClip(c));
    box.append(b);
  }
  drawCurve();
  if (i.clips.length) classifyClip(i.clips.find((c) => c.label === 'drink') || i.clips[0]);
}

// ------------------------------------------------------------------ prediction
async function classifyClip(c) {
  S.active = c;
  document.querySelectorAll('.clip').forEach((b) => b.classList.toggle('active', b.dataset.path === c.path));
  $('video').src = `/clips/${encodeURI(c.path)}`;
  $('video').play().catch(() => {});
  render(await (await fetch(`/api/predict?clip=${encodeURIComponent(c.path)}`, { method: 'POST' })).json(), c.label);
}

async function classifyFile(file) {
  S.active = null;
  document.querySelectorAll('.clip').forEach((b) => b.classList.remove('active'));
  $('video').src = URL.createObjectURL(file);
  $('video').play().catch(() => {});
  $('pred').textContent = '…';
  const r = await fetch('/api/predict', { method: 'POST', headers: { 'X-Filename': file.name }, body: file });
  render(await r.json(), null);
}

function render(res, truth) {
  $('pred').textContent = res.label;
  const t = $('truth'); t.textContent = '';
  if (truth) {
    const ok = truth === res.label;
    t.append(el('span', { class: ok ? 'ok' : 'no' }, ok ? '✓ correct' : '✗ wrong'));
    t.append(` · true sign: ${truth}`);
  } else t.append('uploaded clip');
  const list = $('top5'); list.textContent = '';
  for (const p of res.top) {
    const li = el('li');
    li.append(el('span', {}, p.label));
    const bar = el('span', { class: 'bar' }); const fill = el('i'); fill.style.width = pct(p.confidence); bar.append(fill);
    li.append(bar, el('span', { class: 'p' }, pct(p.confidence)));
    list.append(li);
  }
  $('timing').textContent = `decode ${res.timing_ms.decode} ms · model ${res.timing_ms.model} ms`;
  const fr = $('frames'); fr.textContent = '';
  for (const f of res.frames) {
    const fig = el('figure'); fig.append(el('img', { src: f.src, alt: `frame ${f.index}` }), el('figcaption', {}, `#${f.index + 1}`));
    fr.append(fig);
  }
  $('frames-note').textContent = `8 of the ${res.num_frames} frames sampled from the clip`;
}

// ------------------------------------------------------------------ training curves
function drawCurve() {
  const h = S.info.history, box = $('curve');
  box.textContent = '';
  if (!h.length) { box.append(el('div', { class: 'muted' }, 'No history.json next to the checkpoint.')); return; }
  const acc = S.metric === 'acc';
  $('curve-title').textContent = acc ? 'Accuracy by epoch' : 'Loss by epoch';
  const series = [
    { name: 'Train', color: 'var(--s1)', key: acc ? 'train_acc' : 'train_loss' },
    { name: 'Validation', color: 'var(--s2)', key: acc ? 'val_acc' : 'val_loss' },
  ];
  const W = 560, H = 230, m = { l: 40, r: 118, t: 12, b: 26 };
  const xs = h.map((d) => d.epoch);
  const ymax = acc ? 1 : Math.ceil(Math.max(...h.flatMap((d) => [d.train_loss, d.val_loss])) * 2) / 2;
  const x = (e) => m.l + (e - xs[0]) / Math.max(1, xs[xs.length - 1] - xs[0]) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - v / ymax) * (H - m.t - m.b);
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': $('curve-title').textContent });
  const ticks = acc ? [0, 0.25, 0.5, 0.75, 1] : [0, ymax / 2, ymax];
  for (const t of ticks) {
    s.append(svg('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: 'var(--grid)', 'stroke-width': 1 }));
    const tx = svg('text', { x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end' }); tx.textContent = acc ? `${t * 100}%` : t.toFixed(1); s.append(tx);
  }
  const step = Math.max(1, Math.round(xs.length / 6));
  xs.filter((_, i) => i % step === 0).forEach((e) => {
    const tx = svg('text', { x: x(e), y: H - 6, 'text-anchor': 'middle' }); tx.textContent = e; s.append(tx);
  });
  for (const se of series) {
    const d = h.map((p, i) => `${i ? 'L' : 'M'}${x(p.epoch).toFixed(1)},${y(p[se.key]).toFixed(1)}`).join('');
    s.append(svg('path', { d, fill: 'none', stroke: se.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    const last = h[h.length - 1];
    s.append(svg('circle', { cx: x(last.epoch), cy: y(last[se.key]), r: 4, fill: se.color, stroke: 'var(--panel)', 'stroke-width': 2 }));
  }
  // direct end labels, nudged apart if they would collide
  const ends = series.map((se) => ({ se, v: h[h.length - 1][se.key], yy: y(h[h.length - 1][se.key]) })).sort((a, b) => a.yy - b.yy);
  if (ends[1].yy - ends[0].yy < 14) { const mid = (ends[0].yy + ends[1].yy) / 2; ends[0].yy = mid - 7; ends[1].yy = mid + 7; }
  for (const e of ends) {
    const tx = svg('text', { x: x(h[h.length - 1].epoch) + 10, y: e.yy + 4, class: 'end-label' });
    tx.textContent = `${e.se.name} ${acc ? pct(e.v) : e.v.toFixed(2)}`; s.append(tx);
  }
  if (acc) { // best validation epoch
    const best = h.reduce((a, b) => (b.val_acc > a.val_acc ? b : a));
    s.append(svg('line', { x1: x(best.epoch), x2: x(best.epoch), y1: m.t, y2: H - m.b, stroke: 'var(--faint)', 'stroke-width': 1, 'stroke-dasharray': '3 3', opacity: 0.6 }));
    const tx = svg('text', { x: x(best.epoch) + 5, y: H - m.b - 6 }); tx.textContent = `best val · epoch ${best.epoch}`; s.append(tx);
  }
  // crosshair + tooltip
  const cross = svg('line', { y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-width': 1, visibility: 'hidden' });
  s.append(cross);
  const hit = svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' });
  s.append(hit);
  const tip = $('tip');
  hit.addEventListener('pointermove', (ev) => {
    const r = s.getBoundingClientRect(); const px = (ev.clientX - r.left) * (W / r.width);
    const p = h.reduce((a, b) => (Math.abs(x(b.epoch) - px) < Math.abs(x(a.epoch) - px) ? b : a));
    cross.setAttribute('x1', x(p.epoch)); cross.setAttribute('x2', x(p.epoch)); cross.setAttribute('visibility', 'visible');
    tip.textContent = '';
    tip.append(el('div', { class: 'h' }, `Epoch ${p.epoch}`));
    for (const se of series) {
      const row = el('div', { class: 'row' }); const name = el('span'); const key = el('span', { class: 'key' }); key.style.background = se.color;
      name.append(key, se.name); row.append(name, el('b', {}, acc ? pct(p[se.key]) : p[se.key].toFixed(3))); tip.append(row);
    }
    tip.hidden = false; tip.style.left = `${ev.clientX + 14}px`; tip.style.top = `${ev.clientY + 14}px`;
  });
  hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; });
  box.append(s);
  // table view
  const tbl = el('table'); const head = el('tr');
  ['epoch', 'train acc', 'val acc', 'train loss', 'val loss'].forEach((c) => head.append(el('th', {}, c))); tbl.append(head);
  for (const p of h) {
    const tr = el('tr');
    [p.epoch, pct(p.train_acc), pct(p.val_acc), p.train_loss.toFixed(3), p.val_loss.toFixed(3)].forEach((v) => tr.append(el('td', {}, String(v))));
    tbl.append(tr);
  }
  $('curve-table').textContent = ''; $('curve-table').append(tbl);
}

// ------------------------------------------------------------------ confusion matrix
async function evaluate() {
  const btn = $('btn-eval'); btn.disabled = true; btn.textContent = 'Evaluating…';
  const r = await (await fetch('/api/evaluate')).json();
  btn.textContent = 'Evaluated';
  $('cm-title').textContent = `${r.correct} of ${r.total} clips correct · ${pct(r.accuracy)}`;
  const max = Math.max(...r.matrix.flat());
  const t = el('table', { class: 'cm' });
  const thead = el('thead'); const hr = el('tr'); hr.append(el('th'));
  r.labels.forEach((l) => hr.append(el('th', {}, l))); hr.append(el('th', {}, 'recall')); thead.append(hr); t.append(thead);
  const tb = el('tbody');
  r.labels.forEach((row, i) => {
    const tr = el('tr'); tr.append(el('th', {}, row));
    const total = r.matrix[i].reduce((a, b) => a + b, 0);
    r.matrix[i].forEach((v, j) => {
      const td = el('td', {}, v ? String(v) : '');
      const k = v ? Math.min(RAMP.length - 1, Math.round((v / max) * (RAMP.length - 1))) : -1;
      td.style.background = k < 0 ? 'var(--panel-2)' : RAMP[k];
      td.style.color = k >= 4 ? '#111' : '#fff';
      if (i === j) td.classList.add('diag');
      td.addEventListener('pointermove', (ev) => {
        const tip = $('tip'); tip.textContent = '';
        tip.append(el('div', { class: 'h' }, `true ${row} → predicted ${r.labels[j]}`));
        const rr = el('div', { class: 'row' }); rr.append(el('span', {}, 'clips'), el('b', {}, String(v))); tip.append(rr);
        tip.hidden = false; tip.style.left = `${ev.clientX + 14}px`; tip.style.top = `${ev.clientY + 14}px`;
      });
      td.addEventListener('pointerleave', () => { $('tip').hidden = true; });
      tr.append(td);
    });
    tr.append(el('td', { class: 'acc' }, total ? pct(r.matrix[i][i] / total) : '—'));
    tb.append(tr);
  });
  t.append(tb);
  const cm = $('cm'); cm.textContent = ''; cm.append(t, el('div', { class: 'cm-axis' }, 'rows: true sign · columns: predicted sign'));
}

// ------------------------------------------------------------------ wiring
document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('active', x === b));
  S.metric = b.dataset.metric; drawCurve();
}));
$('btn-eval').addEventListener('click', evaluate);
$('file').addEventListener('change', (e) => { if (e.target.files[0]) classifyFile(e.target.files[0]); });
let drag = 0;
addEventListener('dragenter', (e) => { e.preventDefault(); drag++; $('drop').hidden = false; });
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('dragleave', () => { if (--drag <= 0) { drag = 0; $('drop').hidden = true; } });
addEventListener('drop', (e) => { e.preventDefault(); drag = 0; $('drop').hidden = true; const f = e.dataTransfer.files[0]; if (f) classifyFile(f); });
init();
