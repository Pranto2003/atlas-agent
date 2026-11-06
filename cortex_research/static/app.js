"use strict";
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const NS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const ICON = { planner: "◈", searcher: "⌕", extractor: "⇩", critic: "⟲", synthesizer: "✦", citer: "✓" };
const EXAMPLES = ["How does retrieval-augmented generation reduce LLM hallucinations?", "What are the main challenges of nuclear fusion power?", "How do mRNA vaccines work and what are their limitations?", "What is the current state of quantum error correction?"];

const TL = {};
let depth = "standard", es = null, runId = null, report = null, timer = null, t0 = 0;
let session = localStorage.getItem("cortex_session") || crypto.randomUUID().slice(0, 12);
localStorage.setItem("cortex_session", session);

function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.h); toast.h = setTimeout(() => t.hidden = true, 2200); }

fetch("/api/health").then(r => r.json()).then(h => {
  const p = $("#llm"); p.textContent = h.llm === "offline-extractive" ? "extractive mode · no API key" : h.llm; p.classList.add("ok");
  p.title = "Set ANTHROPIC_API_KEY or OPENAI_API_KEY for generative synthesis";
}).catch(() => { $("#llm").textContent = "backend offline"; $("#llm").classList.add("bad"); });

EXAMPLES.forEach(q => { const b = el("button", "", q); b.type = "button"; b.onclick = () => { $("#q").value = q; $("#q").focus(); }; $("#chips").append(b); });
$("#depth").onclick = e => { const b = e.target.closest("button"); if (!b) return; depth = b.dataset.v; [...$("#depth").children].forEach(x => x.classList.toggle("on", x === b)); };
$("#newSession").onclick = () => { session = crypto.randomUUID().slice(0, 12); localStorage.setItem("cortex_session", session); $("#memory").hidden = true; toast("New session — memory cleared"); };
$("#q").addEventListener("keydown", e => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") $("#form").requestSubmit(); });
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === t));
  ["report", "sources", "plan"].forEach(n => $("#pane-" + n).hidden = n !== t.dataset.t);
});
const showTab = n => document.querySelector(`.tab[data-t=${n}]`).click();

/* ── Graph ── */
const NODES = ["planner", "searcher", "extractor", "critic", "synthesizer", "citer"];
const LABEL = { planner: "Planner", searcher: "Searcher", extractor: "Extractor", critic: "Critic", synthesizer: "Synthesizer", citer: "Citer" };
const SUB = { planner: "decompose", searcher: "wiki + web", extractor: "fetch + quote", critic: "fill gaps", synthesizer: "write", citer: "verify" };
function drawGraph() {
  const g = $("#graph"); g.replaceChildren();
  const xs = NODES.map((_, i) => 42 + i * 87.2), y = 62;
  NODES.slice(1).forEach((n, i) => { const p = svg("path", { d: `M${xs[i] + 24} ${y} L${xs[i + 1] - 24} ${y}`, class: "edge", id: "e-" + n }); g.append(p); });
  NODES.forEach((n, i) => {
    const gr = svg("g", { class: "node", id: "n-" + n, transform: `translate(${xs[i]} ${y})` });
    gr.append(svg("circle", { class: "halo", r: 21 }), svg("circle", { class: "core", r: 21 }));
    const ic = svg("text", { class: "ico", y: 5 }); ic.textContent = ICON[n]; ic.style.textAnchor = "middle";
    const l = svg("text", { y: 44 }); l.textContent = LABEL[n];
    const s = svg("text", { class: "s", y: 57 }); s.textContent = SUB[n];
    const b = svg("text", { class: "badge", y: -32, id: "b-" + n });
    gr.append(ic, l, s, b); g.append(gr);
  });
}
const nodeState = (n, st) => { const e = $("#n-" + n); if (e) e.setAttribute("class", "node " + st); const i = NODES.indexOf(n); if (i > 0) $("#e-" + n).setAttribute("class", "edge " + (st === "running" ? "on" : st === "done" ? "done" : "")); };
const setBadge = (n, t) => { const e = $("#b-" + n); if (e) e.textContent = t; };

/* ── Metrics & log ── */
const M = { sources: 0, fetched: 0, evidence: 0, claims: 0 };
function drawMetrics() {
  const m = $("#metrics"); m.replaceChildren();
  [["Sources", M.sources], ["Fetched", M.fetched], ["Passages", M.evidence], ["Grounding", M.claims]].forEach(([k, v]) => { const d = el("div", "m"); d.append(el("b", "", v), el("span", "", k)); m.append(d); });
}
function log(ev, text, cls = "") {
  const d = el("div", cls); d.append(el("span", "t", ev.t.toFixed(1).padStart(5) + "s"), el("span", "n", ev.node || "run"), document.createTextNode(text));
  const L = $("#log"); L.append(d); L.scrollTop = L.scrollHeight;
}

/* ── Run ── */
$("#form").onsubmit = async e => {
  e.preventDefault();
  const question = $("#q").value.trim(); if (question.length < 8) return;
  if (es) es.close();
  $("#go").disabled = true; $("#go span").textContent = "Researching…";
  $("#work").hidden = false; $("#log").replaceChildren(); report = null;
  for (const k in TL) delete TL[k]; drawTimeline();
  Object.assign(M, { sources: 0, fetched: 0, evidence: 0, claims: "–" }); drawMetrics(); drawGraph();
  ["report", "sources", "plan"].forEach(n => $("#pane-" + n).replaceChildren());
  $("#pane-report").append(el("div", "empty", "Agents are working…")); $("#nsrc").textContent = "0";
  $("#copy").hidden = $("#dl").hidden = true; showTab("report");
  setStatus("running", "running");
  $("#work").scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, depth, session_id: session }) });
    if (!r.ok) throw new Error((await r.json()).detail?.[0]?.msg || "Request rejected");
    runId = (await r.json()).id; listen(runId);
  } catch (err) { finish(false); setStatus("failed", "bad"); toast(err.message); }
};
function setStatus(t, c) { const s = $("#status"); s.textContent = t; s.className = "pill " + (c === "running" ? "run" : c); }
function listen(id) {
  t0 = performance.now(); clearInterval(timer);
  timer = setInterval(() => $("#clock").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s", 100);
  es = new EventSource(`/api/runs/${id}/events`);
  es.onmessage = m => handle(JSON.parse(m.data));
  es.addEventListener("end", () => { es.close(); clearInterval(timer); });
  es.onerror = () => { if (es.readyState === 2) clearInterval(timer); };
}
function finish(ok) { $("#go").disabled = false; $("#go span").textContent = "Run research"; clearInterval(timer); }
function handle(ev) {
  switch (ev.type) {
    case "run_start": log(ev, `LLM: ${ev.llm} · depth: ${ev.depth}`); break;
    case "node_start": TL[ev.node] = { s: ev.t }; nodeState(ev.node, "running"); log(ev, "started"); break;
    case "node_end": if (TL[ev.node]) TL[ev.node].e = ev.t; drawTimeline(); nodeState(ev.node, "done"); log(ev, `done in ${ev.ms}ms`, "ok"); break;
    case "node_error": nodeState(ev.node, "error"); log(ev, ev.message, "err"); break;
    case "plan": renderPlan(ev.subqs); setBadge("searcher", "×" + ev.subqs.length); setBadge("extractor", "∥"); log(ev, `${ev.subqs.length} sub-questions`); break;
    case "sources": M.sources = ev.sources.length; M.fetched = ev.sources.filter(s => s.status === "fetched").length; drawMetrics(); renderSources(ev.sources); break;
    case "metric": M.evidence = ev.evidence; drawMetrics(); break;
    case "log": log(ev, ev.message); if (/recalled/.test(ev.message)) { $("#memory").hidden = false; $("#memory").textContent = "🧠 " + ev.message; } break;
    case "warn": log(ev, ev.message, "warn"); break;
    case "retry": log(ev, ev.message, "retry"); break;
    case "report": M.claims = Math.round(ev.stats.grounding * 100) + "%"; drawMetrics(); setStatus("complete", "ok"); finish(true); fetchReport(); break;
    case "run_failed": setStatus("failed", "bad"); finish(false); log(ev, ev.message, "err");
      $("#pane-report").replaceChildren(el("div", "fail", "Research failed: " + ev.message)); break;
  }
}
async function fetchReport() { const r = await fetch(`/api/runs/${runId}`); report = (await r.json()).report; renderReport(report); }

/* ── Rendering (DOM only — no innerHTML, so source text can never inject markup) ── */
function claimLi(c, i) {
  const li = el("li"); li.style.animationDelay = i * 40 + "ms";
  li.append(el("span", "", c.text + " "));
  c.cites.forEach(n => {
    const b = el("button", "cite", n); b.type = "button";
    const q = c.quotes.find(x => x.n === n);
    b.onmouseenter = ev => showPop(ev, q ? q.quote : ""); b.onmouseleave = hidePop;
    b.onclick = () => { showTab("sources"); const s = $("#src-" + n); s.scrollIntoView({ behavior: "smooth", block: "center" }); s.classList.add("hl"); setTimeout(() => s.classList.remove("hl"), 1800); };
    li.append(b);
  });
  return li;
}
function renderReport(r) {
  const p = $("#pane-report"); p.replaceChildren();
  const st = r.stats, v = el("div", "verdict");
  [[`${st.verified}/${st.claims} claims verified`], [`${st.sources_cited} sources`], [`${r.elapsed}s`], [r.mode]].forEach(([t, c]) => v.append(el("span", "pill " + (c || ""), t)));
  const ring = el("div", "ring", Math.round(st.grounding * 100) + "%"); ring.style.setProperty("--p", st.grounding * 360 + "deg");
  v.prepend(ring); st.method && v.append(el("span", "pill", "verifier: " + st.method));
  const wrap = el("div", "rep"); wrap.append(v);
  wrap.append(el("h2", "", "Key findings")); const k = el("ul", "key"); r.summary.forEach((c, i) => k.append(claimLi(c, i))); wrap.append(k);
  r.sections.forEach(s => { wrap.append(el("h2", "", s.heading)); const u = el("ul"); s.claims.forEach((c, i) => u.append(claimLi(c, i))); wrap.append(u); });
  p.append(wrap);
  renderSources(r.sources.map(s => ({ ...s })), true);
  $("#copy").hidden = false; $("#dl").hidden = false;
  $("#copy").onclick = () => navigator.clipboard.writeText(r.markdown).then(() => toast("Markdown copied"));
  $("#dl").href = `/api/runs/${runId}/report.md`; $("#dl").download = "research-report.md";
}
function renderSources(list, cited = false) {
  const p = $("#pane-sources"); p.replaceChildren(); $("#nsrc").textContent = list.length;
  list.forEach((s, i) => {
    const d = el("div", "src"); d.id = "src-" + (s.n || ""); const n = el("div", "n", cited ? s.n : i + 1);
    const b = el("div"); const a = el("a", "", s.title); a.href = s.url; a.target = "_blank"; a.rel = "noopener noreferrer";
    b.append(a, el("span", "tag", s.origin), el("span", "tag " + s.status, s.status), el("small", "", s.url));
    d.append(n, b); p.append(d);
  });
}
function renderPlan(subqs) {
  const p = $("#pane-plan"); p.replaceChildren(); const u = el("ul", "plan rep");
  subqs.forEach(s => { const li = el("li"); li.append(el("b", "", s.id + " · " + s.heading), el("br"), el("code", "", s.query)); u.append(li); });
  p.append(u);
}
const pop = el("div", "pop"); pop.hidden = true; document.body.append(pop);
function showPop(ev, q) { if (!q) return; pop.replaceChildren(el("q", "", q)); pop.hidden = false; const r = ev.target.getBoundingClientRect(); pop.style.left = Math.min(r.left, innerWidth - 400) + "px"; pop.style.top = r.bottom + 8 + "px"; }
function hidePop() { pop.hidden = true; }
function drawTimeline() {
  const t = $("#timeline"); t.replaceChildren();
  const rows = NODES.filter(n => TL[n]); const end = Math.max(1, ...rows.map(n => TL[n].e ?? TL[n].s));
  rows.forEach(n => {
    const r = el("div", "tl"); r.append(el("span", "", LABEL[n]));
    const bar = el("div", "bar"); const f = el("i"); f.style.left = (TL[n].s / end * 100) + "%"; f.style.width = Math.max(1.5, ((TL[n].e ?? end) - TL[n].s) / end * 100) + "%"; bar.append(f);
    r.append(bar, el("em", "", ((TL[n].e ?? end) - TL[n].s).toFixed(1) + "s")); t.append(r);
  });
}
fetch("/api/model").then(r => r.json()).then(m => {
  const box = $("#model"); if (!m.loaded) { box.append(el("div", "dim", "Learned verifier not trained — using lexical fallback. Run: python -m cortex_research.train_verifier")); return; }
  const row = (k, a, b) => { const d = el("div", "mrow"); d.append(el("span", "", k), el("b", "", a), el("s", "", b)); return d; };
  box.append(row("F1", m.learned.f1, m.baseline_overlap.f1), row("Accuracy", m.learned.accuracy, m.baseline_overlap.accuracy), row("Precision", m.learned.precision, m.baseline_overlap.precision));
  box.append(el("p", "dim small", `${m.model} · ${m.train_pairs.toLocaleString()} train / ${m.test_pairs.toLocaleString()} test pairs from ${m.articles} Wikipedia articles, split by article. Synthetic benchmark (negation, antonym, number, entity, add-on perturbations) — not a general NLI claim.`));
}).catch(() => {});
drawGraph(); drawMetrics();
