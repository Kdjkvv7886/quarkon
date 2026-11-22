/* Quarkon local interface: no build step, no external dependencies.
   Charts are hand-drawn SVG so the UI works on an air-gapped machine. */
(() => {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const params = new URLSearchParams(location.search);
  const token = params.get("token") || "";

  const state = {
    bootstrap: null,
    labels: {},
    templates: [],
    selectedTemplate: null,
    job: null,
    stats: null,
    activeTab: "overview",
    numberFormat: new Intl.NumberFormat(),
    elapsedTimer: null,
    startedAt: null,
    lastOk: true,
  };

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (id) => document.getElementById(id);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

  function fmt(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? state.numberFormat.format(number) : "0";
  }

  function duration(seconds) {
    if (seconds === null || seconds === undefined) return "";
    const total = Math.max(0, Math.round(Number(seconds)));
    if (total < 60) return `${total}s`;
    const minutes = Math.floor(total / 60);
    return `${minutes}m ${String(total % 60).padStart(2, "0")}s`;
  }

  function toast(message, kind = "info") {
    const node = $("toast");
    node.textContent = message;
    node.classList.toggle("error", kind === "error");
    node.classList.add("visible");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => node.classList.remove("visible"), 4200);
  }

  // ------------------------------------------------------------------- api
  async function api(path, options = {}) {
    const headers = { "X-Quarkon-Token": token, "Content-Type": "application/json" };
    let response;
    try {
      response = await fetch(path, { ...options, headers });
    } catch (error) {
      setConnection("offline");
      throw new Error("The interface cannot reach the local server. Is Quarkon still running?");
    }
    const payload = await response.json().catch(() => ({}));
    setConnection("online");
    if (!response.ok) {
      const detail = payload.error || {};
      throw new Error(detail.message || detail.hint || `HTTP ${response.status}`);
    }
    return payload;
  }

  function setConnection(mode) {
    const pill = $("connection");
    const text = $("connection-text");
    pill.classList.remove("pill-ok", "pill-busy", "pill-bad", "pill-muted");
    if (mode === "offline") {
      pill.classList.add("pill-bad");
      text.textContent = "Offline";
      state.lastOk = false;
      return;
    }
    state.lastOk = true;
    if (state.job && state.job.status === "running") {
      pill.classList.add("pill-busy");
      text.textContent = state.labels["status.running"] || "Working...";
    } else if (state.job && state.job.status === "failed") {
      pill.classList.add("pill-bad");
      text.textContent = state.labels["status.failed"] || "Stopped with a problem";
    } else {
      pill.classList.add("pill-ok");
      text.textContent = state.labels["status.ready"] || "Ready";
    }
  }

  // ------------------------------------------------------------------ i18n
  function applyLabels(labels) {
    state.labels = labels || {};
    $$("[data-i18n]").forEach((node) => {
      const key = node.getAttribute("data-i18n");
      const value = state.labels[key];
      if (!value) return;
      if (node.tagName === "INPUT" || node.tagName === "TEXTAREA") {
        node.placeholder = value;
      } else {
        node.textContent = value;
      }
    });
    document.title = "Quarkon";
  }

  // ------------------------------------------------------------------ tabs
  const TABS = ["overview", "setup", "run", "help"];

  function setTab(name, options = {}) {
    const tab = TABS.includes(name) ? name : "overview";
    state.activeTab = tab;
    $$(".tab").forEach((button) => {
      const active = button.dataset.tab === tab;
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.classList.toggle("active", active);
    });
    $$(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${tab}`));
    if (options.push !== false && location.hash !== `#${tab}`) {
      history.replaceState(null, "", `#${tab}`);
    }
    if (tab === "overview") loadStats();
  }

  function wireTabs() {
    $$(".tab").forEach((button) => {
      button.addEventListener("click", () => setTab(button.dataset.tab));
    });
    document.addEventListener("keydown", (event) => {
      if (event.target.matches("input, textarea, select")) return;
      const index = Number(event.key);
      if (index >= 1 && index <= TABS.length) setTab(TABS[index - 1]);
      if (event.key.toLowerCase() === "r") loadStats();
      if (event.key === "Escape") $("toast").classList.remove("visible");
    });
    window.addEventListener("hashchange", () => setTab(location.hash.replace("#", ""), { push: false }));
  }

  // --------------------------------------------------------------- templates
  function renderTemplates(templates) {
    state.templates = templates || [];
    const host = $("templates");
    host.innerHTML = "";
    state.templates.forEach((template) => {
      const button = document.createElement("button");
      button.className = "template";
      button.type = "button";
      button.dataset.id = template.id;
      button.setAttribute("role", "listitem");
      button.innerHTML = `<div class="id">${esc(template.id)}</div>
        <div class="industry">${esc(template.industry)} · ${fmt(template.recommended_samples)} ${esc("examples")}</div>`;
      button.addEventListener("click", () => selectTemplate(template));
      host.appendChild(button);
    });
  }

  function selectTemplate(template) {
    state.selectedTemplate = template.id;
    $$(".template").forEach((node) => node.classList.toggle("active", node.dataset.id === template.id));
    $("objective").value = template.objective;
    setSamples(template.recommended_samples);
  }

  function setSamples(value) {
    const samples = Math.max(50, Math.min(10000, Number(value) || 300));
    $("samples").value = String(Math.min(2000, samples));
    $("samples-number").value = String(samples);
    refreshCost();
  }

  // ------------------------------------------------------------------- cost
  function refreshCost() {
    const samples = Number($("samples-number").value || 300);
    api("/api/plan", { method: "POST", body: JSON.stringify({ samples }) })
      .then((estimate) => {
        $("cost").textContent = "$" + Number(estimate.total_cost_usd).toFixed(2);
        $("cost-meta").textContent = `≈${fmt(estimate.attempts_estimate)} attempts · ${fmt(estimate.completion_tokens)} completion tokens`;
      })
      .catch(() => {
        $("cost").textContent = "—";
        $("cost-meta").textContent = "";
      });
  }

  // ----------------------------------------------------------------- doctor
  function renderDoctor(report) {
    const host = $("doctor");
    host.innerHTML = "";
    const findings = report.findings || [];
    const counts = { ok: 0, warning: 0, blocker: 0 };
    findings.forEach((finding) => {
      const severity = finding.severity || "ok";
      if (severity in counts) counts[severity] += 1;
      const node = document.createElement("div");
      node.className = "finding " + severity;
      node.innerHTML = `<div class="t">${esc(finding.title || finding.code)}</div>
        ${finding.meaning && severity !== "ok" ? `<div class="m">${esc(finding.meaning)}</div>` : ""}
        ${finding.fix ? `<div class="f">${esc("How to fix: " + finding.fix)}</div>` : ""}`;
      host.appendChild(node);
    });
    $("doctor-meta").textContent = report.duration_ms
      ? `${fmt(counts.ok)} ok · ${fmt(counts.warning)} warnings · ${fmt(counts.blocker)} blockers · ${fmt(report.duration_ms)} ms`
      : "";
  }

  // ------------------------------------------------------------------ state
  function renderState(info) {
    const gguf = info.gguf || [];
    $("kpi-models").textContent = fmt(gguf.length);
    if (info.job) renderJob(info.job);
    $("project-meta").textContent = info.has_project
      ? `${fmt(info.dataset_lines)} examples · ${info.has_adapter ? "adapter ready" : "no adapter"}`
      : "No project yet";
  }

  // ------------------------------------------------------------------ stats
  async function loadStats() {
    try {
      const stats = await api("/api/stats");
      state.stats = stats;
      renderDashboard(stats);
    } catch (error) {
      if (!state.lastOk) return;
      toast(String(error.message), "error");
    }
  }

  function renderDashboard(stats) {
    const accepted = Number(stats.accepted || 0);
    const target = stats.target_samples === null || stats.target_samples === undefined
      ? null
      : Number(stats.target_samples);
    const ratio = stats.ratio === null || stats.ratio === undefined ? null : Number(stats.ratio);

    animateNumber($("kpi-accepted"), accepted);
    animateNumber($("kpi-tokens"), (stats.tokens || {}).total || 0);
    animateNumber($("kpi-rejected"), stats.rejected || 0);
    animateNumber($("kpi-models"), ((stats.artefacts || {}).gguf || []).length);
    $("kpis").setAttribute("aria-busy", "false");
    renderSparkline(stats.activity || []);

    const projectName = String(stats.project_dir || "").split(/[\\/]/).filter(Boolean).pop() || "Quarkon";
    $("hero-eyebrow").textContent = projectName;
    $("hero-title").textContent = accepted > 0
      ? "Your model is taking shape"
      : "Your project at a glance";
    const adapter = (stats.artefacts || {}).has_adapter;
    $("hero-sub").textContent = accepted > 0
      ? `${fmt(accepted)} accepted examples${target ? ` of ${fmt(target)}` : ""} · ${adapter ? "adapter trained" : "training not started yet"} · everything stays on this computer.`
      : "Track how much training data is ready and whether this computer can finish the run.";

    renderDonut($("progress-donut"), ratio, accepted, target);
    $("progress-caption").textContent = state.labels["dash.progress"] || "Training data progress";
    $("progress-value").textContent = target
      ? `${fmt(accepted)} / ${fmt(target)}`
      : `${fmt(accepted)} accepted`;

    const empty = accepted === 0;
    $("dash-empty").classList.toggle("hidden", !empty);
    $("charts").classList.toggle("hidden", empty);

    renderReasoning(stats.reasoning || {});
    renderActivity(stats.activity || []);
    renderTiers(stats.tiers || []);
    renderLatency(stats.latency || {});
    renderPhases(stats.phases || []);
  }

  // ------------------------------------------------------------------- SVG
  function svg(tag, attrs = {}) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => {
      if (value !== null && value !== undefined) node.setAttribute(key, String(value));
    });
    return node;
  }

  function defsGradient(id, top, bottom) {
    const defs = svg("defs");
    const gradient = svg("linearGradient", { id, x1: "0", y1: "0", x2: "0", y2: "1" });
    gradient.appendChild(svg("stop", { offset: "0", "stop-color": top }));
    gradient.appendChild(svg("stop", { offset: "1", "stop-color": bottom }));
    defs.appendChild(gradient);
    return defs;
  }

  function renderDonut(host, ratio, accepted, target) {
    host.innerHTML = "";
    const percent = ratio === null ? 0 : Math.max(0, Math.min(1, ratio));
    const size = 172;
    const radius = 68;
    const circumference = 2 * Math.PI * radius;
    const root = svg("svg", { viewBox: `0 0 ${size} ${size}`, role: "img", "aria-label": `${Math.round(percent * 100)}%` });
    const defs = svg("defs");
    const gradient = svg("linearGradient", { id: "donut-gradient", x1: "0", y1: "0", x2: "1", y2: "1" });
    gradient.appendChild(svg("stop", { offset: "0", "stop-color": "#3b82f6" }));
    gradient.appendChild(svg("stop", { offset: "0.6", "stop-color": "#8b5cf6" }));
    gradient.appendChild(svg("stop", { offset: "1", "stop-color": "#22d3ee" }));
    defs.appendChild(gradient);
    root.appendChild(defs);
    root.appendChild(svg("circle", { class: "track", cx: size / 2, cy: size / 2, r: radius }));
    root.appendChild(svg("circle", {
      class: "value",
      cx: size / 2,
      cy: size / 2,
      r: radius,
      "stroke-dasharray": circumference.toFixed(2),
      "stroke-dashoffset": (circumference * (1 - percent)).toFixed(2),
      transform: `rotate(-90 ${size / 2} ${size / 2})`,
    }));
    host.appendChild(root);

    const center = document.createElement("div");
    center.className = "donut-center";
    center.innerHTML = `<strong>${Math.round(percent * 100)}%</strong>
      <span>${target ? fmt(accepted) + " / " + fmt(target) : fmt(accepted)}</span>`;
    host.appendChild(center);
  }

  function renderBars(host, buckets, options = {}) {
    host.innerHTML = "";
    const width = 560;
    const height = 210;
    const pad = { left: 10, right: 10, top: 18, bottom: 28 };
    const innerW = width - pad.left - pad.right;
    const innerH = height - pad.top - pad.bottom;
    const counts = buckets.map((bucket) => Number(bucket.count || 0));
    const max = Math.max(1, ...counts);
    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
    root.appendChild(defsGradient("bar-gradient-" + options.id, options.top || "#3b82f6", options.bottom || "#8b5cf6"));

    [0.25, 0.5, 0.75, 1].forEach((fraction) => {
      const y = pad.top + innerH * (1 - fraction);
      root.appendChild(svg("line", { class: "grid-line", x1: pad.left, x2: width - pad.right, y1: y, y2: y }));
    });

    const gap = 6;
    const barW = Math.max(4, (innerW - gap * (buckets.length - 1)) / Math.max(1, buckets.length));
    buckets.forEach((bucket, index) => {
      const count = counts[index];
      const h = (count / max) * innerH;
      const x = pad.left + index * (barW + gap);
      const y = pad.top + innerH - h;
      const rect = svg("rect", {
        class: "bar",
        x,
        y,
        width: barW,
        height: Math.max(count > 0 ? 3 : 0, h),
        rx: 5,
        fill: `url(#bar-gradient-${options.id})`,
      });
      rect.appendChild(svg("title")).textContent = `${options.unit || ""}${fmt(bucket.from)}–${fmt(bucket.to)}: ${fmt(count)}`;
      root.appendChild(rect);
      if (count > 0) {
        const label = svg("text", {
          class: "value-label",
          x: x + barW / 2,
          y: Math.max(12, y - 6),
          "text-anchor": "middle",
        });
        label.textContent = fmt(count);
        root.appendChild(label);
      }
      const tick = svg("text", {
        class: "tick",
        x: x + barW / 2,
        y: height - 8,
        "text-anchor": "middle",
      });
      tick.textContent = options.tick ? options.tick(bucket, index) : fmt(bucket.from);
      root.appendChild(tick);
    });
    host.appendChild(root);
  }

  function renderReasoning(reasoning) {
    const histogram = reasoning.histogram || [];
    renderBars($("chart-reasoning"), histogram, {
      id: "reasoning",
      unit: "chars ",
      tick: (bucket, index) => (index === histogram.length - 1 ? ">" + fmt(bucket.from) : fmt(bucket.from)),
    });
    $("reasoning-meta").textContent = reasoning.median
      ? `median ${fmt(reasoning.median)} chars · p90 ${fmt(reasoning.p90)} chars`
      : "";
  }

  function renderLatency(latency) {
    const histogram = latency.histogram || [];
    renderBars($("chart-latency"), histogram, {
      id: "latency",
      unit: "ms ",
      top: "#22d3ee",
      bottom: "#3b82f6",
      tick: (bucket, index) => (index === histogram.length - 1 ? ">" + fmt(bucket.from) : fmt(bucket.from)),
    });
    $("latency-meta").textContent = latency.p50_ms
      ? `p50 ${fmt(latency.p50_ms)} ms · p95 ${fmt(latency.p95_ms)} ms`
      : "";
  }

  function animateNumber(node, value) {
    const target = Number(value || 0);
    if (reduceMotion) {
      node.textContent = fmt(target);
      return;
    }
    const from = Number(String(node.textContent).replace(/[^0-9.-]/g, "")) || 0;
    const start = performance.now();
    const steps = 520;
    const frame = (now) => {
      const progress = Math.min(1, (now - start) / steps);
      const eased = 1 - Math.pow(1 - progress, 3);
      node.textContent = fmt(Math.round(from + (target - from) * eased));
      if (progress < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  function renderSparkline(activity) {
    const host = $("hero-spark");
    if (!host) return;
    host.innerHTML = "";
    const counts = activity.map((day) => Number(day.count || 0));
    if (counts.length < 2) return;
    const width = 190;
    const height = 44;
    const max = Math.max(1, ...counts);
    const step = width / (counts.length - 1);
    const points = counts.map((count, index) => {
      const x = step * index;
      const y = height - 4 - (count / max) * (height - 10);
      return [x, y];
    });
    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "activity" });
    const defs = svg("defs");
    const line = svg("linearGradient", { id: "spark-gradient", x1: "0", y1: "0", x2: "1", y2: "0" });
    line.appendChild(svg("stop", { offset: "0", "stop-color": "#3b82f6" }));
    line.appendChild(svg("stop", { offset: "1", "stop-color": "#22d3ee" }));
    const fill = svg("linearGradient", { id: "spark-fill", x1: "0", y1: "0", x2: "0", y2: "1" });
    fill.appendChild(svg("stop", { offset: "0", "stop-color": "rgba(59,130,246,.35)" }));
    fill.appendChild(svg("stop", { offset: "1", "stop-color": "rgba(59,130,246,0)" }));
    defs.appendChild(line);
    defs.appendChild(fill);
    root.appendChild(defs);
    const path = points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    root.appendChild(svg("path", { class: "spark-fill", d: `${path} L${width},${height} L0,${height} Z` }));
    root.appendChild(svg("path", { class: "spark-line", d: path }));
    const last = points[points.length - 1];
    root.appendChild(svg("circle", { cx: last[0], cy: last[1], r: 2.4, fill: "#7dd3fc" }));
    host.appendChild(root);
    const total = counts.reduce((sum, value) => sum + value, 0);
    $("hero-spark-label").textContent = `${fmt(total)} examples · last 14 days`;
  }

  function renderActivity(activity) {
    const host = $("chart-activity");
    host.innerHTML = "";
    const width = 560;
    const height = 210;
    const pad = { left: 12, right: 12, top: 20, bottom: 30 };
    const innerW = width - pad.left - pad.right;
    const innerH = height - pad.top - pad.bottom;
    const counts = activity.map((day) => Number(day.count || 0));
    const max = Math.max(1, ...counts);
    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
    const defs = svg("defs");
    const fill = svg("linearGradient", { id: "area-fill-gradient", x1: "0", y1: "0", x2: "0", y2: "1" });
    fill.appendChild(svg("stop", { offset: "0", "stop-color": "rgba(59, 130, 246, 0.45)" }));
    fill.appendChild(svg("stop", { offset: "1", "stop-color": "rgba(59, 130, 246, 0)" }));
    const stroke = svg("linearGradient", { id: "area-gradient", x1: "0", y1: "0", x2: "1", y2: "0" });
    stroke.appendChild(svg("stop", { offset: "0", "stop-color": "#3b82f6" }));
    stroke.appendChild(svg("stop", { offset: "1", "stop-color": "#22d3ee" }));
    defs.appendChild(fill);
    defs.appendChild(stroke);
    root.appendChild(defs);

    const step = activity.length > 1 ? innerW / (activity.length - 1) : innerW;
    const points = counts.map((count, index) => {
      const x = pad.left + step * index;
      const y = pad.top + innerH * (1 - count / max);
      return [x, y];
    });
    const line = points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const area = `${line} L${(pad.left + step * (activity.length - 1)).toFixed(1)},${pad.top + innerH} L${pad.left},${pad.top + innerH} Z`;
    root.appendChild(svg("path", { class: "area-fill", d: area }));
    root.appendChild(svg("path", { class: "area-line", d: line }));
    points.forEach(([x, y], index) => {
      if (counts[index] === 0) return;
      root.appendChild(svg("circle", { cx: x, cy: y, r: 2.6, fill: "#93c5fd" }));
    });
    activity.forEach((day, index) => {
      if (activity.length > 7 && index % 2 === 1) return;
      const tick = svg("text", {
        class: "tick",
        x: pad.left + step * index,
        y: height - 8,
        "text-anchor": index === 0 ? "start" : index === activity.length - 1 ? "end" : "middle",
      });
      tick.textContent = String(day.date || "").slice(5);
      root.appendChild(tick);
    });
    host.appendChild(root);
    const total = counts.reduce((sum, value) => sum + value, 0);
    $("activity-meta").textContent = `${fmt(total)} examples in the last ${activity.length || 0} days`;
  }

  function renderTiers(tiers) {
    const host = $("chart-tiers");
    host.innerHTML = "";
    const total = tiers.reduce((sum, item) => sum + Number(item.count || 0), 0);
    if (total === 0) {
      host.innerHTML = `<p class="hint">${esc(state.labels["dash.empty"] || "No data yet.")}</p>`;
      return;
    }
    const palette = ["#3b82f6", "#8b5cf6", "#22d3ee", "#f59e0b"];
    const bar = document.createElement("div");
    bar.className = "tier-bar";
    tiers.forEach((item, index) => {
      const count = Number(item.count || 0);
      if (count === 0) return;
      const segment = document.createElement("span");
      segment.style.width = `${(count / total) * 100}%`;
      segment.style.background = palette[index % palette.length];
      segment.title = `${item.tier}: ${fmt(count)}`;
      bar.appendChild(segment);
    });
    host.appendChild(bar);

    const rows = document.createElement("div");
    rows.className = "tier-rows";
    tiers.forEach((item, index) => {
      const count = Number(item.count || 0);
      const share = (count / total) * 100;
      const row = document.createElement("div");
      row.className = "tier-row";
      row.innerHTML = `<span class="tier-label">${esc(item.tier)}</span>
        <span class="tier-track"><span style="width:${share.toFixed(1)}%;background:${palette[index % palette.length]}"></span></span>
        <span class="tier-count">${fmt(count)}</span>`;
      rows.appendChild(row);
    });
    host.appendChild(rows);
    $("tiers-meta").textContent = `${fmt(total)} examples classified`;
  }

  function renderPhases(phases) {
    const host = $("chart-phases");
    host.innerHTML = "";
    phases.forEach((phase) => {
      const row = document.createElement("div");
      row.className = "phase-row";
      row.dataset.status = phase.status || "pending";
      const label = phase.key === "synthesize"
        ? (state.labels["action.synthesize"] || "Generate training data")
        : phase.key === "train"
          ? (state.labels["action.train"] || "Train the model")
          : (state.labels["action.export"] || "Package for your servers");
      const statusText = phase.status === "pending" ? "not started yet" : phase.status;
      row.innerHTML = `<span class="phase-dot" aria-hidden="true"></span>
        <span class="phase-name"><strong>${esc(label)}</strong><span>${esc(statusText)}${phase.duration_s ? " · " + esc(duration(phase.duration_s)) : ""}</span></span>
        <span class="card-meta">${esc(phase.key)}</span>`;
      host.appendChild(row);
    });
    $("phases-meta").textContent = "";
  }

  // -------------------------------------------------------------------- run
  function startPhase(phase) {
    $("error-card").classList.add("hidden");
    api("/api/run", { method: "POST", body: JSON.stringify({ phase }) })
      .then((payload) => {
        state.job = payload.job;
        state.startedAt = Date.now();
        renderJob(payload.job);
        startElapsed();
        pollJob();
        toast(`Started: ${phase}`);
      })
      .catch((error) => {
        $("error-card").classList.remove("hidden");
        $("error-meaning").textContent = String(error.message);
        $("error-fix").textContent = "Create a project first in the Setup tab, then try again.";
      });
  }

  function startElapsed() {
    stopElapsed();
    state.elapsedTimer = setInterval(() => {
      if (!state.job || state.job.status !== "running") return stopElapsed();
      const seconds = (Date.now() - (state.startedAt || Date.now())) / 1000;
      $("job-elapsed").textContent = duration(seconds);
    }, 1000);
  }

  function stopElapsed() {
    if (state.elapsedTimer) clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }

  function formatEvent(event) {
    const phase = event.phase || event.event || "event";
    const detail = Object.entries(event)
      .filter(([key]) => !["phase", "event", "ts"].includes(key))
      .map(([key, value]) => `${key}=${value}`)
      .join(" ");
    const stamp = new Date((event.ts || 0) * 1000).toLocaleTimeString();
    return `${stamp}  ${phase}  ${detail}`.trim();
  }

  function levelOf(line) {
    const lower = line.toLowerCase();
    if (/(error|fail|traceback|exception|blocked)/.test(lower)) return "bad";
    if (/(warn|retry|slow|skipped)/.test(lower)) return "warn";
    if (/(done|ok|accept|finished|saved|complete)/.test(lower)) return "ok";
    return "";
  }

  function renderJob(job) {
    state.job = job;
    const running = job.status === "running";
    const log = $("log");
    log.innerHTML = "";
    (job.events || []).forEach((event) => {
      const line = formatEvent(event);
      const span = document.createElement("span");
      span.className = levelOf(line);
      span.textContent = line + "\n";
      log.appendChild(span);
    });
    if (!(job.events || []).length) log.textContent = "Run a phase to see live progress here.";
    log.scrollTop = log.scrollHeight;

    ["run-synthesize", "run-train", "run-export"].forEach((id) => { $(id).disabled = running; });
    $$(".phase-card").forEach((card) => {
      card.dataset.state = ""; // reset every card first, then highlight the running phase
    });
    const phaseNames = { synthesize: "synthesize", train: "train", export: "export" };
    const activePhase = Object.keys(phaseNames).find((key) => (job.events || []).some(
      (event) => (event.phase || event.event) === key));
    const activeCard = activePhase ? document.querySelector(`.phase-card[data-phase="${activePhase}"]`) : null;
    if (activeCard) activeCard.dataset.state = running ? "active" : job.status === "failed" ? "failed" : "done";

    $("job-title").textContent = running
      ? (state.labels["status.running"] || "Working...")
      : job.status === "done"
        ? (state.labels["status.done"] || "Finished")
        : job.status === "failed"
          ? (state.labels["status.failed"] || "Stopped with a problem")
          : (state.labels["status.running"] || "Working...");
    $("job-pill").classList.remove("pill-busy", "pill-ok", "pill-bad");
    $("job-pill").classList.add(running ? "pill-busy" : job.status === "failed" ? "pill-bad" : "pill-ok");
    $("job-pill-text").textContent = running
      ? (state.labels["status.running"] || "Working...")
      : job.status === "failed"
        ? (state.labels["status.failed"] || "Stopped with a problem")
        : "Idle";
    $("job-progress").classList.toggle("running", running);
    $("job-progress").querySelector("span").style.width = running ? "38%" : "100%";

    if (job.error) {
      $("error-card").classList.remove("hidden");
      $("error-meaning").textContent = `${job.error.title || ""}: ${job.error.meaning || ""}`;
      $("error-fix").textContent = job.error.fix || "";
    } else if (!running) {
      $("error-card").classList.add("hidden");
    }
    if (!running && job.result) {
      $("next-step").textContent = job.status === "done"
        ? "Continue with the next button, or open the audit pack for review."
        : "Fix the problem above and press the same button again; finished work is kept.";
    }
    document.body.classList.toggle("job-live", running);
    if (!running) {
      stopElapsed();
      $("job-elapsed").textContent = "";
      setConnection("online");
      loadStats();
    }
  }

  async function pollJob() {
    if (!state.job || state.job.status !== "running") return;
    try {
      const info = await api("/api/state");
      renderState(info);
      if (info.job) renderJob(info.job);
    } catch (error) { /* transient; keep polling */ }
    if (state.job && state.job.status === "running") setTimeout(pollJob, 1500);
  }

  function refreshState() {
    return api("/api/state").then(renderState).catch(() => {});
  }

  // ------------------------------------------------------------------ audit
  async function exportAudit(format) {
    try {
      const result = await api("/api/audit/export", { method: "POST", body: JSON.stringify({ format }) });
      $("audit-result").textContent = "Saved: " + result.path;
      toast("Audit pack written");
    } catch (error) {
      $("audit-result").textContent = "Problem: " + String(error.message);
      toast(String(error.message), "error");
    }
  }

  // ------------------------------------------------------------------- wire
  function wire() {
    wireTabs();

    $("samples").addEventListener("input", (event) => {
      $("samples-number").value = event.target.value;
      refreshCost();
    });
    $("samples-number").addEventListener("change", (event) => setSamples(event.target.value));

    $("create").addEventListener("click", async () => {
      const payload = {
        template: state.selectedTemplate,
        objective: $("objective").value.trim(),
        samples: Number($("samples-number").value || 300),
        language: $("data-language").value,
      };
      $("setup-result").textContent = "Creating project...";
      try {
        const result = await api("/api/setup", { method: "POST", body: JSON.stringify(payload) });
        $("setup-result").textContent = `Project created:\n${result.project_dir}\nConfig: ${result.config_path}`;
        toast("Project created");
        await refreshState();
        await loadStats();
      } catch (error) {
        $("setup-result").textContent = "Problem: " + String(error.message);
        toast(String(error.message), "error");
      }
    });

    const runDoctor = async () => {
      $("doctor").innerHTML = '<div class="finding">Checking...</div>';
      try { renderDoctor(await api("/api/doctor")); }
      catch (error) { $("doctor").innerHTML = `<div class="finding blocker">${esc(error.message)}</div>`; }
    };
    $("check").addEventListener("click", runDoctor);
    $("doctor-run").addEventListener("click", runDoctor);

    $("run-synthesize").addEventListener("click", () => startPhase("synthesize"));
    $("run-train").addEventListener("click", () => startPhase("train"));
    $("run-export").addEventListener("click", () => startPhase("export"));
    $("cancel").addEventListener("click", () => api("/api/cancel", { method: "POST", body: "{}" })
      .then(() => toast("Cancellation requested"))
      .catch(() => {}));

    $("hero-run").addEventListener("click", () => { setTab("run"); startPhase("synthesize"); });
    $("empty-run").addEventListener("click", () => { setTab("run"); startPhase("synthesize"); });
    $("hero-setup").addEventListener("click", () => setTab("setup"));
    $("hero-refresh").addEventListener("click", () => { loadStats(); toast("Refreshed"); });

    $("audit-json").addEventListener("click", () => exportAudit("json"));
    $("audit-csv").addEventListener("click", () => exportAudit("csv"));

    $("copy-log").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText($("log").innerText || "");
        toast("Log copied");
      } catch (error) {
        toast("Clipboard is not available", "error");
      }
    });

    $("locale-select").addEventListener("change", async (event) => {
      await api("/api/locale", { method: "POST", body: JSON.stringify({ locale: event.target.value }) });
      location.reload();
    });
  }

  // -------------------------------------------------------------- atmosphere
  function setupReveal() {
    const targets = $$(".card, .kpi, .chart-card");
    targets.forEach((node) => node.setAttribute("data-reveal", ""));
    if (reduceMotion || !("IntersectionObserver" in window)) {
      targets.forEach((node) => node.classList.add("revealed"));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("revealed");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    targets.forEach((node) => observer.observe(node));
    // Anything already on screen is revealed at once; a timer guarantees that a
    // browser without IntersectionObserver callbacks never leaves content hidden.
    targets.forEach((node) => {
      if (node.getBoundingClientRect().top < window.innerHeight) node.classList.add("revealed");
    });
    setTimeout(() => targets.forEach((node) => node.classList.add("revealed")), 900);
  }

  const PALETTE_ACTIONS = () => [
    { label: state.labels["nav.overview"] || "Overview", hint: "1", run: () => setTab("overview") },
    { label: state.labels["nav.setup"] || "Setup", hint: "2", run: () => setTab("setup") },
    { label: state.labels["nav.run"] || "Run", hint: "3", run: () => setTab("run") },
    { label: state.labels["nav.help"] || "Help", hint: "4", run: () => setTab("help") },
    { label: state.labels["action.synthesize"] || "Generate training data", hint: "run", run: () => { setTab("run"); startPhase("synthesize"); } },
    { label: "Refresh the overview", hint: "R", run: () => loadStats() },
    { label: "Copy the log", hint: "", run: () => $("copy-log").click() },
    { label: "Export audit pack (JSON)", hint: "", run: () => exportAudit("json") },
  ];

  function setupPalette() {
    const shell = $("palette");
    const input = $("palette-input");
    const list = $("palette-list");
    let items = [];
    let active = 0;

    const close = () => {
      shell.classList.add("hidden");
      input.value = "";
      document.body.style.overflow = "";
    };
    const open = () => {
      shell.classList.remove("hidden");
      document.body.style.overflow = "hidden";
      render("");
      input.focus();
    };
    const render = (query) => {
      const needle = query.trim().toLowerCase();
      items = PALETTE_ACTIONS().filter((item) => !needle || item.label.toLowerCase().includes(needle));
      active = 0;
      list.innerHTML = "";
      items.forEach((item, index) => {
        const row = document.createElement("li");
        row.className = "palette-item";
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", index === active ? "true" : "false");
        row.innerHTML = `<span>${esc(item.label)}</span>${item.hint ? `<kbd>${esc(item.hint)}</kbd>` : ""}`;
        row.addEventListener("mouseenter", () => { active = index; paint(); });
        row.addEventListener("click", () => execute());
        list.appendChild(row);
      });
      paint();
    };
    const paint = () => {
      $$(".palette-item", list).forEach((row, index) => {
        row.setAttribute("aria-selected", index === active ? "true" : "false");
      });
    };
    const execute = () => {
      const item = items[active];
      close();
      if (item) item.run();
    };

    $("palette-open").addEventListener("click", open);
    input.addEventListener("input", () => render(input.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        active = Math.min(items.length - 1, active + 1);
        paint();
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        active = Math.max(0, active - 1);
        paint();
      } else if (event.key === "Enter") {
        event.preventDefault();
        execute();
      } else if (event.key === "Escape") {
        close();
      }
    });
    shell.addEventListener("click", (event) => { if (event.target === shell) close(); });
    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        shell.classList.contains("hidden") ? open() : close();
      }
      if (event.key === "Escape" && !shell.classList.contains("hidden")) close();
    });
  }

  // -------------------------------------------------------------------- boot
  async function boot() {
    wire();
    setupReveal();
    setupPalette();
    try {
      const data = await api("/api/bootstrap");
      state.bootstrap = data;
      applyLabels(data.labels || {});
      $("version").textContent = "v" + data.version;
      document.documentElement.lang = data.locale;
      document.body.dir = (data.locales || []).find((item) => item.code === data.locale)?.rtl ? "rtl" : "ltr";

      const select = $("locale-select");
      select.innerHTML = "";
      (data.locales || []).forEach((locale) => {
        const option = document.createElement("option");
        option.value = locale.code;
        option.textContent = locale.native_name + (locale.coverage === "partial" ? " *" : "");
        if (locale.code === data.locale) option.selected = true;
        select.appendChild(option);
      });

      renderTemplates(data.templates || []);
      $("objective-hint").textContent = data.has_project
        ? "This folder already contains a project: " + data.config_path
        : "Choose a ready-made task or describe your own. You can change everything later.";

      const compliance = $("compliance");
      (data.compliance || []).forEach((note) => {
        const item = document.createElement("li");
        item.innerHTML = `<strong>${esc(note.topic)}:</strong> ${esc(note.statement)}`;
        compliance.appendChild(item);
      });

      if (data.manifest && data.manifest.synthesis) {
        $("log").textContent = "Last run: " + (data.manifest.run_id || "") +
          " (" + String(data.manifest.status || "") + ")";
      }
      if (!($("log").textContent || "").trim()) {
        $("log").textContent = "Run a phase to see live progress here.";
      }
      if (!($("next-step").textContent || "").trim()) {
        $("next-step").textContent = "Run a phase to get a recommendation.";
      }

      $("status-left").textContent = (data.locales || []).length + " languages · loopback only · token-protected";
      $("status-right").textContent = data.project_dir || "";

      await refreshState();
      refreshCost();
      const fromHash = location.hash.replace("#", "");
      setTab(fromHash ? fromHash : (data.has_project ? "overview" : "setup"), { push: false });
      await loadStats();
      if (data.job && data.job.status === "running") {
        state.job = data.job;
        state.startedAt = Date.now();
        renderJob(data.job);
        startElapsed();
        pollJob();
      }
    } catch (error) {
      document.body.innerHTML = `<main><div class="card"><h2>Cannot open Quarkon</h2>
        <p>${esc(String(error.message))}</p>
        <p>Restart it from the terminal with: <code>quarkon ui</code></p></div></main>`;
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
