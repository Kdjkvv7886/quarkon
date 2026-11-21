/* Quarkon web UI: one file, no build step, no external dependencies. */
(() => {
  "use strict";

  const params = new URLSearchParams(location.search);
  const token = params.get("token") || "";
  const state = { bootstrap: null, labels: {}, templates: [], selectedTemplate: null, job: null };

  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

  async function api(path, options = {}) {
    const headers = { "X-Quarkon-Token": token, "Content-Type": "application/json" };
    const response = await fetch(path, { ...options, headers });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.error || {};
      throw new Error(detail.message || detail.hint || ("HTTP " + response.status));
    }
    return payload;
  }

  function applyLabels(labels) {
    state.labels = labels || {};
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      const key = node.getAttribute("data-i18n");
      if (labels[key]) node.textContent = labels[key];
    });
  }

  function renderTemplates(templates) {
    state.templates = templates;
    const host = $("templates");
    host.innerHTML = "";
    templates.forEach((template) => {
      const button = document.createElement("button");
      button.className = "template";
      button.dataset.id = template.id;
      button.innerHTML = `<div class="id">${esc(template.id)}</div>
        <div class="industry">${esc(template.industry)} · ${template.recommended_samples} examples</div>`;
      button.addEventListener("click", () => selectTemplate(template));
      host.appendChild(button);
    });
  }

  function selectTemplate(template) {
    state.selectedTemplate = template.id;
    document.querySelectorAll(".template").forEach((node) => {
      node.classList.toggle("active", node.dataset.id === template.id);
    });
    $("objective").value = template.objective;
    $("samples").value = template.recommended_samples;
    refreshCost();
  }

  function refreshCost() {
    const samples = Number($("samples").value || 300);
    api("/api/plan", { method: "POST", body: JSON.stringify({ samples }) })
      .then((estimate) => {
        $("cost").textContent = "$" + Number(estimate.total_cost_usd).toFixed(2);
        $("cost").title = `≈${estimate.attempts_estimate} attempts, ${estimate.completion_tokens} completion tokens`;
      })
      .catch(() => { $("cost").textContent = "-"; });
  }

  function renderDoctor(report) {
    const host = $("doctor");
    host.innerHTML = "";
    (report.findings || []).forEach((finding) => {
      const node = document.createElement("div");
      node.className = "finding " + (finding.severity || "");
      node.innerHTML = `<div class="t">${esc(finding.title || finding.code)}</div>
        ${finding.meaning && finding.severity !== "ok" ? `<div class="m">${esc(finding.meaning)}</div>` : ""}
        ${finding.fix ? `<div class="f">How to fix: ${esc(finding.fix)}</div>` : ""}`;
      host.appendChild(node);
    });
  }

  function renderState(info) {
    $("dataset-lines").textContent = String(info.dataset_lines ?? 0);
    $("adapter-state").textContent = info.has_adapter ? "ready" : "-";
    $("gguf-count").textContent = String((info.gguf || []).length);
    if (info.job) renderJob(info.job);
  }

  function renderJob(job) {
    state.job = job;
    const log = $("log");
    const lines = (job.events || []).map((event) => {
      const phase = event.phase || event.event || "event";
      const detail = Object.entries(event)
        .filter(([key]) => !["phase", "event", "ts"].includes(key))
        .map(([key, value]) => `${key}=${value}`)
        .join(" ");
      return `${new Date((event.ts || 0) * 1000).toLocaleTimeString()}  ${phase}  ${detail}`;
    });
    log.textContent = lines.join("\n") || "…";
    log.scrollTop = log.scrollHeight;
    const running = job.status === "running";
    ["run-synthesize", "run-train", "run-export"].forEach((id) => { $(id).disabled = running; });
    $("job-title").textContent = job.status === "done"
      ? (state.labels["status.done"] || "Finished")
      : job.status === "failed"
        ? (state.labels["status.failed"] || "Stopped with a problem")
        : (state.labels["status.running"] || "Working...");
    if (job.error) {
      $("error-card").classList.remove("hidden");
      $("error-meaning").textContent = `${job.error.title}: ${job.error.meaning}`;
      $("error-fix").textContent = job.error.fix || "";
    } else if (!running) {
      $("error-card").classList.add("hidden");
    }
    if (!running && job.result) {
      $("next-step").textContent = job.status === "done"
        ? "Continue with the next button, or open the audit pack for review."
        : "Fix the problem above and press the same button again; finished work is kept.";
    }
  }

  async function pollJob() {
    if (!state.job || state.job.status !== "running") return;
    try {
      const info = await api("/api/state");
      renderState(info);
      if (info.job && info.job.status !== "running") {
        renderJob(info.job);
        refreshState();
      } else if (info.job) {
        renderJob(info.job);
      }
    } catch (error) { /* transient; keep polling */ }
    setTimeout(pollJob, 1500);
  }

  function refreshState() {
    return api("/api/state").then(renderState).catch(() => {});
  }

  async function startPhase(phase) {
    $("error-card").classList.add("hidden");
    try {
      const payload = await api("/api/run", { method: "POST", body: JSON.stringify({ phase }) });
      renderJob(payload.job);
      pollJob();
    } catch (error) {
      $("error-card").classList.remove("hidden");
      $("error-meaning").textContent = String(error.message);
      $("error-fix").textContent = "Create a project first in the Setup tab, then try again.";
    }
  }

  function wire() {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((node) => node.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((node) => node.classList.remove("active"));
        tab.classList.add("active");
        $("tab-" + tab.dataset.tab).classList.add("active");
      });
    });

    $("samples").addEventListener("change", refreshCost);
    $("create").addEventListener("click", async () => {
      const payload = {
        template: state.selectedTemplate,
        objective: $("objective").value.trim(),
        samples: Number($("samples").value || 300),
        language: $("data-language").value,
      };
      $("setup-result").textContent = "Creating project...";
      try {
        const result = await api("/api/setup", { method: "POST", body: JSON.stringify(payload) });
        $("setup-result").textContent = `Project created:\n${result.project_dir}\nConfig: ${result.config_path}`;
        await refreshState();
      } catch (error) {
        $("setup-result").textContent = "Problem: " + String(error.message);
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
    $("cancel").addEventListener("click", () => api("/api/cancel", { method: "POST", body: "{}" }).catch(() => {}));

    const exportAudit = async (format) => {
      try {
        const result = await api("/api/audit/export", { method: "POST", body: JSON.stringify({ format }) });
        $("audit-result").textContent = "Saved: " + result.path;
      } catch (error) { $("audit-result").textContent = "Problem: " + String(error.message); }
    };
    $("audit-json").addEventListener("click", () => exportAudit("json"));
    $("audit-csv").addEventListener("click", () => exportAudit("csv"));

    $("locale-select").addEventListener("change", async (event) => {
      await api("/api/locale", { method: "POST", body: JSON.stringify({ locale: event.target.value }) });
      location.reload();
    });
  }

  async function boot() {
    wire();
    try {
      const data = await api("/api/bootstrap");
      state.bootstrap = data;
      applyLabels(data.labels || {});
      $("version").textContent = "v" + data.version;
      document.documentElement.lang = data.locale;
      const select = $("locale-select");
      select.innerHTML = "";
      (data.locales || []).forEach((locale) => {
        const option = document.createElement("option");
        option.value = locale.code;
        option.textContent = locale.native_name + (locale.coverage === "partial" ? " *" : "");
        if (locale.code === data.locale) option.selected = true;
        select.appendChild(option);
      });
      document.body.dir = (data.locales || []).find((item) => item.code === data.locale)?.rtl ? "rtl" : "ltr";
      renderTemplates(data.templates || []);
      $("objective-hint").textContent = data.has_project
        ? "This folder already contains a project: " + data.config_path
        : "Choose a ready-made task or describe your own. You can change everything later.";
      (data.compliance || []).forEach((note) => {
        const item = document.createElement("li");
        item.innerHTML = `<strong>${esc(note.topic)}:</strong> ${esc(note.statement)}`;
        $("compliance").appendChild(item);
      });
      if (data.manifest && data.manifest.synthesis) {
        $("log").textContent = "Last run: " + (data.manifest.run_id || "") + " (" + String(data.manifest.status || "") + ")";
      }
      await refreshState();
      refreshCost();
      if (data.job && data.job.status === "running") { renderJob(data.job); pollJob(); }
    } catch (error) {
      document.body.innerHTML = `<main><div class="card"><h2>Cannot open Quarkon</h2>
        <p>${esc(String(error.message))}</p>
        <p>Restart it from the terminal with: <code>quarkon ui</code></p></div></main>`;
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
