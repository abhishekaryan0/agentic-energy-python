const $ = (id) => document.getElementById(id);

function log(msg) {
  const el = $("log");
  const ts = new Date().toLocaleTimeString();
  el.textContent = `[${ts}] ${msg}\n` + el.textContent;
}

function fmt(n) {
  if (n == null) return "—";
  return Number(n).toLocaleString();
}

function metricCard(label, value, sub = "") {
  return `
    <div class="metric-card">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
      ${sub ? `<div class="sub">${sub}</div>` : ""}
    </div>`;
}

function applyGenerationPreview(data) {
  const g = data.generation || {};
  const previewEl = $("genPreview");
  const btn = $("generateBtn");
  const label = $("generateBtnLabel");

  if (!previewEl) return;

  previewEl.textContent = g.message || "—";

  if (!btn || !label) return;

  const healthOk = $("connectionStatus")?.classList.contains("ok");
  label.textContent = "Generate";

  const canGenerate =
    g.mode === "full_backfill" ||
    (g.mode === "incremental" && g.pending_days > 0);
  btn.disabled = !healthOk || !canGenerate;
}

function renderSummary(data) {
  applyGenerationPreview(data);
  const p = data.production || {};
  const s = data.sensor_readings || {};
  const h = data.master_history || {};
  const inv = data.invoices || {};
  const cat = data.asset_catalog || {};

  $("summaryMetrics").innerHTML = [
    metricCard("Daily production", fmt(p.rows), `${fmt(p.wells)} wells`),
    metricCard("Sensor readings", fmt(s.rows), `~${s.ratio_to_daily || 24}× daily`),
    metricCard("Master history", fmt(h.rows), `${fmt(h.assets)} assets`),
    metricCard("Invoices", fmt(inv.rows), "SYNTH_INVOICES"),
    metricCard("Asset catalog", fmt(cat.rows), "SYNTH_ASSETS"),
    metricCard("Watermark", (p.max_date || "—").slice(0, 10), "last daily date"),
  ].join("");

  const rows = [
    ["Production", p.min_date, p.max_date, p.rows],
    ["Sensors", s.min_timestamp, s.max_timestamp, s.rows],
    ["Master history", h.min_date, h.max_date, h.rows],
    ["Invoices", inv.min_date, inv.max_date, inv.rows],
  ];

  $("coverage").innerHTML = rows
    .map(([name, minD, maxD, count]) => {
      const label = minD && maxD ? `${String(minD).slice(0, 10)} → ${String(maxD).slice(0, 10)}` : "No data";
      const pct = count ? Math.min(100, Math.log10(count + 1) * 25) : 0;
      return `
        <div class="coverage-row">
          <span>${name}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          <span>${fmt(count)} rows</span>
        </div>
        <div style="font-size:0.75rem;color:var(--muted);margin:-0.35rem 0 0.5rem 0">${label}</div>`;
    })
    .join("");

  const wells = data.sample_wells || [];
  $("wellTable").innerHTML =
    wells.length === 0
      ? `<tr><td colspan="4">No well data yet</td></tr>`
      : wells
          .map(
            (w) => `
        <tr>
          <td>${w.well_id}</td>
          <td>${fmt(w.daily_rows)}</td>
          <td>${w.min_date || "—"}</td>
          <td>${w.max_date || "—"}</td>
        </tr>`
          )
          .join("");
}

function renderRun(job) {
  const run = job.run;
  if (!run) return;
  $("runPanel").classList.remove("hidden");
  const mode = run.is_full_backfill ? "Full load" : "Incremental";
  let winText = `${mode}: ${run.window_start} → ${run.window_end}`;
  if (run.pending_days != null && !run.is_full_backfill) {
    winText = `Added ${run.pending_days} day(s) after ${run.last_daily || "—"}\n${winText}`;
  }
  if (run.message) winText += `\n\n${run.message}`;
  else if (!run.daily_rows && !run.master_history_rows) {
    winText += "\n\n⚠ No rows added — data may already be current through today.";
  }
  $("runWindow").textContent = winText;
  $("runMetrics").innerHTML = [
    metricCard("Added daily", fmt(run.daily_rows)),
    metricCard("Added sensors", fmt(run.sensor_rows)),
    metricCard("Added history", fmt(run.master_history_rows)),
    metricCard("Added invoices", fmt(run.invoice_rows)),
  ].join("");
  if (job.summary) renderSummary(job.summary);
}

async function checkHealth() {
  const pill = $("connectionStatus");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ok) {
      pill.className = "status-pill ok";
      pill.querySelector("span:last-child").textContent = "Snowflake connected";
      log("Snowflake connection OK");
    } else {
      pill.className = "status-pill err";
      pill.querySelector("span:last-child").textContent = "Snowflake offline";
      log(`Connection error: ${data.detail || "unknown"}`);
    }
  } catch (e) {
    pill.className = "status-pill err";
    pill.querySelector("span:last-child").textContent = "API unreachable";
    log(`Health check failed: ${e.message}`);
  }
}

async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderSummary(data);
    log(`Summary loaded (${data.database}.${data.schema})`);
  } catch (e) {
    $("summaryMetrics").innerHTML = `<p class="loading">Failed: ${e.message}</p>`;
    log(`Stats error: ${e.message}`);
  }
}

async function pollJob(jobId) {
  const fill = $("progressFill");
  let pct = 30;
  const tick = setInterval(() => {
    pct = Math.min(92, pct + 4);
    fill.style.animation = "none";
    fill.style.width = `${pct}%`;
    fill.style.marginLeft = "0";
  }, 800);

  while (true) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    $("progressText").textContent = `Status: ${job.status}…`;

    if (job.status === "completed") {
      clearInterval(tick);
      fill.style.width = "100%";
      $("progressText").textContent = "Generation complete";
      renderRun(job);
      log(
        `Run done: +${job.run.daily_rows} daily, +${job.run.sensor_rows} sensor rows`
      );
      $("generateBtn").classList.remove("running");
      $("generateBtn").disabled = false;
      setTimeout(() => $("progressWrap").classList.add("hidden"), 2000);
      return;
    }
    if (job.status === "failed") {
      clearInterval(tick);
      $("progressText").textContent = "Failed";
      log(`Generation failed: ${job.error}`);
      $("generateBtn").classList.remove("running");
      $("generateBtn").disabled = false;
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

async function generate() {
  $("generateBtn").disabled = true;
  $("generateBtn").classList.add("running");
  $("progressWrap").classList.remove("hidden");
  $("progressText").textContent = "Starting pipeline…";
  const pending = $("genPreview")?.textContent || "";
  log(`Generate requested — ${pending}`);

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
    });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    log(`Job ${job_id} queued`);
    await pollJob(job_id);
  } catch (e) {
    log(`Generate error: ${e.message}`);
    $("generateBtn").classList.remove("running");
    $("generateBtn").disabled = false;
    $("progressWrap").classList.add("hidden");
  }
}

$("generateBtn").addEventListener("click", generate);
$("refreshBtn").addEventListener("click", () => {
  log("Refreshing summary…");
  loadStats();
});

checkHealth().then(loadStats);
