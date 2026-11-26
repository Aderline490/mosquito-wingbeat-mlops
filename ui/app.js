"use strict";

const NAMES = {
  an_arabiensis: "An. arabiensis",
  an_funestus: "An. funestus",
  culex_pipiens: "Culex pipiens",
  background: "No mosquito",
};
const COLORS = {
  an_arabiensis: "var(--series-1)",
  an_funestus: "var(--series-2)",
  culex_pipiens: "var(--series-3)",
  background: "var(--series-4)",
};
const AUDIO_EXTENSIONS = [".wav", ".flac", ".ogg"];
const CLASS_ORDER = Object.keys(NAMES);
const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

let selectedFile = null;
let previewUrl = null;
let polling = null;
let lastVersion = null;
let pendingMinimum = 6;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.entries(props).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else node.setAttribute(key, value);
  });
  node.append(...children);
  return node;
}

function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  const body = type.includes("json") ? await response.json() : null;
  if (!response.ok) {
    const detail = body && body.detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  return body;
}

function authHeaders() {
  const key = $("api-key").value.trim();
  return key ? { "X-API-Key": key } : {};
}

function setMessage(id, text, kind = "") {
  $(id).textContent = text;
  $(id).className = `message ${kind}`.trim();
}

const percent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;

function duration(seconds) {
  const s = Math.floor(seconds);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

async function refreshHealth() {
  try {
    const health = await api("/health");
    const ok = health.status === "ok";
    $("status-dot").className = `dot ${ok ? "ok" : "bad"}`;
    $("status-text").textContent = ok ? "Online" : "Model not loaded";
    $("tile-uptime").textContent = duration(health.uptime_seconds);
    $("tile-version").textContent = health.model_version ? `Version ${health.model_version}` : "None";
    $("tile-predictions").textContent = health.predictions.toLocaleString();
    if (health.model_version !== lastVersion) {
      lastVersion = health.model_version;
      refreshPerformance();
    }
  } catch {
    $("status-dot").className = "dot bad";
    $("status-text").textContent = "Offline";
  }
}

/* Prediction */
function chooseFile(file) {
  if (!file) return;
  const name = file.name.toLowerCase();
  if (!AUDIO_EXTENSIONS.some((extension) => name.endsWith(extension))) {
    $("predict-result").replaceChildren(el("p", { class: "message error" }, "Please choose a WAV, FLAC or OGG recording."));
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("preview").src = previewUrl;
  $("preview").hidden = false;
  $("drop-text").hidden = true;
  $("file-name").textContent = file.name;
  $("file-name").hidden = false;
  $("predict-button").disabled = false;
  $("predict-result").replaceChildren(el("p", { class: "muted" }, "Ready to predict."));
}

async function runPrediction() {
  if (!selectedFile) return;
  const button = $("predict-button");
  button.disabled = true;
  button.textContent = "Predicting...";
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    showPrediction(await api("/predict", { method: "POST", body: form }));
    refreshHealth();
  } catch (error) {
    $("predict-result").replaceChildren(el("p", { class: "message error" }, error.message));
  } finally {
    button.disabled = false;
    button.textContent = "Predict";
  }
}

function showPrediction(result) {
  const rows = CLASS_ORDER.map((name) => {
    const value = result.probabilities[name];
    const fill = el("div", { class: "prob-fill", style: `width:${value * 100}%;background:${COLORS[name]}` });
    return el("div", { class: "prob" }, el("span", {}, NAMES[name]), el("div", { class: "prob-track" }, fill),
      el("span", { class: "prob-value" }, percent(value)));
  });
  const vector = result.label === "background"
    ? el("p", { class: "badge neutral" }, "No mosquito heard")
    : result.malaria_vector
      ? el("p", { class: "badge alert" }, el("span", { "aria-hidden": "true" }, "\u26A0 "), "Malaria vector")
      : el("p", { class: "badge neutral" }, "Not a malaria vector");
  const agree = result.window_labels.filter((label) => label === result.label).length;
  const nodes = [
    el("p", { class: "muted small" }, "Prediction"),
    el("p", { class: "result-label" }, result.display_name),
    vector,
    el("p", { class: "muted" }, `${percent(result.confidence)} confidence, model version ${result.model_version}`),
    ...rows,
    el("p", { class: "muted small" },
      `${result.duration_seconds.toFixed(1)} s recording, ${agree} of ${result.windows} window${result.windows === 1 ? "" : "s"} agree. ` +
      (result.label === "background" ? "" : `Strongest wingbeat tone: ${Math.round(result.wingbeat_hz)} Hz.`)),
    spectrogramFigure(result.spectrogram),
  ];
  if (result.low_confidence) {
    nodes.push(el("p", { class: "warning" }, "Low confidence. Try a longer, closer recording with less background noise."));
  }
  $("predict-result").replaceChildren(...nodes);
}

function spectrogramFigure(rows) {
  const style = getComputedStyle(document.documentElement);
  const low = hexToRgb(style.getPropertyValue("--spec-low").trim());
  const high = hexToRgb(style.getPropertyValue("--spec-high").trim());
  const height = rows.length, width = rows[0].length;
  const canvas = el("canvas", { width: String(width), height: String(height), class: "spectrogram",
    role: "img", "aria-label": "Spectrogram of the first 2 seconds. Time runs left to right, pitch rises upwards." });
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  rows.forEach((row, r) => row.forEach((value, c) => {
    const t = Math.max(0, Math.min(1, (value + 2) / 5));
    const offset = ((height - 1 - r) * width + c) * 4;
    for (let k = 0; k < 3; k += 1) image.data[offset + k] = Math.round(low[k] + (high[k] - low[k]) * t);
    image.data[offset + 3] = 255;
  }));
  context.putImageData(image, 0, 0);
  return el("figure", { class: "spectrogram-figure" }, canvas,
    el("figcaption", { class: "muted small" }, "Spectrogram, first 2 s. Darker means louder. Pitch rises from 50 Hz at the bottom to 4 kHz at the top."));
}

function hexToRgb(hex) {
  const value = parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

/* Insights */
function showTooltip(event, title, rows) {
  const tip = $("tooltip");
  tip.replaceChildren(el("strong", {}, title), ...rows.map(([label, value]) =>
    el("div", {}, el("span", {}, label), el("span", {}, value))));
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  const x = Math.min(event.clientX + 14, window.innerWidth - box.width - 8);
  const y = Math.max(8, Math.min(event.clientY - box.height - 10, window.innerHeight - box.height - 8));
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${y}px`;
}

function niceMax(value) {
  const steps = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100];
  return steps.find((step) => step >= value) || 100;
}

function drawChart(feature) {
  const W = 340, H = 230, M = { top: 14, right: 12, bottom: 40, left: 36 };
  const plotW = W - M.left - M.right, plotH = H - M.top - M.bottom;
  const bins = feature.edges.length - 1;
  const centers = feature.edges.slice(0, -1).map((edge, i) => (edge + feature.edges[i + 1]) / 2);
  const yMax = niceMax(Math.max(...CLASS_ORDER.flatMap((name) => feature.classes[name].share_pct)));
  const span = feature.max - feature.min;
  const xAt = (i) => M.left + ((centers[i] - feature.min) / span) * plotW;
  const yAt = (value) => M.top + plotH - (value / yMax) * plotH;
  const root = svg("svg", {
    viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: "0",
    "aria-label": `${feature.title} distribution by class. Use left and right arrow keys to read values.`,
  });

  [0, 0.5, 1].forEach((share) => {
    const value = yMax * share;
    root.append(svg("line", { x1: M.left, x2: W - M.right, y1: yAt(value), y2: yAt(value),
      stroke: share === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }));
    root.append(svg("text", { x: M.left - 6, y: yAt(value) + 4, "text-anchor": "end" }, `${value}%`));
  });
  [0, 0.5, 1].forEach((share) => {
    const value = feature.min + span * share;
    const text = `${Number.isInteger(value) ? value : value.toFixed(1)}${share === 1 ? "+" : ""}`;
    root.append(svg("text", { x: M.left + plotW * share, y: H - 25,
      "text-anchor": share === 0 ? "start" : share === 1 ? "end" : "middle" }, text));
  });
  root.append(svg("text", { x: M.left + plotW / 2, y: H - 7, "text-anchor": "middle" }, feature.unit));

  CLASS_ORDER.forEach((name) => {
    const points = feature.classes[name].share_pct.map((value, i) => `${xAt(i).toFixed(1)},${yAt(value).toFixed(1)}`);
    root.append(svg("polyline", { points: points.join(" "), fill: "none", stroke: COLORS[name], "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));
  });

  const labels = CLASS_ORDER.map((name) => {
    const shares = feature.classes[name].share_pct;
    const peak = shares.indexOf(Math.max(...shares));
    return { name, x: xAt(peak) + (peak === 0 ? 24 : 0), y: yAt(shares[peak]) + (peak === 0 ? 4 : -6) };
  }).sort((a, b) => a.y - b.y);
  labels.forEach((label, i) => {
    if (i && label.y - labels[i - 1].y < 13 && Math.abs(label.x - labels[i - 1].x) < 105) label.y = labels[i - 1].y + 13;
    const anchorEnd = label.x > W - 110;
    root.append(svg("text", { class: "series-label", x: anchorEnd ? label.x : label.x + 5,
      y: Math.max(11, label.y), "text-anchor": anchorEnd ? "end" : "start" }, NAMES[label.name]));
  });

  const cross = svg("line", { y1: M.top, y2: M.top + plotH, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  const dots = CLASS_ORDER.map((name) => svg("circle", { r: 4, fill: COLORS[name], stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }));
  root.append(cross, ...dots);

  let active = 0;
  const reveal = (i, event) => {
    active = i;
    cross.setAttribute("x1", xAt(i));
    cross.setAttribute("x2", xAt(i));
    cross.setAttribute("visibility", "visible");
    CLASS_ORDER.forEach((name, index) => {
      dots[index].setAttribute("cx", xAt(i));
      dots[index].setAttribute("cy", yAt(feature.classes[name].share_pct[i]));
      dots[index].setAttribute("visibility", "visible");
    });
    const suffix = i === bins - 1 ? " and above" : "";
    showTooltip(event, `${feature.edges[i]} to ${feature.edges[i + 1]}${suffix}`,
      CLASS_ORDER.map((name) => [NAMES[name], `${feature.classes[name].share_pct[i].toFixed(1)}% of clips`]));
  };
  const hide = () => {
    cross.setAttribute("visibility", "hidden");
    dots.forEach((dot) => dot.setAttribute("visibility", "hidden"));
    $("tooltip").hidden = true;
  };
  root.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * W;
    reveal(Math.max(0, Math.min(bins - 1, Math.floor(((x - M.left) / plotW) * bins))), event);
  });
  root.addEventListener("pointerleave", hide);
  root.addEventListener("blur", hide);
  root.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const next = Math.max(0, Math.min(bins - 1, active + (event.key === "ArrowRight" ? 1 : -1)));
    const box = root.getBoundingClientRect();
    reveal(next, { clientX: box.left + (xAt(next) / W) * box.width, clientY: box.top + 40 });
  });
  return root;
}

function medianTable(feature) {
  const head = el("tr", {}, el("th", {}, "Class"), el("th", {}, "Median"), el("th", {}, "Middle 50%"));
  const rows = CLASS_ORDER.map((name) => {
    const item = feature.classes[name];
    return el("tr", {}, el("td", {}, NAMES[name]), el("td", {}, item.median.toFixed(1)),
      el("td", {}, `${item.q1.toFixed(1)} to ${item.q3.toFixed(1)}`));
  });
  return el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows));
}

async function loadInsights() {
  try {
    const data = await api("/insights");
    $("insights-intro").textContent = `Measured on ${data.clips_total.toLocaleString()} training clips of ` +
      `${data.clip_seconds} seconds each. Each line shows how the clips in a class are spread across the feature.`;
    $("legend").replaceChildren(...CLASS_ORDER.map((name) => el("span", {},
      el("i", { class: "swatch", style: `background:${COLORS[name]}` }), `${NAMES[name]} (${data.class_counts[name]})`)));
    $("charts").replaceChildren(...Object.values(data.features).map((feature) => el("article", { class: "chart" },
      el("h3", {}, feature.title), drawChart(feature), el("p", { class: "story" }, feature.story),
      el("details", {}, el("summary", {}, "View as table"), medianTable(feature)))));
  } catch (error) {
    $("insights-intro").textContent = `Insights are unavailable: ${error.message}`;
  }
}

/* Performance */
async function refreshPerformance() {
  try {
    const metrics = await api("/model/metrics");
    const test = metrics.test;
    $("tile-accuracy").textContent = percent(test.accuracy);
    $("performance-intro").textContent = `Version ${metrics.version}, measured on ${test.samples} test clips of 2 seconds from recording nights the model never heard.`;
    const tiles = [["Accuracy", percent(test.accuracy)], ["Loss", test.loss.toFixed(3)],
      ["Precision", percent(test.precision_macro)], ["Recall", percent(test.recall_macro)],
      ["F1 score", percent(test.f1_macro)], ["ROC-AUC", percent(test.roc_auc_ovr)]];
    $("metric-tiles").replaceChildren(...tiles.map(([label, value]) =>
      el("div", { class: "tile" }, el("span", {}, label), el("strong", {}, value))));

    const peak = Math.max(...test.confusion_matrix.flat(), 1);
    const head = el("tr", {}, el("th", {}, "True \\ Predicted"), ...CLASS_ORDER.map((name) => el("th", {}, NAMES[name])));
    const rows = test.confusion_matrix.map((row, i) => el("tr", {}, el("th", { scope: "row" }, NAMES[CLASS_ORDER[i]]),
      ...row.map((value) => {
        const alpha = value ? 0.1 + 0.55 * (value / peak) : 0;
        return el("td", { class: "cell", style: `background:rgba(var(--matrix),${alpha.toFixed(2)})` }, String(value));
      })));
    $("confusion").replaceChildren(el("thead", {}, head), el("tbody", {}, ...rows));

    const perHead = el("tr", {}, ...["Class", "Precision", "Recall", "F1", "Clips"].map((text) => el("th", {}, text)));
    const perRows = CLASS_ORDER.map((name) => {
      const item = test.per_class[name];
      return el("tr", {}, el("td", {}, NAMES[name]), el("td", {}, percent(item.precision)), el("td", {}, percent(item.recall)),
        el("td", {}, percent(item["f1-score"])), el("td", {}, String(item.support)));
    });
    $("per-class").replaceChildren(el("thead", {}, perHead), el("tbody", {}, ...perRows));
  } catch (error) {
    $("performance-intro").textContent = `Metrics are unavailable: ${error.message}`;
  }
}

/* Upload and retrain */
async function refreshUploads() {
  try {
    const summary = await api("/uploads/summary");
    pendingMinimum = summary.minimum_to_retrain;
    $("key-field").hidden = !summary.api_key_required;
    $("pending-count").textContent = summary.pending;
    const parts = Object.entries(summary.pending_by_class).map(([name, count]) => `${NAMES[name] || name}: ${count}`);
    $("pending-classes").textContent = parts.length
      ? parts.join(", ")
      : `Upload at least ${pendingMinimum} recordings to enable retraining.`;
    $("retrain-button").disabled = polling !== null || summary.pending < pendingMinimum;
  } catch (error) {
    $("pending-classes").textContent = error.message;
  }
}

async function uploadRecordings(event) {
  event.preventDefault();
  const files = [...$("upload-files").files];
  if (!files.length) return;
  const button = $("upload-button");
  button.disabled = true;
  button.textContent = `Uploading ${files.length}...`;
  setMessage("upload-message", "");
  try {
    const form = new FormData();
    form.append("label", $("upload-label").value);
    files.forEach((file) => form.append("files", file));
    const result = await api("/upload", { method: "POST", body: form, headers: authHeaders() });
    const notes = [`Saved ${result.saved} recording${result.saved === 1 ? "" : "s"}.`];
    if (result.duplicates) notes.push(`${result.duplicates} already uploaded.`);
    if (result.rejected.length) notes.push(`${result.rejected.length} rejected: ${result.rejected[0].file}.`);
    setMessage("upload-message", notes.join(" "), result.saved ? "success" : "error");
    $("upload-files").value = "";
    if (result.auto_retrain_job) watchJob(result.auto_retrain_job);
    refreshUploads();
  } catch (error) {
    setMessage("upload-message", error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Upload recordings";
  }
}

async function startRetrain() {
  $("retrain-button").disabled = true;
  setMessage("retrain-message", "");
  $("job-result").replaceChildren();
  try {
    const job = await api("/retrain", { method: "POST", headers: authHeaders() });
    watchJob(job.job_id);
  } catch (error) {
    setMessage("retrain-message", error.message, "error");
    refreshUploads();
  }
}

function watchJob(jobId) {
  if (polling) clearInterval(polling);
  $("job-progress").hidden = false;
  $("job-stage").textContent = "Starting";
  $("retrain-button").disabled = true;
  const tick = async () => {
    try {
      const job = await api(`/retrain/${jobId}`);
      $("job-stage").textContent = job.stage || job.status;
      if (job.status === "completed" || job.status === "failed") {
        clearInterval(polling);
        polling = null;
        $("job-progress").hidden = true;
        setMessage("retrain-message", job.message || job.status, job.status === "failed" ? "error" : "success");
        if (job.result) showComparison(job.result);
        refreshUploads();
        refreshJobs();
        refreshHealth();
      }
    } catch {
      $("job-stage").textContent = "Waiting for the server";
    }
  };
  polling = setInterval(tick, 2000);
  tick();
}

function showComparison(result) {
  const rows = [["Accuracy", "accuracy"], ["Precision", "precision_macro"], ["Recall", "recall_macro"], ["F1 score", "f1_macro"]];
  const head = el("tr", {}, el("th", {}, "Validation"), el("th", {}, "Before"), el("th", {}, "After"));
  const body = rows.map(([label, key]) => el("tr", {}, el("td", {}, label),
    el("td", {}, percent(result.current.validation[key])), el("td", {}, percent(result.candidate.validation[key]))));
  const note = `${result.recordings_used} new recordings used (${result.clips_added} clips), ${result.recordings_rejected} rejected, ${result.epochs_run} epoch${result.epochs_run === 1 ? "" : "s"}.`;
  $("job-result").replaceChildren(el("div", { class: "table-scroll" },
    el("table", {}, el("thead", {}, head), el("tbody", {}, ...body))), el("p", { class: "muted small" }, note));
}

async function refreshJobs() {
  try {
    const { jobs } = await api("/retrain/jobs");
    const active = jobs.find((job) => job.status === "queued" || job.status === "running");
    if (active && polling === null) watchJob(active.id);
    const head = el("tr", {}, ...["Job", "Status", "Recordings", "Version", "Outcome"].map((text) => el("th", {}, text)));
    const rows = jobs.length ? jobs.map((job) => el("tr", {}, el("td", {}, `#${job.id}`), el("td", {}, job.status),
      el("td", {}, String(job.new_files)), el("td", {}, job.to_version ? `v${job.to_version}` : "-"),
      el("td", {}, job.message || job.stage || "")))
      : [el("tr", {}, el("td", { colspan: "5" }, "No retraining runs yet."))];
    $("jobs").replaceChildren(el("thead", {}, head), el("tbody", {}, ...rows));
  } catch {
    /* the status indicator already reports connection problems */
  }
}

function init() {
  $("upload-label").append(...CLASS_ORDER.map((name) => {
    const option = el("option", { value: name }, NAMES[name]);
    return option;
  }));
  $("predict-file").addEventListener("change", (event) => chooseFile(event.target.files[0]));
  $("predict-button").addEventListener("click", runPrediction);
  $("upload-form").addEventListener("submit", uploadRecordings);
  $("retrain-button").addEventListener("click", startRetrain);

  const zone = $("drop-zone");
  ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, (event) => {
    event.preventDefault();
    zone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, (event) => {
    event.preventDefault();
    zone.classList.remove("dragging");
  }));
  zone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

  refreshHealth();
  loadInsights();
  refreshUploads();
  refreshJobs();
  setInterval(refreshHealth, 5000);
}

init();
