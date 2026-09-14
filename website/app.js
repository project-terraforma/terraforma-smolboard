const TABLE_COLUMNS = [
  "model",
  "f1",
  "average_latency",
  "precision",
  "accuracy",
  "recall",
  "tokens_per_second",
  "benchmark_date",
];

let leaderboard = [];
let sortState = { key: "f1", direction: "desc" };

const MODEL_SUBMISSIONS_FILE = "data/model_submissions.json";

async function fetchSubmissions() {
  try {
    const response = await fetch("/api/model-submissions");
    if (!response.ok) return [];
    const data = await response.json();
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

async function saveSavedModels(models) {
  const response = await fetch("/api/model-submissions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(models),
  });

  if (!response.ok) {
    throw new Error("Unable to save model to JSON file.");
  }
}

function makeModelEntry(formData) {
  const modelName = String(formData.get("modelName") || "").trim();
  const huggingFaceUrl = String(formData.get("huggingFaceUrl") || "").trim();
  const parameterCount = Number(formData.get("parameterCount") || 0);

  if (!modelName || !huggingFaceUrl || !Number.isFinite(parameterCount) || parameterCount <= 0) {
    throw new Error("Please complete all fields with valid values.");
  }

  return {
    model: modelName,
    hugging_face_url: huggingFaceUrl,
    parameter_count: parameterCount,
    parameter_bucket: parameterCount < 5000000000 ? "<5B" : parameterCount < 10000000000 ? "<10B" : "<20B",
    benchmark_date: new Date().toISOString().slice(0, 10),
    f1: 0,
    average_latency: 0,
    precision: 0,
    accuracy: 0,
    recall: 0,
    tokens_per_second: 0,
  };
}

function setupModelForm() {
  const form = document.getElementById("model-form");
  const statusEl = document.getElementById("form-status");
  if (!form || !statusEl) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);

    try {
      const entry = makeModelEntry(formData);
      const existing = await fetchSubmissions();
      const duplicate = existing.find((item) => item.model.toLowerCase() === entry.model.toLowerCase());
      if (duplicate) {
        throw new Error("A model with this name already exists.");
      }

      existing.push(entry);
      await saveSavedModels(existing);
      statusEl.textContent = "Saved to data/model_submissions.json.";
      statusEl.className = "form-status success";
      form.reset();

      if (typeof leaderboard !== "undefined") {
        leaderboard = [...leaderboard, entry];
        renderTable();
        populateModelCheckboxes();
        renderMetricCharts();
      }
    } catch (error) {
      statusEl.textContent = error.message || "Unable to save model.";
      statusEl.className = "form-status error";
    }
  });
}

const METRIC_LABELS = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  average_latency: "Average Latency",
};

const METRIC_COLORS = {
  accuracy: "#0d4f36",
  precision: "#0d4f36",
  recall: "#0d4f36",
  f1: "#0d4f36",
  average_latency: "#0d4f36",
};

async function loadData() {
  const response = await fetch("data/leaderboard.json");
  const fileRows = await response.json();
  const savedModels = await fetchSubmissions();
  leaderboard = [...fileRows, ...savedModels];

  if (document.getElementById("leaderboard-body")) {
    renderTable();
  }

  if (document.getElementById("model-checkboxes")) {
    populateModelCheckboxes();
    applyStateFromUrl();
  }

  if (document.getElementById("metric-charts")) {
    renderMetricCharts();
  }
}

function formatValue(key, value) {
  if (value == null) return "—";
  if (key === "benchmark_date") return new Date(value).toISOString().slice(0, 10);
  if (["accuracy", "precision", "recall", "f1"].includes(key)) return Number(value).toFixed(3);
  if (key === "average_latency") return Number(value).toFixed(3);
  if (key === "tokens_per_second") return Number(value).toFixed(1);
  return value;
}

function getFilteredRows() {
  const activeTab = getActiveFilter();
  let rows = [...leaderboard];

  if (activeTab && activeTab !== "all" && activeTab !== "graphs") {
    rows = rows.filter((row) => row.parameter_bucket === activeTab);
  }

  rows.sort((a, b) => {
    const dir = sortState.direction === "asc" ? 1 : -1;
    const aVal = a[sortState.key];
    const bVal = b[sortState.key];
    if (aVal == null && bVal == null) return 0;
    if (aVal == null) return 1 * dir;
    if (bVal == null) return -1 * dir;
    if (typeof aVal === "string" && typeof bVal === "string") {
      return aVal.localeCompare(bVal) * dir;
    }
    return (Number(aVal) - Number(bVal)) * dir;
  });

  return rows;
}

function renderTable() {
  const rows = getFilteredRows();
  const tbody = document.getElementById("leaderboard-body");
  if (!tbody) return;

  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.model || "unknown"}</td>
      <td>${formatValue("f1", row.f1)}</td>
      <td>${formatValue("average_latency", row.average_latency)}</td>
      <td>${formatValue("precision", row.precision)}</td>
      <td>${formatValue("accuracy", row.accuracy)}</td>
      <td>${formatValue("recall", row.recall)}</td>
      <td>${formatValue("tokens_per_second", row.tokens_per_second)}</td>
      <td>${formatValue("benchmark_date", row.benchmark_date)}</td>
    </tr>
  `).join("");
}

function populateModelCheckboxes() {
  const container = document.getElementById("model-checkboxes");
  if (!container) return;

  const uniqueModels = [...new Set(leaderboard.map((row) => row.model).filter(Boolean))];
  container.innerHTML = uniqueModels
    .map((name) => `
      <label>
        <input type="checkbox" class="model-checkbox" value="${name}">
        ${name}
      </label>
    `)
    .join("");
}

function getSelectedModels() {
  return Array.from(document.querySelectorAll(".model-checkbox:checked")).map((el) => el.value);
}

function getSelectedMetrics() {
  return Array.from(document.querySelectorAll("#metric-checkboxes input:checked")).map((el) => el.value);
}

function getPageMode() {
  const path = window.location.pathname.split("/").pop() || "index.html";
  if (path.endsWith("compare.html")) return "compare";
  if (path.endsWith("add-model.html")) return "add-model";
  return "leaderboard";
}

function getUrlState() {
  const params = new URLSearchParams(window.location.search);
  return {
    models: (params.get("models") || "").split(",").map((item) => item.trim()).filter(Boolean),
    metrics: (params.get("metrics") || "").split(",").map((item) => item.trim()).filter(Boolean),
  };
}

function applyStateFromUrl() {
  const { models, metrics } = getUrlState();
  const modelInputs = document.querySelectorAll(".model-checkbox");
  if (modelInputs.length) {
    modelInputs.forEach((input) => {
      input.checked = models.length ? models.includes(input.value) : false;
    });
  }

  const metricInputs = document.querySelectorAll("#metric-checkboxes input");
  if (metricInputs.length) {
    metricInputs.forEach((input) => {
      input.checked = metrics.length ? metrics.includes(input.value) : false;
    });
  }
}

function syncStateToUrl() {
  const selectedModels = getSelectedModels();
  const selectedMetrics = getSelectedMetrics();
  const params = new URLSearchParams();

  if (selectedModels.length) params.set("models", selectedModels.join(","));
  if (selectedMetrics.length) params.set("metrics", selectedMetrics.join(","));

  const nextUrl = `${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}`;
  window.history.replaceState({}, "", nextUrl);
}

function getActiveFilter() {
  return document.querySelector(".filter-pill.is-active")?.dataset.filter
    || document.querySelector(".tab.is-active")?.dataset.filter
    || "all";
}

function adjustColor(hex, amount) {
  const raw = hex.replace("#", "");
  const num = Number.parseInt(raw, 16);
  const r = Math.min(255, Math.max(0, (num >> 16) + amount));
  const g = Math.min(255, Math.max(0, ((num >> 8) & 0x00ff) + amount));
  const b = Math.min(255, Math.max(0, (num & 0x0000ff) + amount));
  return `rgb(${r}, ${g}, ${b})`;
}

function compactModelLabel(model) {
  if (!model) return "";
  const cleaned = model.replace(/[_/]+/g, " ").trim();
  const parts = cleaned.split(/\s+/).filter(Boolean);
  if (parts.length <= 2) return parts.join(" ");
  return `${parts[0]} ${parts[1]} ${parts[2]}`.trim();
}

function drawBarChart(canvas, metric, rows, hoveredIndex = null) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { top: 20, right: 18, bottom: 96, left: 70 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const values = rows.map((row) => Number(row[metric] ?? 0));
  const maxValue = Math.max(...values, 0.0001);
  const steps = 5;
  const groupGap = rows.length ? chartWidth / rows.length : chartWidth;
  const barWidth = rows.length ? groupGap * 0.72 : chartWidth;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#dfead8";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "#0c4d2d";
  ctx.lineWidth = 1;
  ctx.textAlign = "right";

  for (let i = 0; i <= steps; i += 1) {
    const tickValue = (maxValue / steps) * i;
    const y = pad.top + chartHeight - (chartHeight * i) / steps;
    const progress = (y - pad.top) / chartHeight;
    const alpha = 0.95 - progress * 0.8;
    ctx.strokeStyle = `rgba(12, 77, 45, ${Math.max(alpha, 0.15)})`;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(tickValue.toFixed(metric === "average_latency" ? 1 : 3), pad.left - 10, y + 4);
  }

  for (let i = 0; i <= 12; i += 1) {
    const y = pad.top + (chartHeight / 12) * i;
    const alpha = 0.95 - (i / 12) * 0.75;
    ctx.strokeStyle = `rgba(12, 77, 45, ${Math.max(alpha, 0.15)})`;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + 4, y);
    ctx.stroke();
  }

  const leftAxisGradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + chartHeight);
  leftAxisGradient.addColorStop(0, "rgba(12, 77, 45, 0.9)");
  leftAxisGradient.addColorStop(0.5, "rgba(12, 77, 45, 0.55)");
  leftAxisGradient.addColorStop(1, "rgba(12, 77, 45, 0.2)");
  ctx.strokeStyle = leftAxisGradient;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + chartHeight);
  ctx.stroke();

  const bottomAxisGradient = ctx.createLinearGradient(pad.left, 0, width - pad.right, 0);
  bottomAxisGradient.addColorStop(0, "rgba(12, 77, 45, 0.9)");
  bottomAxisGradient.addColorStop(0.5, "rgba(12, 77, 45, 0.55)");
  bottomAxisGradient.addColorStop(1, "rgba(12, 77, 45, 0.2)");
  ctx.strokeStyle = bottomAxisGradient;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + chartHeight);
  ctx.lineTo(width - pad.right, pad.top + chartHeight);
  ctx.stroke();

  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  rows.forEach((row, index) => {
    const value = Number(row[metric] ?? 0);
    const x = pad.left + index * groupGap + (groupGap - barWidth) / 2;
    const barHeight = (value / maxValue) * chartHeight;
    const y = pad.top + chartHeight - barHeight;
    const isHovered = hoveredIndex === index;
    const baseColor = METRIC_COLORS[metric] || "#4c78a8";

    ctx.fillStyle = isHovered ? adjustColor(baseColor, 30) : baseColor;
    ctx.fillRect(x, y, barWidth, barHeight);

    if (isHovered) {
      ctx.strokeStyle = "#0c4d2d";
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 1, y - 1, barWidth + 2, barHeight + 2);

      const tooltipValue = value.toFixed(metric === "average_latency" ? 1 : 3);
      const tooltipWidth = 92;
      const tooltipHeight = 38;
      const tooltipX = Math.min(Math.max(x + barWidth / 2 - tooltipWidth / 2, 18), width - tooltipWidth - 18);
      const tooltipY = Math.max(y - tooltipHeight - 12, 18);

      ctx.fillStyle = "rgba(11, 48, 31, 0.9)";
      ctx.fillRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.strokeRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
      ctx.fillStyle = "#eaf7ef";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`${METRIC_LABELS[metric] || metric}`, tooltipX + tooltipWidth / 2, tooltipY + 15);
      ctx.font = "11px sans-serif";
      ctx.fillText(tooltipValue, tooltipX + tooltipWidth / 2, tooltipY + 30);
    }

    ctx.fillStyle = "#0c4d2d";
    const label = compactModelLabel(row.model);
    const shouldRotate = rows.length > 7;
    ctx.save();
    ctx.translate(x + barWidth / 2, height - 62);
    ctx.rotate(shouldRotate ? -Math.PI / 20 : 0);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(label, 0, 0);
    ctx.restore();
  });

  ctx.fillStyle = "#0c4d2d";
  ctx.font = "bold 12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(METRIC_LABELS[metric] || metric, width / 2, 16);
  ctx.save();
  ctx.translate(18, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Value", 0, 0);
  ctx.restore();
  ctx.fillText("Model", width / 2, height - 12);
  canvas.__chartMeta = { rows, metric, groupGap, barWidth, pad, chartHeight };
}

function renderMetricCharts() {
  const container = document.getElementById("metric-charts");
  if (!container) return;

  const selectedModels = getSelectedModels();
  const selectedMetrics = getSelectedMetrics();

  if (!selectedModels.length || !selectedMetrics.length) {
    container.innerHTML = "<p class=\"chart-empty-state\">Select at least one model and one metric.</p>";
    return;
  }

  const selectedCount = selectedMetrics.length;
  const maxWidth = selectedCount === 1 ? "min(1000px, 68vw)" : selectedCount === 2 ? "50vw" : "none";
  container.style.setProperty("--chart-max-width", maxWidth);

  const rows = leaderboard.filter((row) => selectedModels.includes(row.model));
  container.innerHTML = selectedMetrics.map((metric) => `
    <div class="metric-chart">
      <h3>${METRIC_LABELS[metric] || metric}</h3>
      <canvas width="640" height="360" data-metric="${metric}"></canvas>
    </div>
  `).join("");

  container.querySelectorAll("canvas[data-metric]").forEach((canvas) => {
    const metric = canvas.dataset.metric;
    const renderHover = (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
      const meta = canvas.__chartMeta || { rows, metric, groupGap: canvas.width / Math.max(rows.length, 1), barWidth: (canvas.width / Math.max(rows.length, 1)) * 0.72, pad: { left: 70, top: 20, right: 18, bottom: 70 } };
      const groupGap = meta.groupGap;
      const barWidth = meta.barWidth;
      const idx = rows.findIndex((_, index) => {
        const left = meta.pad.left + index * groupGap + (groupGap - barWidth) / 2;
        const right = left + barWidth;
        const top = meta.pad.top;
        const bottom = meta.pad.top + meta.chartHeight;
        return x >= left && x <= right && y >= top && y <= bottom;
      });
      drawBarChart(canvas, metric, rows, idx >= 0 ? idx : null);
    };

    canvas.onmousemove = renderHover;
    canvas.onmouseleave = () => drawBarChart(canvas, metric, rows, null);
    drawBarChart(canvas, metric, rows, null);
  });
}

function attachEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      const targetUrl = button.dataset.url;
      if (targetUrl) {
        const url = new URL(targetUrl, window.location.origin);
        const selectedModels = getSelectedModels();
        const selectedMetrics = getSelectedMetrics();
        if (selectedModels.length) url.searchParams.set("models", selectedModels.join(","));
        if (selectedMetrics.length) url.searchParams.set("metrics", selectedMetrics.join(","));
        window.location.href = url.toString();
        return;
      }

      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("is-active"));
      button.classList.add("is-active");
      const isGraphs = button.dataset.filter === "graphs";
      document.getElementById("leaderboard-section").classList.toggle("hidden", isGraphs);
      document.getElementById("graphs-section").classList.toggle("hidden", !isGraphs);
      renderTable();
    });
  });

  const selector = document.querySelector(".params-selector");
  const indicator = selector?.querySelector(".selector-indicator");
  const updateSelectorIndicator = () => {
    if (!selector || !indicator) return;
    const buttons = [...selector.querySelectorAll(".filter-pill")];
    const activeButton = selector.querySelector(".filter-pill.is-active") || buttons[0];
    const activeIndex = buttons.indexOf(activeButton);
    const gap = 4;
    const buttonWidth = activeButton.offsetWidth;
    const offset = activeIndex * (buttonWidth + gap);
    indicator.style.width = `${buttonWidth}px`;
    indicator.style.transform = `translateX(${offset}px)`;
  };

  document.querySelectorAll(".filter-pill").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".filter-pill").forEach((pill) => pill.classList.remove("is-active"));
      button.classList.add("is-active");
      updateSelectorIndicator();
      renderTable();
    });
  });

  if (selector) {
    selector.dataset.active = document.querySelector(".filter-pill.is-active")?.dataset.filter || "all";
    updateSelectorIndicator();
    window.addEventListener("resize", updateSelectorIndicator);
  }

  document.querySelectorAll("#leaderboard-table th[data-key]").forEach((header) => {
    header.addEventListener("click", () => {
      const key = header.dataset.key;
      if (sortState.key === key) {
        sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
      } else {
        sortState.key = key;
        sortState.direction = "desc";
      }
      renderTable();
    });
  });

  document.addEventListener("change", (event) => {
    if (event.target.matches(".model-checkbox") || event.target.matches("#metric-checkboxes input")) {
      syncStateToUrl();
      renderMetricCharts();
    }
  });

  const pageMode = getPageMode();
  if (pageMode === "compare") {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.filter === "graphs"));
    document.getElementById("leaderboard-section")?.classList.add("hidden");
    document.getElementById("graphs-section")?.classList.remove("hidden");
    document.getElementById("add-model-section")?.classList.add("hidden");
  } else if (pageMode === "add-model") {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.filter === "add-model"));
    document.getElementById("leaderboard-section")?.classList.add("hidden");
    document.getElementById("graphs-section")?.classList.add("hidden");
    document.getElementById("add-model-section")?.classList.remove("hidden");
  } else {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.filter === "leaderboard"));
    document.getElementById("leaderboard-section")?.classList.remove("hidden");
    document.getElementById("graphs-section")?.classList.add("hidden");
    document.getElementById("add-model-section")?.classList.add("hidden");
  }
}

setupModelForm();
attachEvents();
loadData();
