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
  leaderboard = await response.json();
  renderTable();
  populateModelCheckboxes();
  renderMetricCharts();
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
  const activeTab = document.querySelector(".tab.is-active")?.dataset.filter;
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
  const uniqueModels = [...new Set(leaderboard.map((row) => row.model).filter(Boolean))];
  container.innerHTML = uniqueModels
    .map((name) => `
      <label>
        <input type="checkbox" class="model-checkbox" value="${name}" checked>
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

function adjustColor(hex, amount) {
  const raw = hex.replace("#", "");
  const num = Number.parseInt(raw, 16);
  const r = Math.min(255, Math.max(0, (num >> 16) + amount));
  const g = Math.min(255, Math.max(0, ((num >> 8) & 0x00ff) + amount));
  const b = Math.min(255, Math.max(0, (num & 0x0000ff) + amount));
  return `rgb(${r}, ${g}, ${b})`;
}

function drawBarChart(canvas, metric, rows, hoveredIndex = null) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { top: 20, right: 18, bottom: 70, left: 70 };
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
  ctx.strokeStyle = "#0c4d2d";
  ctx.fillStyle = "#0c4d2d";
  ctx.lineWidth = 1;
  ctx.textAlign = "right";

  for (let i = 0; i <= steps; i += 1) {
    const tickValue = (maxValue / steps) * i;
    const y = pad.top + chartHeight - (chartHeight * i) / steps;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(tickValue.toFixed(metric === "average_latency" ? 1 : 3), pad.left - 10, y + 4);
  }

  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + chartHeight);
  ctx.lineTo(width - pad.right, pad.top + chartHeight);
  ctx.stroke();

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
    }

    ctx.fillStyle = "#0c4d2d";
    ctx.fillText(row.model, x + barWidth / 2, height - 24);
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
  ctx.fillText("Model", width / 2, height - 8);
  canvas.__chartMeta = { rows, metric, groupGap, barWidth, pad, chartHeight };
}

function renderMetricCharts() {
  const container = document.getElementById("metric-charts");
  const selectedModels = getSelectedModels();
  const selectedMetrics = getSelectedMetrics();

  if (!selectedModels.length || !selectedMetrics.length) {
    container.innerHTML = "<p>Select at least one model and one metric.</p>";
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
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("is-active"));
      button.classList.add("is-active");
      if (button.dataset.filter === "graphs") {
        document.getElementById("leaderboard-section").classList.add("hidden");
        document.getElementById("graphs-section").classList.remove("hidden");
      } else {
        document.getElementById("graphs-section").classList.add("hidden");
        document.getElementById("leaderboard-section").classList.remove("hidden");
      }
      renderTable();
    });
  });

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
      renderMetricCharts();
    }
  });
}

attachEvents();
loadData();
