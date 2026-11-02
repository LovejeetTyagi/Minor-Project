// Charts for the Models and History pages (Chart.js is loaded before this file).
(function () {
  const el = document.getElementById("chart-data");
  if (!el || typeof Chart === "undefined") return;
  const data = JSON.parse(el.textContent);

  const css = getComputedStyle(document.documentElement);
  Chart.defaults.color = css.getPropertyValue("--text").trim() || "#222";
  Chart.defaults.borderColor = css.getPropertyValue("--border").trim() || "#ddd";
  Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
  const palette = ["#d6336c", "#4263eb", "#2f9e44", "#f08c00"];
  const common = { responsive: true, maintainAspectRatio: false };

  if (data.page === "models") {
    const models = data.metrics.models;

    new Chart(document.getElementById("metricsChart"), {
      type: "bar",
      data: {
        labels: ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        datasets: models.map((m, i) => ({
          label: m.name,
          backgroundColor: palette[i % palette.length],
          data: [m.accuracy, m.precision, m.recall, m.f1, m.roc_auc].map(v => +(v * 100).toFixed(1)),
        })),
      },
      options: { ...common, scales: { y: { min: 50, max: 100, title: { display: true, text: "Score (%)" } } } },
    });

    const rocSets = models.map((m, i) => ({
      label: `${m.name} (AUC ${m.roc_auc.toFixed(2)})`,
      data: m.roc.fpr.map((x, k) => ({ x, y: m.roc.tpr[k] })),
      borderColor: palette[i % palette.length],
      backgroundColor: palette[i % palette.length],
      pointRadius: 0, borderWidth: 2, tension: 0,
    }));
    rocSets.push({ label: "Random guess", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                   borderColor: "#888", borderDash: [6, 4], pointRadius: 0, borderWidth: 1 });
    new Chart(document.getElementById("rocChart"), {
      type: "line",
      data: { datasets: rocSets },
      options: { ...common, scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate" } },
        y: { type: "linear", min: 0, max: 1, title: { display: true, text: "True positive rate" } } } },
    });
  }

  if (data.page === "history") {
    const stats = data.stats;
    new Chart(document.getElementById("riskChart"), {
      type: "doughnut",
      data: { labels: ["Low", "Moderate", "High"],
              datasets: [{ data: [stats.counts.Low, stats.counts.Moderate, stats.counts.High],
                           backgroundColor: ["#2f9e44", "#f08c00", "#e03131"] }] },
      options: common,
    });
    new Chart(document.getElementById("trendChart"), {
      type: "line",
      data: { labels: stats.recent.map((_, i) => i + 1),
              datasets: [{ label: "Risk score (%)", data: stats.recent, borderColor: "#d6336c",
                           backgroundColor: "#d6336c", tension: 0.25 }] },
      options: { ...common, scales: { y: { min: 0, max: 100 } } },
    });
  }
})();
