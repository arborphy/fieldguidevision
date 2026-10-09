(() => {
  "use strict";

  const data = window.SCALE_ANALYSIS;
  if (!data) return;
  const field = Object.fromEntries(data.schema.map((name, index) => [name, index]));
  const modelNames = data.representations;
  const scales = ["distant", "mid-range", "close-up"];
  const scaleModel = document.getElementById("scale-model");
  const speciesHighlight = document.getElementById("species-highlight");
  const selectedPoint = document.getElementById("scale-selected");

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[character]));
  }

  function fitCanvas(canvas) {
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(260, Math.round(canvas.clientWidth));
    const height = Math.round(width / 1.35);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width, height };
  }

  function extent(values) {
    let min = Infinity;
    let max = -Infinity;
    values.forEach((value) => {
      min = Math.min(min, value);
      max = Math.max(max, value);
    });
    const padding = Math.max((max - min) * 0.05, 0.01);
    return [min - padding, max + padding];
  }

  function project(value, domain, range) {
    const denominator = domain[1] - domain[0] || 1;
    return range[0] + ((value - domain[0]) / denominator) * (range[1] - range[0]);
  }

  function canvasPaletteColor(index) {
    return data.cluster_colors[index % data.cluster_colors.length];
  }

  data.species.forEach((species) => {
    const option = document.createElement("option");
    option.value = species;
    option.textContent = species;
    speciesHighlight.append(option);
  });

  const plotted = new Map();

  function drawScalePlots() {
    const representation = scaleModel.value;
    const highlighted = speciesHighlight.value;
    const points = data.projection[representation].points;
    const xDomain = data.projection[representation].x_extent;
    const yDomain = data.projection[representation].y_extent;
    scales.forEach((scale) => {
      const canvas = document.getElementById(`scale-canvas-${scale}`);
      const { context, width, height } = fitCanvas(canvas);
      const margin = { left: 40, right: 12, top: 12, bottom: 30 };
      const records = points.filter((point) => point[field.scale] === scale);
      context.clearRect(0, 0, width, height);
      context.strokeStyle = "#d8dfda";
      context.lineWidth = 1;
      context.strokeRect(margin.left, margin.top, width - margin.left - margin.right, height - margin.top - margin.bottom);
      context.fillStyle = "#65716a";
      context.font = "11px system-ui";
      context.fillText("PC2", 7, margin.top + 12);
      context.fillText("PC1", width - margin.right - 22, height - 8);
      const rendered = [];
      records.forEach((point) => {
        const x = project(point[field.x], xDomain, [margin.left + 4, width - margin.right - 4]);
        const y = project(point[field.y], yDomain, [height - margin.bottom - 4, margin.top + 4]);
        const isHighlighted = !highlighted || point[field.species] === highlighted;
        context.globalAlpha = highlighted ? (isHighlighted ? 0.95 : 0.08) : 0.62;
        context.fillStyle = canvasPaletteColor(point[field.cluster]);
        context.beginPath();
        context.arc(x, y, highlighted && isHighlighted ? 4.2 : 2.6, 0, Math.PI * 2);
        context.fill();
        if (highlighted && isHighlighted) {
          context.globalAlpha = 1;
          context.strokeStyle = "#17221d";
          context.lineWidth = 1;
          context.stroke();
        }
        rendered.push({ x, y, point });
      });
      context.globalAlpha = 1;
      plotted.set(canvas.id, rendered);
    });
  }

  function showScalePoint(point) {
    const correct = point[field.correct];
    selectedPoint.innerHTML = `
      <a href="${esc(point[field.source])}" target="_blank" rel="noreferrer"><img src="${esc(point[field.image])}" alt="${esc(point[field.species])}"></a>
      <dl>
        <dt>Species</dt><dd><i>${esc(point[field.species])}</i></dd>
        <dt>Organ</dt><dd>${esc(point[field.organ])}</dd>
        <dt>Scale</dt><dd>${esc(point[field.scale])}</dd>
        <dt>Cluster</dt><dd>${String(point[field.cluster]).padStart(2, "0")}</dd>
        <dt>Prediction</dt><dd>${point[field.prediction] ? `${esc(point[field.prediction])} · ${correct ? "correct" : "incorrect"}` : "not a strict-test image"}</dd>
      </dl>`;
  }

  scales.forEach((scale) => {
    const canvas = document.getElementById(`scale-canvas-${scale}`);
    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const nearest = plotted.get(canvas.id)
        .map((item) => ({ item, distance: Math.hypot(item.x - x, item.y - y) }))
        .sort((a, b) => a.distance - b.distance)[0];
      if (nearest && nearest.distance < 18) showScalePoint(nearest.item.point);
    });
  });
  scaleModel.addEventListener("change", drawScalePlots);
  speciesHighlight.addEventListener("change", drawScalePlots);
  window.addEventListener("resize", drawScalePlots);

  function distribution(title, items, total, color = "#315f4a") {
    const rows = items.slice(0, 8).map((item) => {
      const width = total ? Math.max(2, 100 * item.count / total) : 0;
      return `<div class="dist-row"><span>${esc(item.label)}</span><span class="dist-bar"><i style="width:${width}%;background:${color}"></i></span><b>${item.count}</b></div>`;
    }).join("");
    return `<div class="distribution"><strong>${esc(title)}</strong>${rows}</div>`;
  }

  function thumb(item, includeScale = false) {
    return `<a class="thumb" href="${esc(item.source)}" target="_blank" rel="noreferrer"><img loading="lazy" src="${esc(item.image)}" alt="${esc(item.species)}"><span><b>${esc(item.species)}</b><i>${esc(item.organ)}${includeScale ? ` · ${esc(item.scale)}` : ""}</i></span></a>`;
  }

  const clusterModel = document.getElementById("cluster-model");
  const clusterSelect = document.getElementById("cluster-select");
  const clusterCase = document.getElementById("cluster-case");

  function loadClusterOptions() {
    const rows = data.clusters[clusterModel.value];
    clusterSelect.innerHTML = rows.slice(0, 8).map((item) => (
      `<option value="${item.cluster}">cluster ${String(item.cluster).padStart(2, "0")} · purity ${(item.species_purity * 100).toFixed(1)}% · ${item.n} images</option>`
    )).join("");
    renderCluster();
  }

  function renderCluster() {
    const item = data.clusters[clusterModel.value].find((row) => row.cluster === Number(clusterSelect.value));
    if (!item) return;
    const rows = item.by_scale.map((group) => (
      `<div class="scale-row"><strong>${esc(group.scale)} · ${group.images.length} shown</strong><div class="thumb-grid">${group.images.map((image) => thumb(image)).join("") || "<p>No images.</p>"}</div></div>`
    )).join("");
    clusterCase.innerHTML = `<article class="cluster-case"><header><div><div class="eyebrow">${esc(modelNames[clusterModel.value])}</div><h3>Cluster ${String(item.cluster).padStart(2, "0")}</h3></div><p>${item.n} images · ${item.species_count} species<br>species purity ${(item.species_purity * 100).toFixed(1)}% · organ purity ${(item.organ_purity * 100).toFixed(1)}%</p></header><div class="distribution-grid">${distribution("Species", item.species, item.n)}${distribution("Organ", item.organs, item.n, "#b36a36")}${distribution("Shot scale", item.scales, item.n, "#6c5a8f")}</div><h4>Before · original centroid order</h4><div class="thumb-grid">${item.original.map((image) => thumb(image, true)).join("")}</div><h4>After · same cluster separated by shot scale</h4>${rows}</article>`;
  }
  clusterModel.addEventListener("change", loadClusterOptions);
  clusterSelect.addEventListener("change", renderCluster);

  const exampleButtons = [...document.querySelectorAll("[data-example-button]")];
  const examplePanels = [...document.querySelectorAll("[data-example-rep]")];
  function showExamples(rep) {
    exampleButtons.forEach((button) => button.classList.toggle("active", button.dataset.exampleButton === rep));
    examplePanels.forEach((panel) => panel.classList.toggle("active", panel.dataset.exampleRep === rep));
  }
  exampleButtons.forEach((button) => button.addEventListener("click", () => showExamples(button.dataset.exampleButton)));

  const categoricalColors = ["#315f4a", "#b36a36", "#6c5a8f", "#3c7187", "#9b5166", "#7b7833", "#546b9a", "#8d6142"];
  function categoryColor(value, map) {
    if (!map.has(value)) map.set(value, categoricalColors[map.size % categoricalColors.length]);
    return map.get(value);
  }

  function setupLocalExplorer() {
    const controls = document.getElementById("local-controls");
    const explorer = document.getElementById("local-explorer");
    const readyModels = Object.keys(data.local).filter((rep) => data.local[rep].length);
    if (!readyModels.length) {
      controls.innerHTML = "";
      explorer.innerHTML = '<div class="method-note"><strong>Waiting for regenerated frozen vectors.</strong> Local PCA is intentionally not approximated from the saved two-dimensional plot.</div>';
      return;
    }
    controls.innerHTML = `<label>Representation<select id="local-model">${readyModels.map((rep) => `<option value="${rep}">${esc(modelNames[rep])}</option>`).join("")}</select></label><label>Mixed cluster<select id="local-cluster"></select></label><label>Projection<select id="local-pair"><option value="0,1">PC1–PC2</option><option value="0,2">PC1–PC3</option><option value="1,2">PC2–PC3</option><option value="rotate">Rotating PC1–PC4</option></select></label><label>Color by<select id="local-color"><option value="species">species</option><option value="scale">shot scale</option><option value="organ">organ</option><option value="correctness">correct / incorrect</option><option value="background">background</option><option value="leaf">leaf state</option><option value="reproductive">reproductive</option></select></label><button id="rotate-step">Rotate +10°</button>`;
    explorer.innerHTML = '<canvas id="local-canvas" class="deep-canvas" width="820" height="560" aria-label="Local principal-component projection"></canvas><aside id="local-detail" class="deep-detail"><p>Click a point to inspect the image.</p></aside>';
    const model = document.getElementById("local-model");
    const cluster = document.getElementById("local-cluster");
    const pair = document.getElementById("local-pair");
    const color = document.getElementById("local-color");
    const canvas = document.getElementById("local-canvas");
    const detail = document.getElementById("local-detail");
    let angle = 0;
    let localRendered = [];

    function loadLocalClusters() {
      cluster.innerHTML = data.local[model.value].map((item) => `<option value="${item.cluster}">cluster ${String(item.cluster).padStart(2, "0")} · n=${item.n}</option>`).join("");
      drawLocal();
    }
    function drawLocal() {
      const item = data.local[model.value].find((row) => row.cluster === Number(cluster.value));
      if (!item) return;
      const { context, width, height } = fitCanvas(canvas);
      const chosen = pair.value;
      const values = item.points.map((point) => {
        if (chosen === "rotate") {
          const radians = angle * Math.PI / 180;
          return { point, x: Math.cos(radians) * point.pc[0] + Math.sin(radians) * point.pc[2], y: Math.cos(radians) * point.pc[1] + Math.sin(radians) * point.pc[3] };
        }
        const [xIndex, yIndex] = chosen.split(",").map(Number);
        return { point, x: point.pc[xIndex], y: point.pc[yIndex] };
      });
      const xDomain = extent(values.map((value) => value.x));
      const yDomain = extent(values.map((value) => value.y));
      const margin = { left: 45, right: 14, top: 18, bottom: 34 };
      context.clearRect(0, 0, width, height);
      context.strokeStyle = "#d8dfda";
      context.strokeRect(margin.left, margin.top, width - margin.left - margin.right, height - margin.top - margin.bottom);
      context.fillStyle = "#65716a";
      context.font = "11px system-ui";
      context.fillText(chosen === "rotate" ? `rotation ${angle}°` : pair.options[pair.selectedIndex].text, margin.left, height - 10);
      const colorMap = new Map();
      localRendered = values.map((value) => {
        const x = project(value.x, xDomain, [margin.left + 5, width - margin.right - 5]);
        const y = project(value.y, yDomain, [height - margin.bottom - 5, margin.top + 5]);
        const category = value.point[color.value];
        context.fillStyle = categoryColor(category, colorMap);
        context.globalAlpha = .72;
        context.beginPath();
        context.arc(x, y, 4, 0, Math.PI * 2);
        context.fill();
        return { x, y, point: value.point };
      });
      context.globalAlpha = 1;
    }
    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const nearest = localRendered.map((item) => ({ item, distance: Math.hypot(item.x - x, item.y - y) })).sort((a, b) => a.distance - b.distance)[0];
      if (!nearest || nearest.distance > 20) return;
      const point = nearest.item.point;
      const prediction = point.prediction ? `${esc(point.prediction)} · ${esc(point.correctness)}` : "not a strict-test image";
      detail.innerHTML = `<a href="${esc(point.source)}" target="_blank" rel="noreferrer"><img src="${esc(point.image)}" alt="${esc(point.species)}"></a><p><b><i>${esc(point.species)}</i></b><br>${esc(point.organ)} · ${esc(point.scale)}<br>${esc(point.background)} · ${esc(point.leaf)} · reproductive ${esc(point.reproductive)}<br>Prediction: ${prediction}</p>`;
    });
    model.addEventListener("change", loadLocalClusters);
    [cluster, pair, color].forEach((element) => element.addEventListener("change", drawLocal));
    document.getElementById("rotate-step").addEventListener("click", () => { angle = (angle + 10) % 180; pair.value = "rotate"; drawLocal(); });
    window.addEventListener("resize", drawLocal);
    loadLocalClusters();
  }

  function setupSeedExplorer() {
    const controls = document.getElementById("seed-controls");
    const explorer = document.getElementById("seed-explorer");
    const readyModels = Object.keys(data.seeds).filter((rep) => data.seeds[rep].length);
    if (!readyModels.length) {
      controls.innerHTML = "";
      explorer.innerHTML = '<div class="method-note"><strong>Waiting for regenerated frozen vectors.</strong> The committed neighbour file stops at rank 10; it is not extrapolated.</div>';
      return;
    }
    controls.innerHTML = `<label>Representation<select id="seed-model">${readyModels.map((rep) => `<option value="${rep}">${esc(modelNames[rep])}</option>`).join("")}</select></label><label>Seed image<select id="seed-select"></select></label><label>Reveal <b id="neighbor-count-label">30</b> neighbours<input id="neighbor-count" type="range" min="5" max="100" step="5" value="30"></label><label>Semantic boundary <b id="boundary-label">not marked</b><input id="boundary-rank" type="range" min="1" max="100" value="100"></label>`;
    explorer.innerHTML = '<div class="selected-point" id="seed-card"></div><div class="neighbor-grid" id="neighbor-grid"></div><div class="boundary" id="boundary-summary"></div>';
    const model = document.getElementById("seed-model");
    const select = document.getElementById("seed-select");
    const count = document.getElementById("neighbor-count");
    const boundary = document.getElementById("boundary-rank");
    function loadSeedOptions() {
      select.innerHTML = data.seeds[model.value].map((item, index) => `<option value="${index}">cluster ${String(item.cluster).padStart(2, "0")} · ${esc(item.species)} · ${esc(item.scale)}</option>`).join("");
      renderSeed();
    }
    function firstChange(neighbors, seed, key) {
      const item = neighbors.find((neighbor) => neighbor[key] !== seed[key]);
      return item ? `rank ${item.rank}` : "not in Top-100";
    }
    function renderSeed() {
      const seed = data.seeds[model.value][Number(select.value || 0)];
      const n = Math.min(Number(count.value), seed.neighbors.length);
      const marked = Number(boundary.value);
      document.getElementById("neighbor-count-label").textContent = n;
      document.getElementById("boundary-label").textContent = marked < 100 ? `rank ${marked}` : "not marked";
      document.getElementById("seed-card").innerHTML = `<a href="${esc(seed.source)}" target="_blank" rel="noreferrer"><img src="${esc(seed.image)}" alt="${esc(seed.species)}"></a><dl><dt>Seed</dt><dd><i>${esc(seed.species)}</i></dd><dt>Organ</dt><dd>${esc(seed.organ)}</dd><dt>Scale</dt><dd>${esc(seed.scale)}</dd><dt>Cluster</dt><dd>${String(seed.cluster).padStart(2, "0")}</dd></dl>`;
      document.getElementById("neighbor-grid").innerHTML = seed.neighbors.slice(0, n).map((item) => {
        const changed = item.species !== seed.species || item.scale !== seed.scale || item.organ !== seed.organ;
        return `<article class="neighbor ${changed ? "changed" : ""}"${item.rank === marked ? ' style="border-width:3px"' : ""}><a href="${esc(item.source)}" target="_blank" rel="noreferrer"><img loading="lazy" src="${esc(item.image)}" alt="${esc(item.species)}"></a><small>#${item.rank} · cosine ${item.similarity.toFixed(3)}</small><b>${esc(item.species)}</b><small>${esc(item.organ)} · ${esc(item.scale)}</small></article>`;
      }).join("");
      document.getElementById("boundary-summary").textContent = `First species change: ${firstChange(seed.neighbors, seed, "species")} · first scale change: ${firstChange(seed.neighbors, seed, "scale")} · first organ change: ${firstChange(seed.neighbors, seed, "organ")} · reviewer boundary: ${marked < 100 ? `rank ${marked}` : "not marked"}.`;
    }
    model.addEventListener("change", loadSeedOptions);
    select.addEventListener("change", renderSeed);
    count.addEventListener("input", renderSeed);
    boundary.addEventListener("input", renderSeed);
    loadSeedOptions();
  }

  drawScalePlots();
  loadClusterOptions();
  showExamples("bioclip");
  setupLocalExplorer();
  setupSeedExplorer();
})();
