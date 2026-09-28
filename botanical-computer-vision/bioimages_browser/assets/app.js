(() => {
  "use strict";

  const data = window.BIOIMAGES_DATA;
  const main = document.querySelector("main");
  const nav = document.querySelector("nav");
  const navToggle = document.querySelector(".nav-toggle");
  if (!data) {
    main.innerHTML = '<p class="loading">Catalog data could not be loaded.</p>';
    return;
  }

  const imagesById = new Map(data.images.map((image) => [image.id, image]));
  const speciesBySlug = new Map(data.species.map((species) => [species.slug, species]));
  const individualsById = new Map(data.individuals.map((individual) => [individual.id, individual]));
  const groupsBySlug = new Map(data.organ_groups.map((group) => [group.slug, group]));
  const groupOrder = data.organ_groups.map((group) => group.slug);
  const nf = new Intl.NumberFormat("en-US");

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));

  function url(params = {}) {
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") search.set(key, value);
    });
    const query = search.toString();
    return query ? `?${query}` : "index.html";
  }

  function routeLink(label, params, className = "") {
    return `<a ${className ? `class="${className}"` : ""} href="${url(params)}" data-route>${label}</a>`;
  }

  function imageCard(image) {
    return `<article class="image-card">
      <a href="${url({ view: "image", id: image.id })}" data-route aria-label="${escapeHtml(image.primary_label)} — ${escapeHtml(image.species)}">
        <div class="thumb">
          <img src="${escapeHtml(image.thumbnail_url)}" alt="${escapeHtml(image.title || image.primary_label)}" loading="lazy" decoding="async">
          <span class="badge">${escapeHtml(image.group_label)}</span>
        </div>
        <p class="image-label">${escapeHtml(image.organ_category)} · ${escapeHtml(image.subview)}</p>
        <p class="image-meta"><i>${escapeHtml(image.species)}</i></p>
      </a>
    </article>`;
  }

  function mountGallery(element, images, batchSize = 100) {
    if (!element) return;
    let shown = Math.min(batchSize, images.length);
    const draw = () => {
      element.innerHTML = `<div class="gallery">${images.slice(0, shown).map(imageCard).join("")}</div>` +
        (shown < images.length ? `<button class="load-more" type="button">Show ${Math.min(batchSize, images.length - shown)} more</button>` : "");
      const button = element.querySelector(".load-more");
      if (button) button.addEventListener("click", () => { shown = Math.min(shown + batchSize, images.length); draw(); });
    };
    draw();
  }

  function organCard(group) {
    const reps = group.representatives.map((id) => imagesById.get(id)).filter(Boolean);
    return `<article class="organ-card"><a href="${url({ view: "organs", organ: group.slug })}" data-route>
      <div class="organ-collage">${reps.map((image) => `<img src="${escapeHtml(image.thumbnail_url)}" alt="" loading="lazy">`).join("")}</div>
      <div class="organ-copy"><h3>${escapeHtml(group.label)}</h3><span class="count">${nf.format(group.count)} images · ${group.subviews.length} views</span></div>
    </a></article>`;
  }

  function setPage(html, title) {
    main.innerHTML = html;
    document.title = `${title} · BioImages Browser`;
    window.scrollTo(0, 0);
    updateNav();
  }

  function updateNav() {
    const view = new URLSearchParams(location.search).get("view") || "home";
    document.querySelectorAll("nav a").forEach((anchor) => {
      const anchorView = new URL(anchor.href, location.href).searchParams.get("view");
      anchor.classList.toggle("active", anchorView === view || (view === "image" && anchorView === "images"));
    });
  }

  function renderHome() {
    const stats = data.stats;
    setPage(`<section class="hero"><div class="wrap">
      <p class="eyebrow">A modern BioImages collection</p>
      <h1>Browse trees by species, structure, view, and individual.</h1>
      <p class="lede">Every photograph keeps its original BioImages annotation and displays a clear primary label directly beneath the image. Follow one tree across bark, twig, leaf, flower, fruit, and whole-plant views.</p>
      <div class="stats">
        <div class="stat"><strong>${nf.format(stats.images)}</strong><span>images</span></div>
        <div class="stat"><strong>${nf.format(stats.species)}</strong><span>species</span></div>
        <div class="stat"><strong>${nf.format(stats.organ_categories)}</strong><span>organ categories</span></div>
        <div class="stat"><strong>${nf.format(stats.view_types)}</strong><span>view types</span></div>
        <div class="stat"><strong>${nf.format(stats.individuals)}</strong><span>individual plants</span></div>
      </div>
    </div></section>
    <section class="section"><div class="wrap">
      <div class="section-head"><div><p class="eyebrow">Browse by structure</p><h2>Organs and views</h2><p>BioImages’ biological organization, presented as a direct visual index.</p></div>${routeLink("See all tags →", { view: "organs" }, "text-link")}</div>
      <div class="organ-grid">${data.organ_groups.filter((g) => g.slug !== "other").map(organCard).join("")}</div>
    </div></section>
    <section class="section"><div class="wrap">
      <div class="section-head"><div><p class="eyebrow">Browse by taxonomy</p><h2>${stats.species} species</h2><p>Open a species to see its images organized by organ category and subview.</p></div>${routeLink("Browse all species →", { view: "species" }, "text-link")}</div>
      <div class="species-grid">${data.species.slice(0, 9).map(speciesRow).join("")}</div>
    </div></section>
    <section class="section"><div class="wrap"><div class="notice">
      <strong>${nf.format(stats.complete_primary_labels)} of ${nf.format(stats.images)} images have a complete primary label.</strong>
      BioImages remains the canonical annotation source; ${stats.prediction_images} held-out images also show an optional model prediction in their detail view.
    </div></div></section>`, "Home");
  }

  function speciesRow(species) {
    const first = imagesById.get(species.image_ids[0]);
    return `<a class="species-row" href="${url({ view: "species", species: species.slug })}" data-route>
      <img src="${escapeHtml(first.thumbnail_url)}" alt="" loading="lazy">
      <span class="row-copy"><strong>${escapeHtml(species.name)}</strong><small>${escapeHtml(species.common_name || "No common name")}</small><small>${species.image_ids.length} images · ${escapeHtml(species.family)}</small></span>
    </a>`;
  }

  function renderSpeciesList() {
    setPage(`<div class="wrap"><div class="page-head"><div><p class="eyebrow">Taxonomic index</p><h1>Species</h1><p>All ${data.stats.species} species in the BioImages northeastern trees collection.</p></div></div>
      <div class="toolbar"><label>Search species<input id="species-search" type="search" placeholder="Scientific or common name"></label></div>
      <p class="results-line" id="species-count"></p><div class="species-grid" id="species-grid"></div>
    </div>`, "Species");
    const input = document.querySelector("#species-search");
    const grid = document.querySelector("#species-grid");
    const count = document.querySelector("#species-count");
    const draw = () => {
      const term = input.value.trim().toLowerCase();
      const rows = data.species.filter((s) => `${s.name} ${s.common_name} ${s.family}`.toLowerCase().includes(term));
      count.textContent = `${rows.length} species`;
      grid.innerHTML = rows.map(speciesRow).join("") || '<p class="empty">No species match this search.</p>';
    };
    input.addEventListener("input", draw); draw();
  }

  function groupedSections(images) {
    const byGroup = new Map();
    images.forEach((image) => {
      if (!byGroup.has(image.group)) byGroup.set(image.group, []);
      byGroup.get(image.group).push(image);
    });
    return groupOrder.filter((slug) => byGroup.has(slug)).map((slug) => {
      const members = byGroup.get(slug);
      const group = groupsBySlug.get(slug) || { label: "Other" };
      const bySubview = new Map();
      members.forEach((image) => {
        if (!bySubview.has(image.subview)) bySubview.set(image.subview, []);
        bySubview.get(image.subview).push(image);
      });
      return `<section class="group-section"><div class="group-title"><h2>${escapeHtml(group.label)}</h2><span class="count">${members.length} images</span></div>
        ${[...bySubview.entries()].map(([subview, subset]) => `<div class="subview-block"><div class="subview-title"><h3>${escapeHtml(subview)}</h3><span class="count">${subset.length}</span></div><div class="gallery">${subset.map(imageCard).join("")}</div></div>`).join("")}
      </section>`;
    }).join("");
  }

  function renderSpeciesDetail(slug) {
    const species = speciesBySlug.get(slug);
    if (!species) return renderNotFound();
    const images = species.image_ids.map((id) => imagesById.get(id));
    const individuals = new Set(images.map((image) => image.individual_id));
    setPage(`<div class="wrap">
      <p class="crumbs">${routeLink("Species", { view: "species" })} / ${escapeHtml(species.name)}</p>
      <div class="page-head"><div><p class="eyebrow">${escapeHtml(species.family)}</p><h1><i>${escapeHtml(species.name)}</i></h1><p>${escapeHtml(species.common_name)} · ${images.length} images · ${individuals.size} individual plants</p></div><a class="text-link" href="${escapeHtml(species.source_url)}">Original species page ↗</a></div>
      ${groupedSections(images)}
    </div>`, species.name);
  }

  function renderOrgans(params) {
    const slug = params.get("organ");
    if (!slug) {
      setPage(`<div class="wrap"><div class="page-head"><div><p class="eyebrow">Biological index</p><h1>Organs and views</h1><p>Start with an organ, then refine by BioImages subview.</p></div></div><div class="organ-grid">${data.organ_groups.map(organCard).join("")}</div>
      <section class="section"><h2>Exact-view vocabulary</h2><div class="table-wrap"><table><thead><tr><th>Primary label</th><th class="number">Images</th></tr></thead><tbody>${data.taxonomy.exact_labels.map((row) => `<tr><td>${escapeHtml(row.name)}</td><td class="number">${row.count}</td></tr>`).join("")}</tbody></table></div></section></div>`, "Organs and views");
      return;
    }
    const group = groupsBySlug.get(slug);
    if (!group) return renderNotFound();
    renderFilteredImages({ title: group.label, eyebrow: "Organ browser", base: data.images.filter((image) => image.group === slug), fixedOrgan: slug });
  }

  function optionList(rows, allLabel) {
    return `<option value="">${allLabel}</option>${rows.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  }

  function renderFilteredImages({ title = "All images", eyebrow = "Image catalog", base = data.images, fixedOrgan = "" }) {
    setPage(`<div class="wrap"><div class="page-head"><div><p class="eyebrow">${escapeHtml(eyebrow)}</p><h1>${escapeHtml(title)}</h1><p>Every thumbnail includes its BioImages primary label. Use the filters to narrow the collection.</p></div></div>
      <div class="toolbar">
        <label>Search<input id="filter-search" type="search" placeholder="Species, label, individual ID"></label>
        <label>Species<select id="filter-species">${optionList(data.species.map((s) => s.name), "All species")}</select></label>
        ${fixedOrgan ? "" : `<label>Organ<select id="filter-organ">${optionList(data.organ_groups.map((g) => g.slug), "All organs")}</select></label>`}
        <label>Subview<select id="filter-subview">${optionList([...new Set(base.map((i) => i.subview))].sort(), "All views")}</select></label>
      </div><p class="results-line" id="filter-count"></p><div id="filtered-gallery"></div>
    </div>`, title);
    const search = document.querySelector("#filter-search");
    const species = document.querySelector("#filter-species");
    const organ = document.querySelector("#filter-organ");
    const subview = document.querySelector("#filter-subview");
    const count = document.querySelector("#filter-count");
    const draw = () => {
      const term = search.value.trim().toLowerCase();
      const filtered = base.filter((image) =>
        (!term || `${image.species} ${image.common_name} ${image.primary_label} ${image.individual_id}`.toLowerCase().includes(term)) &&
        (!species.value || image.species === species.value) &&
        (!organ || !organ.value || image.group === organ.value) &&
        (!subview.value || image.subview === subview.value));
      count.textContent = `${nf.format(filtered.length)} images`;
      mountGallery(document.querySelector("#filtered-gallery"), filtered);
    };
    [search, species, organ, subview].filter(Boolean).forEach((el) => el.addEventListener(el.tagName === "INPUT" ? "input" : "change", draw));
    draw();
  }

  function individualRow(individual) {
    const image = imagesById.get(individual.representatives[0]);
    return `<a class="individual-row" href="${url({ view: "individuals", id: individual.id })}" data-route>
      <img src="${escapeHtml(image.thumbnail_url)}" alt="" loading="lazy">
      <span class="row-copy"><strong>${escapeHtml(individual.species)}</strong><small>${escapeHtml(individual.id)}</small><small>${individual.image_ids.length} images · ${individual.groups.map((g) => groupsBySlug.get(g)?.label || g).join(", ")}</small></span>
    </a>`;
  }

  function renderIndividuals(params) {
    const id = params.get("id");
    if (id) return renderIndividualDetail(id);
    setPage(`<div class="wrap"><div class="page-head"><div><p class="eyebrow">Provenance index</p><h1>Individual plants</h1><p>${data.stats.individuals} linked plants. One individual can connect whole-tree, bark, twig, leaf, flower, and fruit views.</p></div></div>
      <div class="toolbar"><label>Search<input id="individual-search" type="search" placeholder="Species or individual ID"></label><label>Organ<select id="individual-organ">${optionList(data.organ_groups.map((g) => g.slug), "Any organ")}</select></label></div>
      <p class="results-line" id="individual-count"></p><div class="individual-grid" id="individual-grid"></div>
    </div>`, "Individuals");
    const search = document.querySelector("#individual-search");
    const organ = document.querySelector("#individual-organ");
    const grid = document.querySelector("#individual-grid");
    const count = document.querySelector("#individual-count");
    const draw = () => {
      const term = search.value.trim().toLowerCase();
      const rows = data.individuals.filter((item) => (!term || `${item.species} ${item.common_name} ${item.id}`.toLowerCase().includes(term)) && (!organ.value || item.groups.includes(organ.value)));
      count.textContent = `${rows.length} individuals`;
      grid.innerHTML = rows.map(individualRow).join("") || '<p class="empty">No individuals match these filters.</p>';
    };
    [search, organ].forEach((el) => el.addEventListener(el.tagName === "INPUT" ? "input" : "change", draw)); draw();
  }

  function renderIndividualDetail(id) {
    const individual = individualsById.get(id);
    if (!individual) return renderNotFound();
    const images = individual.image_ids.map((imageId) => imagesById.get(imageId));
    const species = data.species.find((s) => s.name === individual.species);
    setPage(`<div class="wrap"><p class="crumbs">${routeLink("Individuals", { view: "individuals" })} / ${escapeHtml(id)}</p>
      <div class="page-head"><div><p class="eyebrow">Individual plant</p><h1><i>${escapeHtml(individual.species)}</i></h1><p>${escapeHtml(id)} · ${images.length} linked photographs</p></div>${species ? routeLink("View species →", { view: "species", species: species.slug }, "text-link") : ""}</div>
      <div class="notice">These photographs share the same BioImages <code>individual_id</code>, allowing different biological structures to be viewed as one documented plant.</div>
      ${groupedSections(images)}
    </div>`, `${individual.species} individual`);
  }

  function renderImageDetail(id) {
    const image = imagesById.get(id);
    if (!image) return renderNotFound();
    const species = data.species.find((s) => s.name === image.species);
    const location = [image.locality, image.county, image.state_province, image.country_code].filter(Boolean).join(", ");
    const captured = image.captured_at ? new Date(image.captured_at).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" }) : "Not recorded";
    const prediction = image.prediction?.organ_category || image.prediction?.organ_tag;
    const predictionHtml = prediction ? `<div class="prediction"><h3>Auxiliary model prediction</h3>
      <p>Predicted organ: <strong>${escapeHtml(prediction.label)}</strong> · ${(prediction.confidence * 100).toFixed(1)}% confidence</p>
      <p>BioImages organ: <strong>${escapeHtml(image.organ_category)}</strong> · <span class="${prediction.match ? "match" : "mismatch"}">${prediction.match ? "match" : "mismatch"}</span></p>
      <p class="image-meta">DINOv3 frozen linear probe, ${escapeHtml(prediction.protocol)}. The model does not predict the fine subview.</p></div>` : "";
    const reviewHtml = image.review_flags.length ? `<p class="notice review"><strong>Manual metadata review:</strong> ${escapeHtml(image.review_flags.join(", "))}.</p>` : "";
    setPage(`<div class="wrap"><p class="crumbs">${routeLink("All images", { view: "images" })} / ${escapeHtml(image.id)}</p>
      <div class="detail"><div class="detail-image"><a href="${escapeHtml(image.image_url)}"><img src="${escapeHtml(image.image_url)}" alt="${escapeHtml(image.title)}"></a></div>
      <div class="detail-copy"><p class="eyebrow">Image detail</p><h1><i>${escapeHtml(image.species)}</i></h1><p>${escapeHtml(image.common_name)}</p>
        <p class="primary-label">${escapeHtml(image.organ_category)} · ${escapeHtml(image.subview)}</p>${reviewHtml}${predictionHtml}
        <dl class="definition">
          <div><dt>Primary label</dt><dd>${escapeHtml(image.primary_label)}</dd></div>
          <div><dt>Coarse tag</dt><dd>${escapeHtml(image.organ_tag)}</dd></div>
          <div><dt>View code</dt><dd>${escapeHtml(image.view_code)}</dd></div>
          <div><dt>Species</dt><dd>${species ? routeLink(`<i>${escapeHtml(image.species)}</i>`, { view: "species", species: species.slug }) : escapeHtml(image.species)}</dd></div>
          <div><dt>Individual</dt><dd>${routeLink(escapeHtml(image.individual_id), { view: "individuals", id: image.individual_id })}</dd></div>
          <div><dt>Captured</dt><dd>${escapeHtml(captured)}</dd></div>
          <div><dt>Location</dt><dd>${escapeHtml(location || "Not recorded")}</dd></div>
          <div><dt>Creator</dt><dd>${escapeHtml(image.creator || "Not recorded")}</dd></div>
          <div><dt>Credit</dt><dd>${escapeHtml(image.credit)}</dd></div>
          <div><dt>License</dt><dd><a href="${escapeHtml(image.license)}">${escapeHtml(image.usage_terms || image.license)}</a></dd></div>
          <div><dt>Source</dt><dd><a href="${escapeHtml(image.source_url)}">BioImages record ↗</a>${image.archive_url ? ` · <a href="${escapeHtml(image.archive_url)}">archived file ↗</a>` : ""}</dd></div>
        </dl>
      </div></div></div>`, image.primary_label);
  }

  function renderModels() {
    const model = data.model_evaluation;
    if (!model.available) {
      setPage('<div class="wrap"><div class="page-head"><h1>Model evaluation</h1></div><p class="empty">No model evaluation is available.</p></div>', "Model evaluation"); return;
    }
    const individual = model.summary.filter((r) => r.protocol === "individual_disjoint");
    const species = model.summary.filter((r) => r.protocol === "species_disjoint");
    const table = (rows) => `<div class="table-wrap"><table><thead><tr><th>Model</th><th class="number">Test images</th><th class="number">Accuracy</th><th class="number">Macro F1</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${escapeHtml(r.model)}</td><td class="number">${r.test_images}</td><td class="number">${(r.top1 * 100).toFixed(1)}%</td><td class="number">${(r.macro_f1 * 100).toFixed(1)}%</td></tr>`).join("")}</tbody></table></div>`;
    const perLabels = [...new Set(model.per_class.map((r) => r.label))];
    const modelNames = [...new Set(model.per_class.map((r) => r.model))];
    setPage(`<div class="wrap"><div class="page-head"><div><p class="eyebrow">Auxiliary validation</p><h1>Model evaluation</h1><p>How consistently do frozen visual representations reproduce the BioImages coarse organ tag? This page supports the catalog; it is not the catalog’s organizing principle.</p></div><a class="text-link" href="${model.benchmark_url}">Full benchmark →</a></div>
      <section class="section"><h2>Individual-disjoint organ tags</h2>${table(individual)}</section>
      <section class="section"><h2>Species-disjoint organ tags</h2>${table(species)}</section>
      <section class="section"><h2>Per-organ held-out performance</h2><div class="table-wrap"><table><thead><tr><th>Organ</th>${modelNames.map((name) => `<th class="number">${escapeHtml(name)} F1</th>`).join("")}</tr></thead><tbody>${perLabels.map((label) => `<tr><td>${escapeHtml(label)}</td>${modelNames.map((name) => { const row = model.per_class.find((r) => r.label === label && r.model === name); return `<td class="number">${row ? (row.f1 * 100).toFixed(1) + "%" : "—"}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div></section>
      <section class="section"><h2>Confusion matrices</h2><div class="matrix-grid">${Object.entries(model.confusions).map(([key, src]) => `<figure><a href="${src}"><img src="${src}" alt="${escapeHtml(key)} confusion matrix" loading="lazy"></a><figcaption>${escapeHtml(key.replace("efficientnet_b0", "EfficientNet-B0"))}</figcaption></figure>`).join("")}</div></section>
    </div>`, "Model evaluation");
  }

  function renderNotFound() {
    setPage(`<div class="wrap"><div class="page-head"><div><p class="eyebrow">Not found</p><h1>This catalog entry does not exist.</h1><p>${routeLink("Return to the collection", {})}</p></div></div></div>`, "Not found");
  }

  function render() {
    const params = new URLSearchParams(location.search);
    const view = params.get("view") || "home";
    if (view === "home") renderHome();
    else if (view === "species") params.get("species") ? renderSpeciesDetail(params.get("species")) : renderSpeciesList();
    else if (view === "organs") renderOrgans(params);
    else if (view === "individuals") renderIndividuals(params);
    else if (view === "images") renderFilteredImages({});
    else if (view === "image") renderImageDetail(params.get("id"));
    else if (view === "models") renderModels();
    else renderNotFound();
  }

  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-route]");
    if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    history.pushState({}, "", link.getAttribute("href"));
    nav.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
    render();
  });
  navToggle.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
  });
  window.addEventListener("popstate", render);
  render();
})();
