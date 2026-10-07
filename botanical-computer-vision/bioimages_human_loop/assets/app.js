(() => {
  "use strict";

  const data = window.BIOIMAGES_DATA;
  const loop = window.BIOIMAGES_HUMAN_LOOP;
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
  const reviewRuntime = window.BIOIMAGES_RUNTIME || {};
  const reviewBackendConfigured = Boolean(reviewRuntime.supabaseUrl && reviewRuntime.supabaseAnonKey && window.supabase?.createClient);
  const reviewClient = reviewBackendConfigured
    ? window.supabase.createClient(reviewRuntime.supabaseUrl, reviewRuntime.supabaseAnonKey, {
      auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
    })
    : null;
  const reviewBatchId = reviewRuntime.reviewBatchId || "00000000-0000-4000-8000-000000000001";
  let reviewSession = null;
  let remoteHumanGold = null;
  let reviewBackendState = reviewBackendConfigured ? "loading" : "local";

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
    document.title = `${title} · BioImages Human Loop`;
    window.scrollTo(0, 0);
    updateNav();
  }

  function updateNav() {
    const view = new URLSearchParams(location.search).get("view") || "home";
    document.querySelectorAll("nav a").forEach((anchor) => {
      const anchorView = new URL(anchor.href, location.href).searchParams.get("view") || "home";
      anchor.classList.toggle("active", anchorView === view || (view === "mode" && anchorView === "home") || (view === "image" && anchorView === "full-gallery"));
    });
  }

  function analysisCategories(image, modelKey) {
    const prediction = image.tagging?.models?.[modelKey];
    if (!prediction) return [];
    const categories = [];
    if (prediction.bioimages_match === true && prediction.gemma_f1 >= 0.67) categories.push("good match");
    if (prediction.bioimages_match === false) categories.push("BioImages mismatch");
    if (prediction.gemma_jaccard < 0.33) categories.push("Gemma disagreement");
    if (image.tagging.model_disagreement) categories.push("model disagreement");
    return categories;
  }

  function tagPills(tags, withScores = false) {
    if (!tags?.length) return '<span class="muted">None</span>';
    return `<span class="tag-list">${tags.map((item) => {
      const tag = typeof item === "string" ? item : item.tag;
      const score = typeof item === "string" ? "" : ` ${(item.confidence * 100).toFixed(0)}%`;
      return `<span class="tag-pill">${escapeHtml(tag)}${withScores ? score : ""}</span>`;
    }).join("")}</span>`;
  }

  function analysisCard(entry) {
    const { image, modelKey } = entry;
    const prediction = image.tagging.models[modelKey];
    const status = prediction.bioimages_match === null ? "not evaluable" : prediction.bioimages_match ? "BioImages match ✓" : "BioImages mismatch ✗";
    const categories = analysisCategories(image, modelKey);
    return `<article class="analysis-card">
      <a class="analysis-image" href="${url({ view: "image", id: image.id })}" data-route><img src="${escapeHtml(image.thumbnail_url)}" alt="${escapeHtml(image.title)}" loading="lazy"></a>
      <div class="analysis-copy">
        <p class="analysis-species"><i>${escapeHtml(image.species)}</i> <span>${escapeHtml(prediction.model)}</span></p>
        <h3>Model tags</h3>${tagPills(prediction.tags, true)}
        <h3>BioImages</h3><p>${escapeHtml(image.tagging.bioimages_label)}</p>
        <h3>Gemma</h3>${tagPills(image.tagging.gemma_tags)}
        <p class="analysis-score"><span class="${prediction.bioimages_match ? "match" : "mismatch"}">${status}</span><span>Gemma F1 ${(prediction.gemma_f1 * 100).toFixed(0)}%</span><span>Jaccard ${(prediction.gemma_jaccard * 100).toFixed(0)}%</span></p>
        <p class="analysis-issues">${categories.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</p>
      </div>
    </article>`;
  }

  function mountAnalysisGallery(element, entries, batchSize = 36) {
    let shown = Math.min(batchSize, entries.length);
    const draw = () => {
      element.innerHTML = `<div class="analysis-grid">${entries.slice(0, shown).map(analysisCard).join("")}</div>` +
        (shown < entries.length ? `<button class="load-more" type="button">Show ${Math.min(batchSize, entries.length - shown)} more</button>` : "") +
        (!entries.length ? '<p class="empty">No images match these filters.</p>' : "");
      element.querySelector(".load-more")?.addEventListener("click", () => { shown = Math.min(entries.length, shown + batchSize); draw(); });
    };
    draw();
  }

  const HUMAN_STORAGE_KEY = "bioimages-human-gold-v2";
  const humanSetById = new Map((loop?.human_set || []).map((item) => [item.image_id, item]));
  const coarseTagMap = {
    leaf: "leaf", needle: "leaf", twig: "twig", branch: "twig", stem: "twig", bud: "twig",
    bark: "bark", trunk: "bark", flower: "flower", inflorescence: "flower", fruit: "fruit",
    cone: "cone", seed: "seed", "whole plant": "whole plant", "whole tree": "whole plant",
  };

  function loadHumanGold() {
    if (reviewBackendConfigured && reviewSession) return remoteHumanGold || {};
    try { return JSON.parse(localStorage.getItem(HUMAN_STORAGE_KEY) || "{}"); }
    catch (_) { return {}; }
  }

  function saveLocalHumanGold(records) {
    localStorage.setItem(HUMAN_STORAGE_KEY, JSON.stringify(records));
  }

  async function loadRemoteHumanGold() {
    if (!reviewClient || !reviewSession) { remoteHumanGold = {}; return; }
    const { data: rows, error } = await reviewClient.from("annotations")
      .select("image_id,visible_tags,primary_subject,ambiguous,other_tag,note,revision,submitted_at,updated_at")
      .eq("batch_id", reviewBatchId);
    if (error) throw error;
    remoteHumanGold = Object.fromEntries((rows || []).map((row) => [row.image_id, {
      tags: row.visible_tags,
      primary: row.primary_subject || "",
      ambiguous: row.ambiguous,
      other_tag: row.other_tag || "",
      note: row.note || "",
      revision: row.revision,
      labeled_at: row.updated_at || row.submitted_at,
    }]));
  }

  async function saveHumanGoldEntry(imageId, record) {
    if (!reviewBackendConfigured || !reviewSession) {
      const records = loadHumanGold(); records[imageId] = record; saveLocalHumanGold(records); return;
    }
    const { error } = await reviewClient.rpc("submit_annotation", {
      p_batch_id: reviewBatchId,
      p_image_id: imageId,
      p_visible_tags: record.tags,
      p_primary_subject: record.primary || null,
      p_ambiguous: record.ambiguous,
      p_other_tag: record.other_tag || null,
      p_note: record.note || null,
    });
    if (error) throw error;
    await loadRemoteHumanGold();
  }

  function reviewerStatusHtml() {
    if (!reviewBackendConfigured) return `<div class="reviewer-status local"><strong>Local preview</strong><span>Answers stay in this browser until exported.</span></div>`;
    if (reviewBackendState === "loading") return `<div class="reviewer-status"><strong>Connecting…</strong><span>Checking the shared Human Review database.</span></div>`;
    if (reviewSession) return `<div class="reviewer-status synced"><div><strong>Shared anonymous review is active</strong><span>No email is required. This browser has a private reviewer ID and every answer is stored separately.</span></div></div>`;
    return `<div class="reviewer-status local"><strong>Shared saving is unavailable</strong><span>The complete review still works locally. Export JSON or CSV before changing browsers.</span></div>`;
  }

  function wireReviewerStatus() {}

  async function initializeReviewBackend() {
    if (!reviewBackendConfigured) return;
    try {
      const { data, error } = await reviewClient.auth.getSession();
      if (error) throw error;
      reviewSession = data.session;
      if (!reviewSession) {
        const anonymous = await reviewClient.auth.signInAnonymously();
        if (anonymous.error) throw anonymous.error;
        reviewSession = anonymous.data.session;
      }
      await loadRemoteHumanGold();
      reviewBackendState = "ready";
    } catch (error) {
      reviewBackendState = "error";
      console.error("Human Review backend initialization failed", error);
    }
    render();
    reviewClient.auth.onAuthStateChange((_event, session) => {
      reviewSession = session;
      reviewBackendState = "loading";
      setTimeout(async () => {
        try { await loadRemoteHumanGold(); reviewBackendState = "ready"; }
        catch (error) { reviewBackendState = "error"; console.error("Human Review synchronization failed", error); }
        render();
      }, 0);
    });
  }

  function coarseTagsFromPrediction(image, modelKey) {
    const tags = image.tagging?.models?.[modelKey]?.tags || [];
    return [...new Set(tags.map((item) => {
      const raw = String(item.tag || "").toLowerCase();
      if (coarseTagMap[raw]) return coarseTagMap[raw];
      const prefix = Object.keys(coarseTagMap).find((key) => raw === key || raw.startsWith(`${key} `));
      return prefix ? coarseTagMap[prefix] : null;
    }).filter(Boolean))];
  }

  function humanEvaluation(records) {
    const labeled = Object.entries(records).filter(([id, record]) => imagesById.has(id) && record?.tags?.length);
    const byModel = Object.entries(loop.models).map(([modelKey, model]) => {
      let truePositive = 0; let predicted = 0; let truth = 0;
      labeled.forEach(([id, record]) => {
        const human = new Set(record.tags.filter((tag) => tag !== "other"));
        const modelTags = new Set(coarseTagsFromPrediction(imagesById.get(id), modelKey));
        truePositive += [...modelTags].filter((tag) => human.has(tag)).length;
        predicted += modelTags.size; truth += human.size;
      });
      return { modelKey, model, precision: predicted ? truePositive / predicted : 0, completeness: truth ? truePositive / truth : 0 };
    });
    const modes = loop.modes.map((mode) => {
      const memberIds = new Set(loop.human_set.filter((item) => item.error_modes?.includes(mode.id)).map((item) => item.image_id));
      const reviewed = labeled.filter(([id]) => memberIds.has(id));
      let modelError = 0; let annotationAmbiguity = 0;
      reviewed.forEach(([id, record]) => {
        const human = new Set(record.tags.filter((tag) => tag !== "other"));
        const image = imagesById.get(id);
        const anyMismatch = Object.keys(loop.models).some((key) => {
          const predicted = new Set(coarseTagsFromPrediction(image, key));
          return [...predicted].some((tag) => !human.has(tag)) || [...human].some((tag) => !predicted.has(tag));
        });
        if (anyMismatch) modelError += 1;
        const canonical = coarseTagMap[String(image.organ_tag || "").toLowerCase()];
        if ((canonical && !human.has(canonical)) || human.size > 1) annotationAmbiguity += 1;
      });
      return { ...mode, humanReviewed: reviewed.length, modelError, annotationAmbiguity };
    });
    return { labeled, byModel, modes };
  }

  const reproductiveTags = new Set(["flower", "inflorescence", "fruit", "cone", "seed", "bud", "immature fruit"]);
  const woodyTags = new Set(["twig", "branch", "stem"]);

  function relevantModelTags(image, modelKey, accepted = null, limit = 3) {
    const tags = image.tagging?.models?.[modelKey]?.tags || [];
    const filtered = accepted ? tags.filter((item) => accepted(item.tag)) : tags;
    return filtered.slice(0, limit).map((item) => `${item.tag} ${(item.confidence * 100).toFixed(0)}%`);
  }

  function modeConflict(image, modeId) {
    const gemma = new Set(image.tagging?.gemma_normalized_tags || []);
    const modelEntries = Object.entries(loop.models);
    const missingModels = (wanted) => modelEntries.filter(([key]) => !coarseTagsFromPrediction(image, key).some((tag) => wanted.has(tag))).map(([, label]) => label);
    const modelTop = modelEntries.map(([key, label]) => `${label}: ${relevantModelTags(image, key, null, 1)[0] || "none"}`).join(" · ");
    if (modeId === "high_confidence_extra_tag") {
      const extras = modelEntries.flatMap(([key, label]) => (image.tagging.models[key].tags || []).filter((item) => item.confidence >= 0.85 && !gemma.has(item.tag)).slice(0, 2).map((item) => `${label} adds ${item.tag} ${(item.confidence * 100).toFixed(0)}%`));
      return extras.length ? `${extras.join("; ")} · absent from Gemma tags` : "High-confidence model tag is not present in Gemma’s visual tag set.";
    }
    if (modeId === "strong_model_disagreement") return `Top predictions diverge — ${modelTop}`;
    if (modeId === "secondary_woody_structure_missed") {
      const missing = missingModels(new Set(["twig"]));
      return `Gemma sees ${[...gemma].filter((tag) => woodyTags.has(tag)).join(" + ") || "a woody support"}; omitted by ${missing.join(", ") || "one or more models"}.`;
    }
    if (modeId === "reproductive_structure_missed" || modeId === "leaf_dominates_reproductive") {
      const visible = [...gemma].filter((tag) => reproductiveTags.has(tag));
      const missing = missingModels(new Set(["flower", "fruit", "cone", "seed"]));
      return `${visible.length ? `Gemma sees ${visible.join(" + ")}` : `BioImages marks ${image.organ_tag}`}; ${missing.join(", ") || "a model"} omits the reproductive structure${modeId === "leaf_dominates_reproductive" ? " while retaining leaf" : ""}.`;
    }
    if (modeId === "cone_seed_under_detection") {
      const target = image.organ_tag === "cone" || image.organ_tag === "seed" ? image.organ_tag : [...gemma].find((tag) => tag === "cone" || tag === "seed") || "cone/seed";
      return `${target} is the visual target; omitted by ${missingModels(new Set([target])).join(", ") || "one or more models"}.`;
    }
    if (modeId === "whole_plant_scale_split") return `BioImages: ${image.organ_tag} · models split scene scale — ${modelTop}`;
    if (modeId === "fruit_state_view_confusion") {
      const detail = modelEntries.map(([key, label]) => `${label}: ${relevantModelTags(image, key, (tag) => tag.includes("fruit") || tag === "seed", 2).join(" + ") || "no fruit view"}`).join(" · ");
      return `BioImages view: ${image.subview} · ${detail}`;
    }
    if (modeId === "leaf_view_confusion") {
      const detail = modelEntries.map(([key, label]) => `${label}: ${relevantModelTags(image, key, (tag) => tag.startsWith("leaf") || tag.includes("petiole") || tag === "needle", 2).join(" + ") || "no leaf view"}`).join(" · ");
      return `BioImages view: ${image.subview} · ${detail}`;
    }
    if (modeId === "bark_scale_confusion") {
      const detail = modelEntries.map(([key, label]) => `${label}: ${relevantModelTags(image, key, (tag) => tag.includes("bark") || tag === "trunk", 2).join(" + ") || "no bark scale"}`).join(" · ");
      return `BioImages view: ${image.subview} · ${detail}`;
    }
    if (modeId === "flower_fruit_bud_ambiguity") {
      const seen = [...new Set(modelEntries.flatMap(([key]) => relevantModelTags(image, key, (tag) => reproductiveTags.has(tag), 2).map((tag) => tag.replace(/ \d+%$/, ""))))];
      return `BioImages: ${image.organ_tag} · developmental predictions span ${seen.slice(0, 5).join(" ↔ ") || "flower / fruit / bud"}.`;
    }
    if (modeId === "bioimages_single_label_ambiguity") return `BioImages names ${image.organ_tag}; Gemma visibly tags ${(image.tagging.gemma_normalized_tags || []).slice(0, 6).join(" + ")}.`;
    return `BioImages: ${image.tagging.bioimages_label} · ${modelTop}`;
  }

  function evidenceLine(label, content) {
    return `<p><strong>${escapeHtml(label)}</strong><span>${escapeHtml(content || "None")}</span></p>`;
  }

  function modeImageCard(imageId, modeId, compact = false) {
    const image = imagesById.get(imageId);
    if (!image) return "";
    const modelSummary = Object.entries(loop.models).map(([key, label]) => `${label}: ${relevantModelTags(image, key, null, compact ? 1 : 2).join(", ") || "none"}`).join(" · ");
    return `<article class="mode-image-card">
      <a class="mode-photo" href="${url({ view: "image", id: image.id })}" data-route><img src="${escapeHtml(image.image_url)}" alt="${escapeHtml(image.primary_label)}" loading="lazy" decoding="async"></a>
      <div class="mode-image-copy"><p class="mode-species"><i>${escapeHtml(image.species)}</i><span>${escapeHtml(image.id)}</span></p><h4>${escapeHtml(image.organ_category)} · ${escapeHtml(image.subview)}</h4>
        <p class="conflict-line"><strong>Conflict</strong>${escapeHtml(modeConflict(image, modeId))}</p>
        ${evidenceLine("BioImages", image.tagging.bioimages_label)}
        ${evidenceLine("Gemma", (image.tagging.gemma_normalized_tags || []).slice(0, compact ? 5 : 9).join(", "))}
        ${evidenceLine("Models", modelSummary)}
      </div>
    </article>`;
  }

  function renderErrorMode(params) {
    const mode = loop?.modes?.find((item) => item.id === params.get("id"));
    if (!mode) return renderNotFound();
    const allImages = mode.member_image_ids.map((id) => imagesById.get(id)).filter(Boolean);
    const modelOptions = Object.entries(loop.models).map(([key, label]) => `<option value="${key}">${escapeHtml(label)}</option>`).join("");
    const tags = [...new Set(allImages.flatMap((image) => Object.values(image.tagging.models).flatMap((prediction) => prediction.tags.map((item) => item.tag))))].sort();
    setPage(`<div class="wrap mode-page"><p class="crumbs">${routeLink("Error analysis", {})} / ${escapeHtml(mode.name)}</p><div class="page-head"><div><p class="eyebrow">Complete error-mode drill-down</p><h1>${escapeHtml(mode.name)}</h1><p>${escapeHtml(mode.description)}</p></div><strong class="large-count">${nf.format(mode.image_count)} images</strong></div>
      <div class="notice"><strong>What the contradiction means:</strong> ${escapeHtml(mode.cause)} Counts overlap other modes; these are candidates for Human Gold verification, not confirmed errors.</div>
      <dl class="mode-facts mode-page-facts"><div><dt>Most affected</dt><dd>${modelCountLine(mode)}</dd></div><div><dt>Visual corroboration</dt><dd>${mode.vlm_supported} / ${mode.vlm_reviewed} audited examples</dd></div><div><dt>Likely cause</dt><dd>${escapeHtml(mode.vlm_common_cause || mode.cause)}</dd></div></dl>
      <div class="toolbar mode-toolbar"><label>Search<input id="mode-search" type="search" placeholder="Species, image ID, label"></label><label>Affected model<select id="mode-model"><option value="">All models</option>${modelOptions}</select></label><label>Any predicted tag<select id="mode-tag"><option value="">All tags</option>${tags.map((tag) => `<option value="${escapeHtml(tag)}">${escapeHtml(tag)}</option>`).join("")}</select></label></div><p class="results-line" id="mode-count"></p><div id="mode-gallery-all"></div></div>`, mode.name);
    const search = document.querySelector("#mode-search"); const model = document.querySelector("#mode-model"); const tag = document.querySelector("#mode-tag");
    const draw = () => {
      const term = search.value.trim().toLowerCase();
      const filtered = allImages.filter((image) => (!term || `${image.species} ${image.id} ${image.primary_label}`.toLowerCase().includes(term)) && (!model.value || mode.member_models[image.id]?.includes(model.value)) && (!tag.value || Object.values(image.tagging.models).some((prediction) => prediction.tags.some((item) => item.tag === tag.value))));
      document.querySelector("#mode-count").textContent = `${nf.format(filtered.length)} of ${nf.format(allImages.length)} images`;
      const mount = document.querySelector("#mode-gallery-all"); let shown = Math.min(30, filtered.length);
      const paint = () => { mount.innerHTML = `<div class="mode-gallery all-mode-gallery">${filtered.slice(0, shown).map((image) => modeImageCard(image.id, mode.id)).join("")}</div>${shown < filtered.length ? `<button class="load-more" type="button">Show ${Math.min(30, filtered.length - shown)} more</button>` : ""}`; mount.querySelector(".load-more")?.addEventListener("click", () => { shown = Math.min(shown + 30, filtered.length); paint(); }); };
      paint();
    };
    [search, model, tag].forEach((control) => control.addEventListener(control.tagName === "INPUT" ? "input" : "change", draw)); draw();
  }

  function modelCountLine(mode) {
    return Object.entries(mode.model_counts || {}).sort((a, b) => b[1] - a[1])
      .map(([model, count]) => `${escapeHtml(model)} ${nf.format(count)}`).join(" · ");
  }

  function renderHumanMetrics(records) {
    const evaluation = humanEvaluation(records);
    const labeledCount = evaluation.labeled.length;
    if (!labeledCount) return `<div class="pending-evaluation"><strong>Waiting for Human Gold.</strong><p>Complete labels in the Human Review page. This section will then calculate tag precision, truth completeness, and validate the already-discovered modes—without rerunning discovery.</p></div>`;
    const bars = evaluation.byModel.map((row) => `<div class="metric-row"><strong>${escapeHtml(row.model)}</strong><div><span>Tag precision</span><span class="bar"><i style="width:${(row.precision * 100).toFixed(1)}%"></i></span><b>${(row.precision * 100).toFixed(1)}%</b></div><div><span>Truth completeness</span><span class="bar completeness"><i style="width:${(row.completeness * 100).toFixed(1)}%"></i></span><b>${(row.completeness * 100).toFixed(1)}%</b></div></div>`).join("");
    const modeRows = evaluation.modes.filter((mode) => mode.humanReviewed).map((mode) => `<tr><td>${escapeHtml(mode.name)}</td><td class="number">${mode.humanReviewed}</td><td class="number">${mode.modelError}</td><td class="number">${mode.annotationAmbiguity}</td></tr>`).join("");
    return `<p class="notice"><strong>${labeledCount} / 100 reviewed.</strong> Metrics below use only the current browser’s completed Human Gold.</p><div class="metric-stack">${bars}</div><div class="table-wrap"><table><thead><tr><th>Error mode</th><th class="number">Human reviewed</th><th class="number">Model error present</th><th class="number">Annotation ambiguity</th></tr></thead><tbody>${modeRows}</tbody></table></div>`;
  }

  function renderHumanLoopHome() {
    if (!loop) return renderCatalogHome();
    const records = loadHumanGold();
    const completed = Object.values(records).filter((record) => record?.tags?.length).length;
    const storageNote = reviewBackendConfigured
      ? reviewSession
        ? `<strong>Shared anonymous review active:</strong> this browser has a private reviewer ID and every submitted answer is stored separately.`
        : `<strong>Local fallback active:</strong> review immediately and export JSON or CSV before changing browsers.`
      : `<strong>Current GitHub Pages preview:</strong> answers are private to this browser. The Vercel deployment creates an anonymous Supabase reviewer automatically.`;
    const modeBlocks = loop.modes.map((mode, index) => `<article class="error-mode${index >= 8 ? " secondary-mode" : ""}">
      <div class="mode-heading"><div><p class="eyebrow">Error mode ${String(index + 1).padStart(2, "0")}</p><h3>${escapeHtml(mode.name)} — ${nf.format(mode.image_count)} images</h3><p>${escapeHtml(mode.description)}</p></div><span class="mode-count">${nf.format(mode.image_count)}</span></div>
      <dl class="mode-facts"><div><dt>Most affected</dt><dd>${modelCountLine(mode)}</dd></div><div><dt>Visual audit</dt><dd>${mode.vlm_reviewed ? `${mode.vlm_supported} / ${mode.vlm_reviewed} representatives supported` : "VLM audit pending"}</dd></div><div><dt>Likely cause</dt><dd>${escapeHtml(mode.vlm_common_cause || mode.cause)}</dd></div></dl>
      <div class="mode-gallery">${mode.representative_image_ids.slice(0, 9).map((id) => modeImageCard(id, mode.id, true)).join("")}</div>
      <div class="mode-read-all">${routeLink(`View all ${nf.format(mode.image_count)} examples →`, { view: "mode", id: mode.id }, "text-link")}</div>
    </article>`).join("");
    setPage(`<section class="hero human-loop-hero"><div class="wrap"><p class="eyebrow">Error discovery → targeted human verification</p><h1>See how the tagging systems repeatedly fail—before labeling by hand.</h1><p class="lede">Programmatic signals across all 1,899 images identify recurring confusions. Existing image-grounded Gemma tags corroborate high-signal examples, then a diverse 100-image blind review tests which modes are genuine errors and which are single-label ambiguity.</p><div class="stats"><div class="stat"><strong>${nf.format(loop.generated_from_images)}</strong><span>images analyzed</span></div><div class="stat"><strong>${loop.modes.length}</strong><span>recurring modes</span></div><div class="stat"><strong>${nf.format(loop.vlm_audited_images)}</strong><span>VLM-audited examples</span></div><div class="stat"><strong>${loop.human_set.length}</strong><span>targeted human images</span></div><div class="stat"><strong>${completed}</strong><span>Human Gold complete</span></div></div></div></section>
    <section class="section"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Main error modes</p><h2>Repeated patterns found in the Full Gallery</h2><p>Counts are overlapping mode memberships, not mutually exclusive classes. Representative images favor species and individual diversity; visual corroboration reuses Gemma tags from the high-signal subset.</p></div></div><div class="mode-list">${modeBlocks}</div><button class="load-more" id="show-all-modes" type="button">Show all ${loop.modes.length} modes</button></div></section>
    <section class="section" id="human-set"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Human Review Set</p><h2>100 targeted, blind multi-label decisions</h2><p>The set covers major modes, every major tag, easy agreements, strong disagreements, rare classes, and high-confidence errors while maximizing species and individual diversity.</p></div>${routeLink(completed ? "Continue review →" : "Start blind review →", { view: "review" }, "text-link")}</div><div class="review-summary"><strong>${completed} / 100 complete</strong><span>85 species · 100 individuals · selection reason saved for every image</span></div><div class="deployment-note">${storageNote} <a href="VERCEL_HUMAN_REVIEW.md">Deployment details</a>.</div></div></section>
    <section class="section"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Which error modes are real?</p><h2>Human validation, updated locally</h2><p>Model error and annotation ambiguity are measured only after a blind answer is saved.</p></div></div><div id="human-evaluation">${renderHumanMetrics(records)}</div></div></section>
    <section class="section"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Drill down</p><h2>Full Gallery remains available</h2><p>Inspect all good matches, BioImages mismatches, Gemma disagreements, and model disagreements with the original filters.</p></div>${routeLink("Open Full Gallery →", { view: "full-gallery" }, "text-link")}</div></div></section>`, "Error analysis");
    document.querySelector("#show-all-modes")?.addEventListener("click", (event) => {
      document.querySelectorAll(".secondary-mode").forEach((el) => el.classList.remove("secondary-mode"));
      event.currentTarget.remove();
    });
  }

  function downloadText(filename, textValue, type) {
    const blob = new Blob([textValue], { type });
    const href = URL.createObjectURL(blob); const anchor = document.createElement("a");
    anchor.href = href; anchor.download = filename; anchor.click(); URL.revokeObjectURL(href);
  }

  function exportHumanGold(format) {
    const records = loadHumanGold();
    const rows = loop.human_set.map((entry) => ({ ...entry, ...(records[entry.image_id] || {}) }));
    if (format === "json") return downloadText("bioimages-human-gold.json", JSON.stringify(rows, null, 2), "application/json");
    const headers = ["review_index", "image_id", "species", "individual_id", "visible_tags", "primary_subject", "ambiguous", "other_tag", "note", "selection_reason", "labeled_at"];
    const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const csv = [headers.join(","), ...rows.map((row) => headers.map((key) => quote(key === "visible_tags" ? (row.tags || []).join("|") : row[key])).join(","))].join("\n");
    downloadText("bioimages-human-gold.csv", csv, "text/csv;charset=utf-8");
  }

  function referenceReveal(image) {
    return `<div class="reveal-grid"><div><h3>BioImages</h3><p>${escapeHtml(image.tagging.bioimages_label)}</p></div><div><h3>Gemma</h3>${tagPills(image.tagging.gemma_tags)}</div>${Object.entries(loop.models).map(([key, label]) => `<div><h3>${escapeHtml(label)}</h3>${tagPills(image.tagging.models[key].tags, true)}</div>`).join("")}</div>`;
  }

  function renderHumanReview(params) {
    if (!loop) return renderNotFound();
    const requested = Number(params.get("n") || 1);
    const index = Math.max(0, Math.min(loop.human_set.length - 1, requested - 1));
    const entry = loop.human_set[index]; const image = imagesById.get(entry.image_id); const records = loadHumanGold(); const saved = records[entry.image_id];
    const completed = Object.values(records).filter((record) => record?.tags?.length).length;
    if (reviewBackendConfigured && reviewBackendState === "loading") {
      setPage(`<div class="wrap review-page"><div class="review-top"><div><p class="eyebrow">Blind Human Review</p><h1>Preparing the shared 100-image review</h1><p>No email is required. A private anonymous reviewer identity is created in this browser.</p></div></div>${reviewerStatusHtml()}<div class="pending-evaluation"><strong>Your labels remain private during blind review.</strong><p>BioImages, Gemma, DINOv3, BioCLIP, and EfficientNet answers appear only after you submit each image.</p></div></div>`, "Preparing Human review");
      wireReviewerStatus(); return;
    }
    setPage(`<div class="wrap review-page"><div class="review-top"><div><p class="eyebrow">Blind Human Review</p><h1>Image ${index + 1} of ${loop.human_set.length}</h1><p>${completed} complete · model and reference answers remain hidden until this image is submitted.</p></div><div class="review-actions"><button type="button" id="export-json">Export JSON</button><button type="button" id="export-csv">Export CSV</button></div></div>${reviewerStatusHtml()}<div class="review-progress"><i style="width:${completed}%"></i></div>
      <div class="label-workspace"><div class="label-image"><img src="${escapeHtml(image.image_url)}" alt="Plant photograph for blind labeling"></div><form id="human-form" class="label-form"><fieldset><legend>Which tags are truly visible?</legend><p class="form-help">Select any number. Judge the pixels, not what you expect from the species.</p><div class="tag-checks">${loop.human_tags.map((tag) => `<label><input type="checkbox" name="tag" value="${escapeHtml(tag)}" ${saved?.tags?.includes(tag) ? "checked" : ""}><span>${escapeHtml(tag)}</span></label>`).join("")}</div></fieldset><label>Primary subject <span>(optional)</span><select name="primary"><option value="">Not specified</option>${loop.human_tags.map((tag) => `<option value="${escapeHtml(tag)}" ${saved?.primary === tag ? "selected" : ""}>${escapeHtml(tag)}</option>`).join("")}</select></label><label class="check-line"><input type="checkbox" name="ambiguous" ${saved?.ambiguous ? "checked" : ""}> Ambiguous / genuinely difficult</label><label>Other tag <span>(optional)</span><input name="other_tag" value="${escapeHtml(saved?.other_tag || "")}" placeholder="Visible structure not listed"></label><label>Short note <span>(optional)</span><textarea name="note" rows="3" placeholder="Why this is difficult or what is visible">${escapeHtml(saved?.note || "")}</textarea></label><p class="form-error" id="form-error" role="alert"></p><button class="submit-label" type="submit">${saved ? "Update answer and reveal" : "Submit answer and reveal"}</button><p class="selection-rationale"><strong>Why selected:</strong> ${escapeHtml(entry.selection_reason)}</p></form></div>
      <section class="answer-reveal ${saved ? "visible" : ""}" id="answer-reveal"><p class="eyebrow">Shown only after submission</p><h2>References and model answers</h2>${referenceReveal(image)}</section><div class="review-nav">${index ? routeLink("← Previous", { view: "review", n: index }, "text-link") : "<span></span>"}<span>${index + 1} / ${loop.human_set.length}</span>${index + 1 < loop.human_set.length ? routeLink("Next →", { view: "review", n: index + 2 }, "text-link") : routeLink("Return to results →", {}, "text-link")}</div></div>`, `Human review ${index + 1}`);
    wireReviewerStatus();
    document.querySelector("#export-json").addEventListener("click", () => exportHumanGold("json"));
    document.querySelector("#export-csv").addEventListener("click", () => exportHumanGold("csv"));
    document.querySelector("#human-form").addEventListener("submit", async (event) => {
      event.preventDefault(); const form = new FormData(event.currentTarget); const tags = form.getAll("tag");
      if (!tags.length) { document.querySelector("#form-error").textContent = "Select at least one visible tag."; return; }
      const button = event.currentTarget.querySelector("button[type=submit]");
      const record = { tags, primary: form.get("primary"), ambiguous: form.get("ambiguous") === "on", other_tag: form.get("other_tag").trim(), note: form.get("note").trim(), labeled_at: new Date().toISOString() };
      button.disabled = true; document.querySelector("#form-error").textContent = reviewBackendConfigured ? "Saving securely…" : "Saving…";
      try {
        await saveHumanGoldEntry(entry.image_id, record);
        records[entry.image_id] = record;
        document.querySelector("#form-error").textContent = reviewBackendConfigured ? "Saved to the shared Human Gold database." : "Saved in this browser.";
        document.querySelector("#answer-reveal").classList.add("visible");
        document.querySelector("#answer-reveal").scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        document.querySelector("#form-error").textContent = `Could not save: ${error.message}`;
      } finally { button.disabled = false; }
    });
  }

  function renderTaggingHome() {
    const analysis = data.tagging_analysis;
    const models = analysis.summary;
    const modelKeys = models.map((row) => row.model_key);
    const allEntries = data.images.flatMap((image) => modelKeys.map((modelKey) => ({ image, modelKey })));
    const goodEntries = allEntries
      .filter(({ image, modelKey }) => analysisCategories(image, modelKey).includes("good match"))
      .sort((a, b) => b.image.tagging.models[b.modelKey].gemma_f1 - a.image.tagging.models[a.modelKey].gemma_f1);
    const issueEntries = allEntries
      .filter(({ image, modelKey }) => analysisCategories(image, modelKey).some((category) => category !== "good match"))
      .sort((a, b) => {
        const ap = a.image.tagging.models[a.modelKey]; const bp = b.image.tagging.models[b.modelKey];
        return Number(ap.bioimages_match) - Number(bp.bioimages_match) || ap.gemma_jaccard - bp.gemma_jaccard || a.image.tagging.mean_pairwise_jaccard - b.image.tagging.mean_pairwise_jaccard;
      });
    const modelOptions = models.map((row) => `<option value="${row.model_key}">${escapeHtml(row.model)}</option>`).join("");
    const tagOptions = analysis.vocabulary.map((item) => `<option value="${escapeHtml(item.tag)}">${escapeHtml(item.tag)}</option>`).join("");
    const speciesOptions = data.species.map((item) => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`).join("");
    const summaryRows = models.map((row) => `<tr><td>${escapeHtml(row.model)}</td><td class="number">${(row.bioimages_match_rate * 100).toFixed(1)}%</td><td class="number">${(row.top1_exact_match * 100).toFixed(1)}%</td><td class="number">${(row.gemma_micro_f1 * 100).toFixed(1)}%</td><td class="number">${(row.gemma_mean_jaccard * 100).toFixed(1)}%</td><td class="number">${row.mean_predicted_tags.toFixed(1)}</td></tr>`).join("");
    const bestBioImages = [...models].sort((a, b) => b.bioimages_match_rate - a.bioimages_match_rate)[0];
    const bestGemma = [...models].sort((a, b) => b.gemma_micro_f1 - a.gemma_micro_f1)[0];
    const headlineConclusion = bestBioImages.model_key === bestGemma.model_key
      ? `${bestBioImages.model} leads both comparisons: ${(bestBioImages.bioimages_match_rate * 100).toFixed(1)}% BioImages match and ${(bestGemma.gemma_micro_f1 * 100).toFixed(1)}% held-out Gemma agreement.`
      : `${bestBioImages.model} leads BioImages match; ${bestGemma.model} leads held-out Gemma agreement.`;
    setPage(`<section class="hero tagging-hero"><div class="wrap">
      <p class="eyebrow">Full-corpus multi-label tagging</p><h1>What each model sees in every BioImages photograph.</h1>
      <p class="lede">DINOv3, BioCLIP 2.5, and EfficientNet-B0 each tag all 1,899 images with one or more visible structures and views. BioImages supplies one canonical reference label; Gemma supplies multi-label teacher/reference annotations.</p>
      <div class="stats"><div class="stat"><strong>1,899</strong><span>images tagged per model</span></div><div class="stat"><strong>3</strong><span>frozen visual models</span></div><div class="stat"><strong>${analysis.vocabulary.length}</strong><span>auditable tags</span></div><div class="stat"><strong>${nf.format(analysis.gemma_images)}</strong><span>Gemma references</span></div><div class="stat"><strong>0</strong><span>fine-tuned backbones</span></div></div>
    </div></section>
    <section class="section"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Headline comparison</p><h2>Which model produces the most useful tags?</h2><p>BioImages match asks whether its single coarse label appears anywhere in the model’s multi-label set. Gemma agreement is micro multi-label F1 on normalized tags.</p></div></div>
      <div class="table-wrap"><table><thead><tr><th>Model</th><th class="number">BioImages match</th><th class="number">Top-1 exact</th><th class="number">Gemma agreement</th><th class="number">Mean Jaccard</th><th class="number">Tags / image</th></tr></thead><tbody>${summaryRows}</tbody></table></div>
      <p class="notice result-note"><strong>Result:</strong> ${escapeHtml(headlineConclusion)}</p>
      <p class="method-note">${escapeHtml(analysis.method.name)}. Every reported prediction is out-of-fold by <code>individual_id</code>; Gemma labels for the held-out image never enter its model fit, and no backbone was fine-tuned. Gemma is not human ground truth.</p>
    </div></section>
    <section class="section"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Tag-level results</p><h2>Performance by visible structure and view</h2><p>BioImages match is available for its coarse canonical structures. Gemma F1 evaluates both structures and finer view tags.</p></div><label class="inline-filter">Model<select id="per-tag-model">${modelOptions}</select></label></div><div class="table-wrap"><table><thead><tr><th>Tag</th><th>Type</th><th class="number">BioImages support</th><th class="number">BioImages match</th><th class="number">Gemma support</th><th class="number">Gemma precision</th><th class="number">Gemma recall</th><th class="number">Gemma F1</th></tr></thead><tbody id="per-tag-body"></tbody></table></div></div></section>
    <section class="section"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Good matches</p><h2>Images where references and model agree</h2><p>BioImages canonical tag is present and Gemma multi-label F1 is at least 0.67.</p></div><span class="count" id="good-count"></span></div>
      <div class="toolbar compact-toolbar"><label>Model<select id="good-model"><option value="">All models</option>${modelOptions}</select></label><label>Tag<select id="good-tag"><option value="">All tags</option>${tagOptions}</select></label><label>Species<select id="good-species"><option value="">All species</option>${speciesOptions}</select></label></div><div id="good-gallery"></div></div></section>
    <section class="section"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Error and disagreement gallery</p><h2>Inspect every important mismatch</h2><p>Filter the full set of BioImages misses, low Gemma overlap, and cross-model disagreements. Click any image for all three tag sets.</p></div><span class="count" id="issue-count"></span></div>
      <div class="toolbar analysis-toolbar"><label>Issue<select id="issue-category"><option value="">Any issue</option><option>BioImages mismatch</option><option>Gemma disagreement</option><option>model disagreement</option></select></label><label>Model<select id="issue-model"><option value="">All models</option>${modelOptions}</select></label><label>Tag<select id="issue-tag"><option value="">Any predicted tag</option>${tagOptions}</select></label><label>Species<select id="issue-species"><option value="">All species</option>${speciesOptions}</select></label></div><div id="issue-gallery"></div>
    </div></section>
    <section class="section"><div class="wrap"><div class="section-head"><div><p class="eyebrow">Collection browser</p><h2>Continue browsing BioImages</h2><p>The species, organ, view, and individual indexes are unchanged.</p></div>${routeLink("Browse organs and views →", { view: "organs" }, "text-link")}</div><div class="organ-grid">${data.organ_groups.filter((g) => g.slug !== "other").map(organCard).join("")}</div></div></section>`, "Tagging analysis");

    const vocabByTag = new Map(analysis.vocabulary.map((item) => [item.tag, item]));
    const perTagSelect = document.querySelector("#per-tag-model");
    const drawPerTag = () => {
      const rows = analysis.per_tag.filter((row) => row.model_key === perTagSelect.value);
      document.querySelector("#per-tag-body").innerHTML = rows.map((row) => `<tr><td>${escapeHtml(row.tag)}</td><td>${escapeHtml(vocabByTag.get(row.tag)?.kind || "")}</td><td class="number">${row.bioimages_support || "—"}</td><td class="number">${row.bioimages_match_rate === "" ? "—" : (row.bioimages_match_rate * 100).toFixed(1) + "%"}</td><td class="number">${row.gemma_support}</td><td class="number">${(row.gemma_precision * 100).toFixed(1)}%</td><td class="number">${(row.gemma_recall * 100).toFixed(1)}%</td><td class="number">${(row.gemma_f1 * 100).toFixed(1)}%</td></tr>`).join("");
    };
    perTagSelect.addEventListener("change", drawPerTag); drawPerTag();

    const wireGallery = (prefix, entries, issueMode = false) => {
      const model = document.querySelector(`#${prefix}-model`); const tag = document.querySelector(`#${prefix}-tag`); const species = document.querySelector(`#${prefix}-species`); const category = issueMode ? document.querySelector("#issue-category") : null;
      const draw = () => {
        const filtered = entries.filter((entry) => {
          const tags = entry.image.tagging.models[entry.modelKey].tags.map((item) => item.tag);
          const categories = analysisCategories(entry.image, entry.modelKey);
          return (!model.value || entry.modelKey === model.value) && (!tag.value || tags.includes(tag.value)) && (!species.value || entry.image.species === species.value) && (!category || !category.value || categories.includes(category.value));
        });
        document.querySelector(`#${prefix}-count`).textContent = `${nf.format(filtered.length)} model–image results`;
        mountAnalysisGallery(document.querySelector(`#${prefix}-gallery`), filtered);
      };
      [model, tag, species, category].filter(Boolean).forEach((control) => control.addEventListener("change", draw)); draw();
    };
    wireGallery("good", goodEntries); wireGallery("issue", issueEntries, true);
  }

  function renderHome() {
    if (loop) renderHumanLoopHome();
    else if (data.tagging_analysis?.available) renderTaggingHome();
    else renderCatalogHome();
  }

  function renderCatalogHome() {
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
    const predictionHtml = prediction && !image.tagging ? `<div class="prediction"><h3>Auxiliary model prediction</h3>
      <p>Predicted organ: <strong>${escapeHtml(prediction.label)}</strong> · ${(prediction.confidence * 100).toFixed(1)}% confidence</p>
      <p>BioImages organ: <strong>${escapeHtml(image.organ_category)}</strong> · <span class="${prediction.match ? "match" : "mismatch"}">${prediction.match ? "match" : "mismatch"}</span></p>
      <p class="image-meta">DINOv3 frozen linear probe, ${escapeHtml(prediction.protocol)}. The model does not predict the fine subview.</p></div>` : "";
    const taggingHtml = image.tagging ? `<section class="detail-tagging"><p class="eyebrow">Full-corpus multi-label analysis</p>
      <div class="reference-block"><h3>BioImages canonical annotation</h3><p><strong>${escapeHtml(image.tagging.bioimages_label)}</strong>${image.tagging.bioimages_canonical_tag ? ` · comparison tag <span class="tag-pill">${escapeHtml(image.tagging.bioimages_canonical_tag)}</span>` : " · not evaluable as a visual tag"}</p></div>
      <div class="reference-block"><h3>Gemma free visual tags</h3>${tagPills(image.tagging.gemma_tags)}<p class="image-meta">Normalized for scoring: ${escapeHtml(image.tagging.gemma_normalized_tags.join(", "))}</p></div>
      <div class="model-tag-stack">${Object.values(image.tagging.models).map((item) => `<div class="model-tag-row"><div><h3>${escapeHtml(item.model)}</h3><p class="image-meta">${item.bioimages_match === null ? "BioImages not evaluable" : item.bioimages_match ? "BioImages match ✓" : "BioImages mismatch ✗"} · Gemma F1 ${(item.gemma_f1 * 100).toFixed(0)}% · Jaccard ${(item.gemma_jaccard * 100).toFixed(0)}%</p></div>${tagPills(item.tags, true)}</div>`).join("")}</div>
      ${image.tagging.model_disagreement ? `<p class="notice review"><strong>Model disagreement:</strong> mean pairwise tag-set Jaccard ${(image.tagging.mean_pairwise_jaccard * 100).toFixed(0)}%.</p>` : ""}
    </section>` : "";
    const reviewHtml = image.review_flags.length ? `<p class="notice review"><strong>Manual metadata review:</strong> ${escapeHtml(image.review_flags.join(", "))}.</p>` : "";
    setPage(`<div class="wrap"><p class="crumbs">${routeLink("Full gallery", { view: "full-gallery" })} / ${escapeHtml(image.id)}</p>
      <div class="detail"><div class="detail-image"><a href="${escapeHtml(image.image_url)}"><img src="${escapeHtml(image.image_url)}" alt="${escapeHtml(image.title)}"></a></div>
      <div class="detail-copy"><p class="eyebrow">Image detail</p><h1><i>${escapeHtml(image.species)}</i></h1><p>${escapeHtml(image.common_name)}</p>
        <p class="primary-label">${escapeHtml(image.organ_category)} · ${escapeHtml(image.subview)}</p>${reviewHtml}${taggingHtml}${predictionHtml}
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
    else if (view === "mode") renderErrorMode(params);
    else if (view === "review") renderHumanReview(params);
    else if (view === "full-gallery") renderTaggingHome();
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
  initializeReviewBackend();
})();
