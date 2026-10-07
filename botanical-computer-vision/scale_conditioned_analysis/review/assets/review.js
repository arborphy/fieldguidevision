(() => {
  "use strict";

  const data = window.BIOIMAGES_SCALE_REVIEW;
  const runtime = window.BIOIMAGES_RUNTIME || {};
  if (!data?.items?.length) {
    document.body.innerHTML = '<p style="padding:3rem">The scale-review data could not be loaded.</p>';
    return;
  }

  const STORAGE_KEY = "bioimages-scale-review-v1";
  const labelColors = {
    distant: "var(--distant)",
    "mid-range": "var(--mid-range)",
    "close-up": "var(--close-up)",
    uncertain: "var(--uncertain)",
  };
  const labelHints = {
    distant: "Whole organism, tree, shrub, or most of the crown",
    "mid-range": "Connected branch, trunk section, or several organs",
    "close-up": "One organ, surface, or fine detail dominates",
    uncertain: "Rules conflict, boundary case, or insufficient evidence",
  };
  const configured = Boolean(runtime.supabaseUrl && runtime.supabaseAnonKey);
  let client = null;
  const batchId = runtime.scaleReviewBatchId || data.batch_id;
  let session = null;
  let backendState = configured ? "connecting" : "local";
  let backendMessage = configured ? "Creating a private anonymous reviewer identity…" : "Answers are stored in this browser and can be exported.";
  let records = loadLocalRecords();
  let currentIndex = initialIndex();

  function loadSupabaseSdk() {
    if (window.supabase?.createClient) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      const timeout = window.setTimeout(() => reject(new Error("shared-service connection timed out")), 8000);
      script.src = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2";
      script.async = true;
      script.onload = () => {
        window.clearTimeout(timeout);
        if (window.supabase?.createClient) resolve();
        else reject(new Error("shared-service client did not initialize"));
      };
      script.onerror = () => {
        window.clearTimeout(timeout);
        reject(new Error("shared-service client could not be loaded"));
      };
      document.head.append(script);
    });
  }

  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));

  function loadLocalRecords() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
    catch (_) { return {}; }
  }

  function persistLocalRecords() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  }

  function initialIndex() {
    const requested = Number(new URLSearchParams(location.search).get("n"));
    if (Number.isFinite(requested) && requested >= 1 && requested <= data.items.length) return requested - 1;
    const firstOpen = data.items.findIndex((item) => !records[item.image_id]?.human_scale);
    return firstOpen >= 0 ? firstOpen : 0;
  }

  function setUrlIndex(index) {
    const target = new URL(location.href);
    target.searchParams.set("n", String(index + 1));
    history.replaceState({}, "", target);
  }

  function completedCount() {
    return data.items.filter((item) => records[item.image_id]?.human_scale).length;
  }

  function renderStatus() {
    let title = "Local review is ready";
    let className = "status offline";
    if (backendState === "connecting") {
      title = "Connecting shared review";
      className = "status";
    } else if (backendState === "synced") {
      title = "Shared anonymous review is active";
      className = "status";
    } else if (backendState === "error") {
      title = "Shared saving is unavailable; local review still works";
    }
    $("#review-status").innerHTML = `<div class="${className}"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(backendMessage)}</span></div>`;
  }

  function renderCodebook() {
    $("#codebook-grid").innerHTML = data.labels.map((label) => {
      const examples = (data.examples[label] || []).map((item) => `<a href="${escapeHtml(item.source_url)}" target="_blank" rel="noreferrer" title="${escapeHtml(item.caption)}"><img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.species)}"><span>${escapeHtml(item.species)}</span></a>`).join("");
      return `<article class="code-card" style="--scale-color:${labelColors[label]}"><div class="code-copy"><h3>${escapeHtml(label)}</h3><p>${escapeHtml(data.definitions[label])}</p></div><div class="mini-examples">${examples}</div></article>`;
    }).join("");
  }

  function renderProgress() {
    const complete = completedCount();
    $("#progress-count").textContent = `${complete} / ${data.items.length}`;
    $("#progress-bar").style.width = `${(complete / data.items.length) * 100}%`;
  }

  function scaleChoices(saved) {
    return data.labels.map((label, index) => `<label class="scale-choice" style="--scale-color:${labelColors[label]}"><input type="radio" name="human_scale" value="${escapeHtml(label)}" ${saved?.human_scale === label ? "checked" : ""}><span class="choice-number">${index + 1}</span><span class="choice-copy"><strong>${escapeHtml(label)}</strong><small>${escapeHtml(labelHints[label])}</small></span><kbd>${index + 1}</kbd></label>`).join("");
  }

  function evidenceReveal(item, saved) {
    if (!saved?.human_scale) return "";
    const agrees = saved.human_scale === item.provisional_scale;
    const evidence = Object.entries(item.evidence).map(([label, values]) => {
      const lines = values.filter(Boolean);
      return `<div><strong>${escapeHtml(label)}</strong>${lines.length ? lines.map((line) => `<p>${escapeHtml(line)}</p>`).join("") : '<p>No positive evidence recorded.</p>'}</div>`;
    }).join("");
    return `<p class="eyebrow">Revealed after your answer</p><h3>Provisional label and evidence</h3><div class="reveal-summary"><span class="reveal-pill">Your answer: <b>${escapeHtml(saved.human_scale)}</b></span><span class="reveal-pill">Provisional: <b>${escapeHtml(item.provisional_scale)}</b> · ${escapeHtml(item.provisional_confidence)}</span><span class="reveal-pill">${agrees ? "Agreement" : "Disagreement—keep for adjudication"}</span></div><p>${escapeHtml(item.rationale)}</p><div class="evidence-grid">${evidence}</div>`;
  }

  function renderQueue() {
    $("#queue-grid").innerHTML = data.items.map((item, index) => {
      const done = Boolean(records[item.image_id]?.human_scale);
      const current = index === currentIndex;
      return `<button class="queue-item${done ? " done" : ""}${current ? " current" : ""}" type="button" data-index="${index}" title="${index + 1}. ${escapeHtml(item.species)} · ${escapeHtml(item.organ_category)}">${index + 1}</button>`;
    }).join("");
    $("#queue-grid").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => navigateTo(Number(button.dataset.index))));
  }

  function renderCurrent() {
    const item = data.items[currentIndex];
    const saved = records[item.image_id];
    setUrlIndex(currentIndex);
    $("#review-position").textContent = `Image ${currentIndex + 1} of ${data.items.length}`;
    $("#review-title").innerHTML = `<i>${escapeHtml(item.species)}</i>`;
    $("#review-meta").textContent = `${item.organ_category} · ${item.subview} · ${item.image_id}`;
    $("#review-image").src = item.image_url;
    $("#review-image").alt = `${item.species}; classify the visible photographic scale`;
    $("#image-link").href = item.source_url;
    $("#scale-choices").innerHTML = scaleChoices(saved);
    const form = $("#review-form");
    form.elements.confidence.value = saved?.confidence || "medium";
    form.elements.note.value = saved?.note || "";
    $("#form-message").textContent = saved ? "Saved. You may revise this answer." : "";
    $("#form-message").className = "form-message";
    const reveal = $("#machine-reveal");
    reveal.hidden = !saved?.human_scale;
    reveal.innerHTML = evidenceReveal(item, saved);
    $("#previous-item").disabled = currentIndex === 0;
    $("#next-item").disabled = currentIndex === data.items.length - 1;
    renderProgress();
    renderQueue();
  }

  function navigateTo(index, focus = true) {
    currentIndex = Math.max(0, Math.min(data.items.length - 1, index));
    renderCurrent();
    if (focus) $("#review-workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function nextUnreviewed() {
    const total = data.items.length;
    for (let offset = 1; offset <= total; offset += 1) {
      const index = (currentIndex + offset) % total;
      if (!records[data.items[index].image_id]?.human_scale) return navigateTo(index);
    }
    $("#form-message").textContent = "All 92 samples are reviewed. Export the results or revise any answer.";
  }

  async function saveRemote(item, record) {
    if (!client || !session || backendState !== "synced") return;
    const { error } = await client.rpc("submit_scale_annotation", {
      p_batch_id: batchId,
      p_image_id: item.image_id,
      p_human_scale: record.human_scale,
      p_confidence: record.confidence,
      p_note: record.note || null,
    });
    if (error) throw error;
  }

  async function submitCurrent(event) {
    event.preventDefault();
    const item = data.items[currentIndex];
    const formData = new FormData(event.currentTarget);
    const humanScale = formData.get("human_scale");
    if (!humanScale) {
      $("#form-message").textContent = "Choose one of the four photographic-scale labels.";
      $("#form-message").className = "form-message error";
      return;
    }
    const record = {
      human_scale: humanScale,
      confidence: formData.get("confidence") || "medium",
      note: String(formData.get("note") || "").trim(),
      human_agrees: humanScale === item.provisional_scale,
      labeled_at: new Date().toISOString(),
    };
    const button = event.currentTarget.querySelector("button[type=submit]");
    button.disabled = true;
    $("#form-message").className = "form-message";
    $("#form-message").textContent = backendState === "synced" ? "Saving locally and to the shared review…" : "Saving in this browser…";
    records[item.image_id] = record;
    persistLocalRecords();
    try {
      await saveRemote(item, record);
      $("#form-message").textContent = backendState === "synced" ? "Saved to the shared review." : "Saved in this browser.";
    } catch (error) {
      backendState = "error";
      backendMessage = `The shared database rejected this save (${error.message}). Your answer is safe in this browser and can be exported.`;
      $("#form-message").textContent = "Saved locally; shared save failed.";
      $("#form-message").className = "form-message error";
      renderStatus();
    } finally {
      button.disabled = false;
      renderProgress();
      renderQueue();
      const reveal = $("#machine-reveal");
      reveal.innerHTML = evidenceReveal(item, record);
      reveal.hidden = false;
      reveal.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function rowsForExport() {
    return data.items.map((item) => ({
      review_index: item.index,
      image_id: item.image_id,
      species: item.species,
      organ_category: item.organ_category,
      subview: item.subview,
      provisional_scale: item.provisional_scale,
      provisional_confidence: item.provisional_confidence,
      human_scale: records[item.image_id]?.human_scale || "",
      human_confidence: records[item.image_id]?.confidence || "",
      human_agrees: records[item.image_id]?.human_agrees ?? "",
      review_note: records[item.image_id]?.note || "",
      labeled_at: records[item.image_id]?.labeled_at || "",
    }));
  }

  function download(filename, content, type) {
    const blob = new Blob([content], { type });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function exportJson() {
    download("bioimages-scale-review.json", JSON.stringify(rowsForExport(), null, 2), "application/json");
  }

  function exportCsv() {
    const rows = rowsForExport();
    const headers = Object.keys(rows[0]);
    const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const csv = [headers.join(","), ...rows.map((row) => headers.map((key) => quote(row[key])).join(","))].join("\n");
    download("bioimages-scale-review.csv", csv, "text/csv;charset=utf-8");
  }

  async function initializeBackend() {
    if (!configured) {
      renderStatus();
      return;
    }
    try {
      await loadSupabaseSdk();
      client = window.supabase.createClient(runtime.supabaseUrl, runtime.supabaseAnonKey, {
        auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: false },
      });
      let { data: authData, error } = await client.auth.getSession();
      if (error) throw error;
      session = authData.session;
      if (!session) {
        ({ data: authData, error } = await client.auth.signInAnonymously());
        if (error) throw error;
        session = authData.session;
      }
      const response = await client.from("scale_annotations")
        .select("image_id,human_scale,confidence,note,updated_at,submitted_at")
        .eq("batch_id", batchId);
      if (response.error) throw response.error;
      for (const row of response.data || []) {
        records[row.image_id] = {
          human_scale: row.human_scale,
          confidence: row.confidence,
          note: row.note || "",
          human_agrees: row.human_scale === data.items.find((item) => item.image_id === row.image_id)?.provisional_scale,
          labeled_at: row.updated_at || row.submitted_at,
        };
      }
      persistLocalRecords();
      backendState = "synced";
      backendMessage = "No email is required. This browser has a private reviewer ID, and answers resume here automatically.";
    } catch (error) {
      backendState = "error";
      backendMessage = `Anonymous shared access is not enabled yet (${error.message}). The complete interface remains usable with local save and export.`;
    }
    renderStatus();
    renderCurrent();
  }

  function handleKeyboard(event) {
    const tag = event.target.tagName;
    if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
    if (["1", "2", "3", "4"].includes(event.key)) {
      const label = data.labels[Number(event.key) - 1];
      const input = $(`#review-form input[name="human_scale"][value="${label}"]`);
      if (input) { input.checked = true; input.focus(); }
    } else if (event.key === "ArrowLeft") navigateTo(currentIndex - 1);
    else if (event.key === "ArrowRight") navigateTo(currentIndex + 1);
    else if (event.key.toLowerCase() === "s") nextUnreviewed();
    else if (event.key === "Enter") $("#review-form").requestSubmit();
  }

  renderCodebook();
  renderStatus();
  renderCurrent();
  $("#review-form").addEventListener("submit", submitCurrent);
  $("#previous-item").addEventListener("click", () => navigateTo(currentIndex - 1));
  $("#next-item").addEventListener("click", () => navigateTo(currentIndex + 1));
  $("#next-unreviewed").addEventListener("click", nextUnreviewed);
  $("#export-json").addEventListener("click", exportJson);
  $("#export-csv").addEventListener("click", exportCsv);
  window.addEventListener("keydown", handleKeyboard);
  initializeBackend();
})();
