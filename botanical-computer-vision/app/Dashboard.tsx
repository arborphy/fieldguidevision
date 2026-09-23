"use client";

import { useMemo, useState } from "react";

type TopPrediction = { species: string; confidence: number };
type Outcome = "top1" | "top5" | "miss";
type ImageResult = {
  id: string;
  truth: string;
  vernacular: string;
  family: string;
  organ: string;
  organ_detail: string;
  subview: string;
  predicted: string;
  confidence: number;
  top5: TopPrediction[];
  rank: number | null;
  outcome: Outcome;
  thumbnail: string;
  image: string;
  creator: string;
  license: string;
};
type SpeciesResult = {
  name: string;
  vernacular: string;
  total: number;
  top1: number;
  top1_accuracy: number;
  top5: number;
  top5_accuracy: number;
};
type OrganResult = SpeciesResult & { name: string };
type DashboardData = {
  summary: Record<string, string | number | boolean>;
  outcomes: Record<Outcome, number>;
  species: SpeciesResult[];
  organs: OrganResult[];
  confidence: { label: string; total: number; accuracy: number; average_confidence: number }[];
  errors: { truth: string; predicted: string; count: number }[];
  images: ImageResult[];
};
type ComparisonModel = {
  model: string;
  method: string;
  top1_correct: number;
  top1_accuracy: number;
  top5_correct: number;
  top5_accuracy: number;
  macro_top1_accuracy: number;
};
type ComparisonRow = {
  total: number;
  bioclip_zero_top1_accuracy: number;
  bioclip_zero_top5_accuracy: number;
  bioclip_probe_top1_accuracy: number;
  bioclip_probe_top5_accuracy: number;
  dinov3_probe_top1_accuracy: number;
  dinov3_probe_top5_accuracy: number;
  top1_delta_probe_minus_zero: number;
  top1_delta_bioclip_probe_minus_dinov3: number;
};
type PairwiseResult = {
  both_correct: number;
  first_only: number;
  second_only: number;
  both_wrong: number;
  same_prediction: number;
  top1_delta: number;
  top5_delta: number;
  top1_paired_bootstrap_95_ci: number[];
  top5_paired_bootstrap_95_ci: number[];
  exact_mcnemar_p: number;
};
type StrictComparison = {
  summary: {
    protocol: { test_images: number; species: number; individual_disjoint: boolean; candidate_set_identical: boolean };
    bioclip_zero_shot: ComparisonModel;
    bioclip_linear_probe: ComparisonModel & { embedding_dimension: number; best_C: number };
    dinov3_linear_probe: ComparisonModel & { embedding_dimension: number; best_C: number };
    pairwise: {
      bioclip_probe_vs_zero_shot: PairwiseResult;
      bioclip_probe_vs_dinov3: PairwiseResult;
      zero_shot_vs_dinov3: PairwiseResult;
    };
  };
  organs: (ComparisonRow & { organ_category: string })[];
  species: (ComparisonRow & { species: string })[];
};

const outcomeCopy: Record<Outcome, { label: string; short: string; detail: string }> = {
  top1: { label: "Top‑1 直接命中", short: "Top‑1 对", detail: "第一名就是正确物种" },
  top5: { label: "Top‑5 才命中", short: "Top‑5 救回", detail: "第一名错误，但正确答案在第 2–5 名" },
  miss: { label: "Top‑5 未命中", short: "Top‑5 错", detail: "前五名都没有正确物种" },
};

const organNames: Record<string, string> = {
  leaf: "叶片",
  "whole plant": "整株 / 远景",
  "whole tree": "整株 / 远景",
  "whole tree (or vine)": "整株 / 藤本",
  inflorescence: "花序",
  fruit: "果实",
  twig: "枝条",
  bark: "树皮",
  cone: "球果",
  seed: "种子",
  other: "其他",
};

function pct(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function ConfidenceBar({ value, tone = "green" }: { value: number; tone?: "green" | "coral" | "lime" }) {
  return (
    <div className="mini-track" aria-label={`${pct(value)} confidence`}>
      <span className={`mini-fill ${tone}`} style={{ width: `${Math.max(value * 100, 1)}%` }} />
    </div>
  );
}

function SectionHeading({ kicker, title, note }: { kicker: string; title: string; note?: string }) {
  return (
    <div className="section-heading">
      <div>
        <p className="kicker">{kicker}</p>
        <h2>{title}</h2>
      </div>
      {note && <p className="section-note">{note}</p>}
    </div>
  );
}

function StrictComparisonSection({ comparison }: { comparison: StrictComparison }) {
  const { summary, organs, species } = comparison;
  const zero = summary.bioclip_zero_shot;
  const probe = summary.bioclip_linear_probe;
  const dino = summary.dinov3_linear_probe;
  const probeVsZero = summary.pairwise.bioclip_probe_vs_zero_shot;
  const probeVsDino = summary.pairwise.bioclip_probe_vs_dinov3;
  const metrics = [
    { label: "Top‑1", zero: zero.top1_accuracy, probe: probe.top1_accuracy, dino: dino.top1_accuracy },
    { label: "Top‑5", zero: zero.top5_accuracy, probe: probe.top5_accuracy, dino: dino.top5_accuracy },
    { label: "Macro Top‑1", zero: zero.macro_top1_accuracy, probe: probe.macro_top1_accuracy, dino: dino.macro_top1_accuracy },
  ];
  const strongestDino = [...species]
    .filter((item) => item.dinov3_probe_top1_accuracy > item.bioclip_probe_top1_accuracy)
    .sort((a, b) => a.top1_delta_bioclip_probe_minus_dinov3 - b.top1_delta_bioclip_probe_minus_dinov3)
    .slice(0, 3);

  return (
    <section className="section shell strict-comparison" id="comparison">
      <SectionHeading
        kicker="01 / STRICT HEAD-TO-HEAD"
        title="同一批图片、同一组候选，才是真比较"
        note="三条 baseline 使用完全相同的 211 张 test images 和 63 个 species。两个 linear probe 共享相同 train / validation、超参数网格与选择规则。"
      />

      <div className="comparison-metrics panel">
        <div className="comparison-head comparison-grid-row">
          <span>指标</span><strong>BioCLIP zero-shot</strong><strong>BioCLIP + probe</strong><strong>DINOv3 + probe</strong>
        </div>
        {metrics.map((metric) => (
          <div className="comparison-grid-row metric-row" key={metric.label}>
            <strong>{metric.label}</strong>
            <div className="model-score bio"><span style={{ width: `${metric.zero * 100}%` }} /><b>{pct(metric.zero)}</b></div>
            <div className="model-score probe"><span style={{ width: `${metric.probe * 100}%` }} /><b>{pct(metric.probe)}</b></div>
            <div className="model-score dino"><span style={{ width: `${metric.dino * 100}%` }} /><b>{pct(metric.dino)}</b></div>
          </div>
        ))}
      </div>

      <div className="paired-grid">
        <article className="paired-card dark-card">
          <p className="card-label">相同 classifier · 不同 representation</p>
          <h3>BioCLIP probe 比 DINOv3 高 30.8 pp</h3>
          <div className="paired-stack" aria-label="Paired prediction outcomes">
            <span className="both" style={{ width: `${probeVsDino.both_correct / 2.11}%` }} />
            <span className="bio-only" style={{ width: `${probeVsDino.first_only / 2.11}%` }} />
            <span className="dino-only" style={{ width: `${probeVsDino.second_only / 2.11}%` }} />
            <span className="neither" style={{ width: `${probeVsDino.both_wrong / 2.11}%` }} />
          </div>
          <div className="paired-counts">
            <span><i className="pair-dot both" />共同正确 <b>{probeVsDino.both_correct}</b></span>
            <span><i className="pair-dot bio-only" />仅 BioCLIP probe <b>{probeVsDino.first_only}</b></span>
            <span><i className="pair-dot dino-only" />仅 DINOv3 <b>{probeVsDino.second_only}</b></span>
            <span><i className="pair-dot neither" />共同错误 <b>{probeVsDino.both_wrong}</b></span>
          </div>
          <p className="stat-note">Exact McNemar p={probeVsDino.exact_mcnemar_p.toExponential(2)}；Top‑1 差值的 paired bootstrap 95% CI 为 {pct(probeVsDino.top1_paired_bootstrap_95_ci[0])}–{pct(probeVsDino.top1_paired_bootstrap_95_ci[1])}。</p>
        </article>

        <article className="paired-card protocol-card light-paired">
          <p className="card-label">相同 BioCLIP backbone · labels 的增益</p>
          <h3>Linear probe 只提高 1.4 pp</h3>
          <div className="paired-stack" aria-label="BioCLIP probe versus zero-shot outcomes">
            <span className="both" style={{ width: `${probeVsZero.both_correct / 2.11}%` }} />
            <span className="bio-only" style={{ width: `${probeVsZero.first_only / 2.11}%` }} />
            <span className="dino-only" style={{ width: `${probeVsZero.second_only / 2.11}%` }} />
            <span className="neither" style={{ width: `${probeVsZero.both_wrong / 2.11}%` }} />
          </div>
          <div className="paired-counts">
            <span><i className="pair-dot both" />共同正确 <b>{probeVsZero.both_correct}</b></span>
            <span><i className="pair-dot bio-only" />仅 probe <b>{probeVsZero.first_only}</b></span>
            <span><i className="pair-dot dino-only" />仅 zero-shot <b>{probeVsZero.second_only}</b></span>
            <span><i className="pair-dot neither" />共同错误 <b>{probeVsZero.both_wrong}</b></span>
          </div>
          <p className="light-stat">差值 95% CI 为 {pct(probeVsZero.top1_paired_bootstrap_95_ci[0])}–{pct(probeVsZero.top1_paired_bootstrap_95_ci[1])}，McNemar p={probeVsZero.exact_mcnemar_p.toFixed(2)}。这个小增益在当前 211 张 test 上不显著。</p>
        </article>
      </div>

      <div className="protocol-summary panel">
        <div><strong>BioCLIP zero-shot</strong><span>不使用 BioImages labels；image ↔ species text similarity。</span></div>
        <div><strong>BioCLIP + probe</strong><span>1024d frozen image embedding；同一 logistic regression，C={probe.best_C}。</span></div>
        <div><strong>DINOv3 + probe</strong><span>768d frozen CLS embedding；同一 logistic regression，C={dino.best_C}。</span></div>
      </div>

      <div className="strict-organ panel">
        <div className="strict-organ-head"><div><h3>按器官比较 Top‑1</h3><p>同一 test subset；小样本器官需谨慎解读。</p></div><span>Zero-shot / BioCLIP probe / DINOv3</span></div>
        {organs.map((item) => (
          <div className="strict-organ-row" key={item.organ_category}>
            <div><strong>{organNames[item.organ_category] ?? item.organ_category}</strong><small>n={item.total}</small></div>
            <div className="dual-track"><span className="bio" style={{ width: `${item.bioclip_zero_top1_accuracy * 100}%` }} /><b>{pct(item.bioclip_zero_top1_accuracy)}</b></div>
            <div className="dual-track"><span className="probe" style={{ width: `${item.bioclip_probe_top1_accuracy * 100}%` }} /><b>{pct(item.bioclip_probe_top1_accuracy)}</b></div>
            <div className="dual-track"><span className="dino" style={{ width: `${item.dinov3_probe_top1_accuracy * 100}%` }} /><b>{pct(item.dinov3_probe_top1_accuracy)}</b></div>
            <strong className="delta-positive">+{(item.top1_delta_bioclip_probe_minus_dinov3 * 100).toFixed(1)} pp</strong>
          </div>
        ))}
      </div>

      <div className="comparison-footer">
        <p>DINOv3 仍有 {probeVsDino.second_only} 张相对 BioCLIP probe 独占正确{strongestDino.length ? `，优势物种包括 ${strongestDino.map((item) => item.species).join("、")}` : ""}。但主要结论很稳定：差距来自 representation，而不是 classifier。</p>
        <div><a href="/downloads/strict_paired_predictions.csv" download>三模型逐图 CSV ↘</a><a href="/downloads/strict_comparison_summary.json" download>结果摘要 JSON ↘</a></div>
      </div>
    </section>
  );
}

function SpeciesExplorer({ species }: { species: SpeciesResult[] }) {
  const [mode, setMode] = useState<"weak" | "strong">("weak");
  const rows = useMemo(() => {
    const sorted = [...species].sort((a, b) =>
      mode === "weak"
        ? a.top1_accuracy - b.top1_accuracy || b.total - a.total
        : b.top1_accuracy - a.top1_accuracy || b.total - a.total,
    );
    return sorted.slice(0, 12);
  }, [species, mode]);

  return (
    <div className="panel species-panel">
      <div className="panel-topline">
        <div>
          <h3>物种难度排行</h3>
          <p>每条都同时显示样本量，避免被小样本的 0% 或 100% 误导。</p>
        </div>
        <div className="segmented" aria-label="排序方式">
          <button className={mode === "weak" ? "active" : ""} onClick={() => setMode("weak")}>最难识别</button>
          <button className={mode === "strong" ? "active" : ""} onClick={() => setMode("strong")}>最易识别</button>
        </div>
      </div>
      <div className="rank-list">
        {rows.map((item, index) => (
          <div className="rank-row" key={item.name}>
            <span className="rank-index">{String(index + 1).padStart(2, "0")}</span>
            <div className="rank-name">
              <strong><i>{item.name}</i></strong>
              <span>{item.vernacular} · n={item.total}</span>
            </div>
            <div className="rank-bar">
              <span style={{ width: `${item.top1_accuracy * 100}%` }} />
            </div>
            <strong className="rank-value">{pct(item.top1_accuracy)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function ImageGallery({ images, species }: { images: ImageResult[]; species: SpeciesResult[] }) {
  const [outcome, setOutcome] = useState<Outcome>("top1");
  const [speciesFilter, setSpeciesFilter] = useState("all");
  const [organFilter, setOrganFilter] = useState("all");
  const [page, setPage] = useState(0);

  const organs = useMemo(() => Array.from(new Set(images.map((item) => item.organ))).sort(), [images]);
  const filtered = useMemo(() => {
    const subset = images.filter(
      (item) =>
        item.outcome === outcome &&
        (speciesFilter === "all" || item.truth === speciesFilter) &&
        (organFilter === "all" || item.organ === organFilter),
    );
    return subset.sort((a, b) => {
      if (outcome === "top1") return b.confidence - a.confidence;
      if (outcome === "top5") return (a.rank ?? 6) - (b.rank ?? 6) || b.confidence - a.confidence;
      return b.confidence - a.confidence;
    });
  }, [images, outcome, speciesFilter, organFilter]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / 9));
  const shown = filtered.slice((page % pageCount) * 9, (page % pageCount) * 9 + 9);

  function choose(next: Outcome) {
    setOutcome(next);
    setPage(0);
  }

  return (
    <div className="gallery-shell">
      <div className="gallery-controls">
        <div className="outcome-tabs">
          {(Object.keys(outcomeCopy) as Outcome[]).map((key) => (
            <button key={key} className={outcome === key ? `active ${key}` : ""} onClick={() => choose(key)}>
              {outcomeCopy[key].short}
              <span>{images.filter((item) => item.outcome === key).length}</span>
            </button>
          ))}
        </div>
        <div className="filters">
          <label>
            <span>真实物种</span>
            <select value={speciesFilter} onChange={(event) => { setSpeciesFilter(event.target.value); setPage(0); }}>
              <option value="all">全部物种</option>
              {species.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
          </label>
          <label>
            <span>器官</span>
            <select value={organFilter} onChange={(event) => { setOrganFilter(event.target.value); setPage(0); }}>
              <option value="all">全部器官</option>
              {organs.map((item) => <option key={item} value={item}>{organNames[item] ?? item}</option>)}
            </select>
          </label>
        </div>
      </div>

      <div className="gallery-status">
        <p><strong>{filtered.length}</strong> 张符合筛选 · {outcomeCopy[outcome].detail}</p>
        {filtered.length > 9 && (
          <div className="pager">
            <button aria-label="上一页" onClick={() => setPage((page - 1 + pageCount) % pageCount)}>←</button>
            <span>{(page % pageCount) + 1} / {pageCount}</span>
            <button aria-label="下一页" onClick={() => setPage((page + 1) % pageCount)}>→</button>
          </div>
        )}
      </div>

      {shown.length ? (
        <div className="image-grid">
          {shown.map((item) => (
            <article className="image-card" key={item.id}>
              <a href={item.image} target="_blank" rel="noreferrer" className="image-wrap" aria-label={`查看 ${item.truth} 原图`}>
                <img src={item.thumbnail} alt={`${item.truth}, ${organNames[item.organ] ?? item.organ}`} loading="lazy" />
                <span className={`result-pill ${item.outcome}`}>{outcomeCopy[item.outcome].short}</span>
                <span className="organ-pill">{organNames[item.organ] ?? item.organ}</span>
              </a>
              <div className="image-copy">
                <p className="image-id">{item.id}</p>
                <h3><i>{item.truth}</i></h3>
                <p className="common-name">{item.vernacular || item.family}</p>
                <div className="prediction-line">
                  <span>模型首选</span>
                  <strong className={item.outcome === "top1" ? "correct" : "wrong"}><i>{item.predicted}</i></strong>
                </div>
                <div className="confidence-line">
                  <ConfidenceBar value={item.confidence} tone={item.outcome === "top1" ? "green" : "coral"} />
                  <strong>{pct(item.confidence)}</strong>
                </div>
                <details>
                  <summary>展开 Top‑5</summary>
                  <ol>
                    {item.top5.map((prediction, index) => (
                      <li key={prediction.species} className={prediction.species === item.truth ? "is-truth" : ""}>
                        <span>{index + 1}. <i>{prediction.species}</i></span>
                        <strong>{pct(prediction.confidence)}</strong>
                      </li>
                    ))}
                  </ol>
                </details>
                <p className="credit">Photo: {item.creator || "BioImages contributor"}</p>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="empty-state">这个筛选组合没有图片。</div>
      )}
    </div>
  );
}

export default function Dashboard({ data, comparison }: { data: DashboardData; comparison: StrictComparison }) {
  const maxError = Math.max(...data.errors.map((item) => item.count));

  return (
    <main>
      <nav className="nav shell">
        <a href="#top" className="brand"><span>BI</span> BioImages Benchmark</a>
        <div className="nav-links">
          <a href="#comparison">严格对比</a>
          <a href="#analysis">BioCLIP 深入分析</a>
          <a href="#examples">图片样例</a>
          <a className="download-link" href="/downloads/strict_paired_predictions.csv" download>下载对比 CSV ↘</a>
        </div>
      </nav>

      <header className="hero shell" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span /> STRICT TEST · THREE FORMAL BASELINES</p>
          <h1>同一批 test images<br /><em>三条 baseline</em></h1>
          <p className="hero-lede">
            211 张严格 test images，63 个 species。现在同时比较 BioCLIP zero-shot、BioCLIP frozen + linear probe，以及 DINOv3 frozen + linear probe。
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#comparison">查看公平对比 <span>↓</span></a>
            <a className="secondary-action" href="#examples">查看真实图片</a>
          </div>
        </div>

        <div className="score-card" aria-label="Benchmark headline metrics">
          <div className="score-card-top">
            <span>STRICT TOP‑1</span>
            <span className="status-dot">运行成功</span>
          </div>
          <div className="headline-score three-way">
            <div>
              <span>BioCLIP zero</span>
              <strong>{pct(comparison.summary.bioclip_zero_shot.top1_accuracy)}</strong>
              <small>{comparison.summary.bioclip_zero_shot.top1_correct} / 211</small>
            </div>
            <div>
              <span>BioCLIP probe</span>
              <strong>{pct(comparison.summary.bioclip_linear_probe.top1_accuracy)}</strong>
              <small>{comparison.summary.bioclip_linear_probe.top1_correct} / 211</small>
            </div>
            <div>
              <span>DINOv3 + probe</span>
              <strong>{pct(comparison.summary.dinov3_linear_probe.top1_accuracy)}</strong>
              <small>{comparison.summary.dinov3_linear_probe.top1_correct} / 211</small>
            </div>
          </div>
          <div className="outcome-stack" aria-label="逐图配对结果分布">
            <span className="stack-top1" style={{ width: `${comparison.summary.pairwise.bioclip_probe_vs_dinov3.both_correct / 2.11}%` }} />
            <span className="stack-top5" style={{ width: `${comparison.summary.pairwise.bioclip_probe_vs_dinov3.first_only / 2.11}%` }} />
            <span className="stack-dino" style={{ width: `${comparison.summary.pairwise.bioclip_probe_vs_dinov3.second_only / 2.11}%` }} />
            <span className="stack-miss" style={{ width: `${comparison.summary.pairwise.bioclip_probe_vs_dinov3.both_wrong / 2.11}%` }} />
          </div>
          <div className="run-facts">
            <span><strong>211</strong> test images</span>
            <span><strong>63</strong> same species</span>
            <span><strong>0</strong> individual overlap</span>
          </div>
        </div>
      </header>

      <section className="insight-strip">
        <div className="shell insights">
          <article><span>01</span><p>最高 Top‑1</p><strong>BioCLIP probe · 87.2%</strong><small>zero-shot 为 85.8%</small></article>
          <article><span>02</span><p>Labels 的净增益</p><strong>+1.4 pp</strong><small>95% CI −3.3–6.2 pp · p=0.69</small></article>
          <article><span>03</span><p>Representation 差距</p><strong>+30.8 pp</strong><small>同一 linear classifier</small></article>
          <article><span>04</span><p>Macro Top‑1</p><strong>86.7 / 85.4 / 51.2</strong><small>probe / zero / DINOv3</small></article>
        </div>
      </section>

      <StrictComparisonSection comparison={comparison} />

      <section className="section shell" id="process">
        <SectionHeading
          kicker="02 / BIOCLIP FULL-RUN PIPELINE"
          title="BioCLIP 的预测过程，可以拆到这里"
          note="它不是对每个物种逐一“思考”，而是在同一个向量空间里一次性比较相似度。"
        />
        <div className="process-flow">
          {[
            ["01", "Original image", "读取原图并缩放、裁剪为模型输入"],
            ["02", "Image encoder", "Vision Transformer 将画面压缩成一个视觉向量"],
            ["03", "85 text vectors", "科学名与英文俗名预先编码，并按物种平均"],
            ["04", "Cosine similarity", "视觉向量同时与 85 个物种向量计算相似度"],
            ["05", "Softmax", "把 85 个分数归一化为封闭候选集内的概率"],
            ["06", "Top‑5", "按分数排序，输出第一名与前五名"],
          ].map(([number, title, description], index) => (
            <article className="process-step" key={number}>
              <span>{number}</span>
              <h3>{title}</h3>
              <p>{description}</p>
              {index < 5 && <i aria-hidden="true">→</i>}
            </article>
          ))}
        </div>
        <div className="interpretability-note">
          <div className="note-mark">!</div>
          <div>
            <h3>没有可读的“中间推理过程”</h3>
            <p>BioCLIP 能给出 embedding、85 个相似度和排序，但不会说明“因为叶缘锯齿、叶脉形状所以是某物种”。若以后需要定位模型在看哪里，要另加 Grad‑CAM / attention rollout；它们是可视化解释工具，不是模型本身的语言推理。</p>
          </div>
          <div className="vector-chip"><span>visual embedding</span><code>[ 0.08, −0.14, … ]</code></div>
        </div>
      </section>

      <section className="section tinted" id="analysis">
        <div className="shell">
          <SectionHeading
          kicker="03 / BIOCLIP FULL-RUN ANALYSIS"
            title="器官不同，难度差很多"
            note="果实和叶片最好；树皮与整株远景最难。Top‑5 说明模型是否至少把正确物种放进候选清单。"
          />
          <div className="organ-table panel">
            <div className="organ-head"><span>器官</span><span>样本量</span><span>Top‑1</span><span>Top‑5</span></div>
            {data.organs.map((item) => (
              <div className="organ-row" key={item.name}>
                <strong>{organNames[item.name] ?? item.name}</strong>
                <span className="organ-count">{item.total}</span>
                <div className="accuracy-cell"><div className="table-track"><span style={{ width: `${item.top1_accuracy * 100}%` }} /></div><strong>{pct(item.top1_accuracy)}</strong></div>
                <div className="accuracy-cell"><div className="table-track light"><span style={{ width: `${item.top5_accuracy * 100}%` }} /></div><strong>{pct(item.top5_accuracy)}</strong></div>
              </div>
            ))}
          </div>

          <div className="analysis-grid">
            <SpeciesExplorer species={data.species} />
            <div className="panel confusion-panel">
              <div className="panel-topline"><div><h3>最常见的错误方向</h3><p>箭头左边是真实物种，右边是模型首选。</p></div></div>
              <div className="confusion-list">
                {data.errors.slice(0, 10).map((item, index) => (
                  <div className="confusion-row" key={`${item.truth}-${item.predicted}`}>
                    <span className="confusion-index">{index + 1}</span>
                    <div className="confusion-names"><i>{item.truth}</i><b>→</b><i>{item.predicted}</i></div>
                    <div className="confusion-bar"><span style={{ width: `${(item.count / maxError) * 100}%` }} /></div>
                    <strong>{item.count}</strong>
                  </div>
                ))}
              </div>
              <div className="confusion-callout"><strong>模式很清楚：</strong>很多错误发生在同属近缘种之间，例如红栎 → 针栎、美国白蜡 → 绿白蜡。这比“随机猜错”更有信息量。</div>
            </div>
          </div>
        </div>
      </section>

      <section className="section shell">
        <SectionHeading
          kicker="04 / BIOCLIP CONFIDENCE CHECK"
          title="置信度越高通常越准，但它没有校准"
          note="每组把模型平均置信度与实际 Top‑1 准确率并排。两者差距就是过度或不足自信。"
        />
        <div className="confidence-panel panel">
          <div className="confidence-legend"><span><i className="legend-line model" />平均模型置信度</span><span><i className="legend-line actual" />实际准确率</span></div>
          <div className="confidence-grid">
            {data.confidence.map((item) => (
              <div className="confidence-column" key={item.label}>
                <div className="confidence-bars">
                  <span className="confidence-model" style={{ height: `${item.average_confidence * 100}%` }}><i>{pct(item.average_confidence, 0)}</i></span>
                  <span className="confidence-actual" style={{ height: `${item.accuracy * 100}%` }}><i>{pct(item.accuracy, 0)}</i></span>
                </div>
                <strong>{item.label}</strong>
                <small>n={item.total}</small>
              </div>
            ))}
          </div>
          <div className="confidence-warning"><span>结论</span><p>90–100% 置信区间有 1,406 张图，但实际准确率是 91.0%。因此页面中的 confidence 只表示“在这 85 个候选中模型有多偏向第一名”，不能当成开放世界下的真实正确概率。</p></div>
        </div>
      </section>

      <section className="section examples-section" id="examples">
        <div className="shell">
          <SectionHeading
            kicker="05 / BIOCLIP IMAGE-LEVEL EVIDENCE"
            title="直接看每一张图，模型到底选了什么"
            note="图片仍由 BioImages 云端托管；点击缩略图可查看原始展示图。可按结果、物种和器官筛选。"
          />
          <ImageGallery images={data.images} species={data.species} />
        </div>
      </section>

      <section className="section shell methodology">
        <SectionHeading kicker="06 / BIOCLIP FULL-RUN METHODOLOGY" title="完整 BioCLIP baseline 为什么可复现" />
        <div className="method-grid">
          <article><span>01</span><h3>没有训练</h3><p>使用预训练 BioCLIP 2.5，BioImages labels 没有参与模型训练或微调。</p></article>
          <article><span>02</span><h3>Ground truth 隐藏</h3><p>推理清单只有 image_id、文件名和路径。1,899 张预测完成后才重新读取 corpus.json 计算准确率。</p></article>
          <article><span>03</span><h3>同一候选集合</h3><p>每张图都与完全相同的 85 个物种名称比较，没有使用该图对应的真实类别做提示。</p></article>
          <article><span>04</span><h3>运行环境</h3><p>NVIDIA L4、BF16、batch 32。纯 GPU 推理 21.9 秒，完整流程 98.8 秒。</p></article>
        </div>
        <div className="method-footer">
          <div><span>MODEL</span><code>imageomics/bioclip-2.5-vith14</code></div>
          <div><span>METHOD</span><code>zero-shot closed-set cosine similarity</code></div>
          <a href="/downloads/bioclip25_predictions.csv" download>下载逐图结果 CSV ↘</a>
        </div>
      </section>

      <footer>
        <div className="shell footer-inner">
          <div><strong>BioImages Benchmark</strong><p>BioCLIP zero-shot、BioCLIP linear probe 与 DINOv3 linear probe 的严格、逐图配对比较。</p></div>
          <div><span>211 strict test images</span><span>63 species</span><span>2026 benchmark</span></div>
        </div>
      </footer>
    </main>
  );
}
