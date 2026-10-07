import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const output = join(root, "dist");
const required = ["SUPABASE_URL", "SUPABASE_ANON_KEY"];
const missing = required.filter((name) => !process.env[name]);
if (missing.length) throw new Error(`Missing Vercel environment variables: ${missing.join(", ")}`);

rmSync(output, { recursive: true, force: true });
mkdirSync(output, { recursive: true });
for (const name of ["assets", "data", "analysis", "tagging"]) {
  cpSync(join(root, name), join(output, name), { recursive: true });
}
const scaleReviewSource = join(root, "..", "scale_conditioned_analysis", "review");
const scaleReviewOutput = join(output, "scale-review");
mkdirSync(scaleReviewOutput, { recursive: true });
for (const name of ["assets", "data"]) {
  cpSync(join(scaleReviewSource, name), join(scaleReviewOutput, name), { recursive: true });
}
cpSync(join(scaleReviewSource, "index.html"), join(scaleReviewOutput, "index.html"));
for (const name of ["index.html", "README.md", "VERCEL_HUMAN_REVIEW.md"]) {
  cpSync(join(root, name), join(output, name));
}

const config = {
  supabaseUrl: process.env.SUPABASE_URL,
  supabaseAnonKey: process.env.SUPABASE_ANON_KEY,
  reviewBatchId: process.env.REVIEW_BATCH_ID || "00000000-0000-4000-8000-000000000001",
  scaleReviewBatchId: process.env.SCALE_REVIEW_BATCH_ID || "00000000-0000-4000-8000-000000000002",
};
writeFileSync(join(output, "data", "runtime-config.js"), `window.BIOIMAGES_RUNTIME = ${JSON.stringify(config, null, 2)};\n`);
writeFileSync(join(output, "scale-review", "data", "runtime-config.js"), `window.BIOIMAGES_RUNTIME = ${JSON.stringify(config, null, 2)};\n`);

const scaleReviewHtmlPath = join(scaleReviewOutput, "index.html");
const scaleReviewHtml = readFileSync(scaleReviewHtmlPath, "utf8")
  .replace('href="assets/review.css?v=1"', 'href="/scale-review/assets/review.css?v=1"')
  .replace('src="data/runtime-config.js?v=1"', 'src="/scale-review/data/runtime-config.js?v=1"')
  .replace('src="data/scale-review-data.js?v=1"', 'src="/scale-review/data/scale-review-data.js?v=1"')
  .replace('src="assets/review.js?v=1"', 'src="/scale-review/assets/review.js?v=1"')
  .replace('href="../index.html"', 'href="/index.html"');
writeFileSync(scaleReviewHtmlPath, scaleReviewHtml);

const htmlPath = join(output, "index.html");
const html = readFileSync(htmlPath, "utf8").replace(
  'href="../bioimages_browser/"',
  'href="https://arborphy.github.io/fieldguidevision/botanical-computer-vision/bioimages_browser/"',
).replace('href="../scale_conditioned_analysis/review/"', 'href="scale-review/"');
writeFileSync(htmlPath, html);
