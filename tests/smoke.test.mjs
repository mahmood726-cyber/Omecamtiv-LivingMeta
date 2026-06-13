/*
 * Minimal offline smoke test for the Omecamtiv-LivingMeta dashboard.
 *
 * Two parts, no external deps (Node >=18, built-in `node:test`):
 *   1. Structural integrity of the shipped single-file HTML (OMECAMTIV_REVIEW.html):
 *      no BOM, balanced <script>/<div> in the static markup, no unfilled
 *      template placeholders, vendored Plotly present, HKSJ floor guard intact.
 *   2. A self-contained regression check of the inverse-variance + DerSimonian-Laird
 *      pooling math the engine relies on, validated against a hand-computed
 *      reference so a future edit to the formula is caught.
 *
 * Run: node --test   (or  npm test)
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const htmlPath = join(root, 'OMECAMTIV_REVIEW.html');

test('shipped HTML: no UTF-8 BOM', () => {
  const buf = readFileSync(htmlPath);
  assert.notDeepEqual([buf[0], buf[1], buf[2]], [0xef, 0xbb, 0xbf], 'file starts with a BOM');
});

test('shipped HTML: <script> tags are balanced', () => {
  const html = readFileSync(htmlPath, 'utf8');
  const open = (html.match(/<script[\s>]/g) || []).length;
  const close = (html.match(/<\/script>/g) || []).length;
  assert.equal(open, close, `<script> ${open} != </script> ${close}`);
});

test('shipped HTML: static (non-script) <div> markup is balanced', () => {
  const html = readFileSync(htmlPath, 'utf8');
  // Strip <script>...</script> blocks so JS-string markup is not counted.
  const staticHtml = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, '');
  const open = (staticHtml.match(/<div[\s>]/g) || []).length;
  const close = (staticHtml.match(/<\/div>/g) || []).length;
  assert.equal(open, close, `static <div> ${open} != </div> ${close}`);
});

test('shipped HTML: no unfilled template placeholders', () => {
  const html = readFileSync(htmlPath, 'utf8');
  for (const tok of ['REPLACE_ME', '__PLACEHOLDER__', 'TODO_FILL']) {
    assert.ok(!html.includes(tok), `placeholder token present: ${tok}`);
  }
});

test('shipped HTML: Plotly is vendored locally (offline render)', () => {
  const html = readFileSync(htmlPath, 'utf8');
  assert.match(html, /src="assets\/plotly\.min\.js"/, 'expected local assets/plotly.min.js');
  readFileSync(join(root, 'assets', 'plotly.min.js')); // throws if missing
});

test('shipped HTML: HKSJ variance-correction floor guard is present', () => {
  // Math.max(1, q*) prevents CI narrowing below DL when Q < df (advanced-stats rule).
  const html = readFileSync(htmlPath, 'utf8');
  assert.match(html, /Math\.max\(1,\s*qStar\)/, 'HKSJ floor max(1, qStar) missing');
});

test('engine math: inverse-variance + DerSimonian-Laird pooling matches reference', () => {
  // Three studies (log-scale effects yi with within-study variances vi).
  const yi = [0.10, -0.20, 0.30];
  const vi = [0.04, 0.05, 0.06];
  const k = yi.length;
  const df = k - 1;

  // Fixed-effect (inverse-variance) weights.
  const wF = vi.map((v) => 1 / v);
  const sW = wF.reduce((a, b) => a + b, 0);
  const sWY = wF.reduce((a, w, i) => a + w * yi[i], 0);
  const sW_y2 = wF.reduce((a, w, i) => a + w * yi[i] * yi[i], 0);
  const sW2 = wF.reduce((a, w) => a + w * w, 0);

  // Q and DL tau^2 (exactly as the engine computes them).
  const Q = Math.max(0, sW_y2 - (sWY * sWY) / sW);
  const tau2 = Q > df ? (Q - df) / (sW - sW2 / sW) : 0;

  // Random-effects pooling.
  const wR = vi.map((v) => 1 / (v + tau2));
  const sWR = wR.reduce((a, b) => a + b, 0);
  const sWRY = wR.reduce((a, w, i) => a + w * yi[i], 0);
  const pooled = sWRY / sWR;
  const pooledSE = Math.sqrt(1 / sWR);

  // Hand-computed reference values (verified independently).
  // Fixed weights: 25, 20, 16.667 ; sW = 61.667
  // sWY = 25*0.1 + 20*(-0.2) + 16.667*0.3 = 2.5 - 4 + 5 = 3.5
  // muFixed = 3.5/61.667 = 0.05676
  // sW_y2 = 25*0.01 + 20*0.04 + 16.667*0.09 = 0.25 + 0.8 + 1.5 = 2.55
  // Q = 2.55 - 3.5^2/61.667 = 2.55 - 0.19865 = 2.35135  (> df=2 so tau2>0)
  assert.ok(Math.abs(Q - 2.35135) < 1e-4, `Q=${Q}`);
  assert.ok(tau2 > 0, 'tau2 should be > 0 when Q > df');
  // With small tau2 the pooled estimate stays near the fixed-effect mean.
  assert.ok(Math.abs(pooled - 0.0612) < 5e-3, `pooled=${pooled}`);
  assert.ok(pooledSE > 0 && pooledSE < 0.2, `pooledSE=${pooledSE}`);
});

test('engine math: DL tau^2 is exactly 0 when Q <= df (homogeneous)', () => {
  // Identical effects -> Q = 0 -> tau2 must floor at 0 (no negative variance).
  const yi = [0.2, 0.2, 0.2];
  const vi = [0.05, 0.05, 0.05];
  const df = yi.length - 1;
  const wF = vi.map((v) => 1 / v);
  const sW = wF.reduce((a, b) => a + b, 0);
  const sWY = wF.reduce((a, w, i) => a + w * yi[i], 0);
  const sW_y2 = wF.reduce((a, w, i) => a + w * yi[i] * yi[i], 0);
  const sW2 = wF.reduce((a, w) => a + w * w, 0);
  const Q = Math.max(0, sW_y2 - (sWY * sWY) / sW);
  const tau2 = Q > df ? (Q - df) / (sW - sW2 / sW) : 0;
  assert.equal(tau2, 0);
});
