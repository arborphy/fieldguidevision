import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../assets/review.js', import.meta.url), 'utf8');
async function openReview({ search = '', rows = [], local = {}, error = null } = {}) {
  const elements = new Map();
  const storage = new Map([['bioimages-scale-review-v1', JSON.stringify(local)]]);
  const ranges = [];
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, {
      style: {}, classList: { toggle() {} },
      elements: { confidence: {}, note: {} },
      querySelectorAll: () => [], addEventListener() {}, scrollIntoView() {},
    });
    return elements.get(selector);
  };
  const query = {
    select() { return this; }, eq() { return this; }, order() { return this; },
    range(from, to) { ranges.push([from, to]); return Promise.resolve({ data: rows.slice(from, to + 1), error }); },
  };
  const data = {
    batch_id: 'batch', labels: ['close-up'], examples: {}, definitions: {},
    items: ['a', 'b', 'c'].map((image_id, index) => ({ image_id, index,
      species: 'Plant', provisional_scale: 'close-up', evidence: {} })),
  };
  const context = {
    window: { BIOIMAGES_SCALE_REVIEW: data, BIOIMAGES_RUNTIME: { supabaseUrl: 'url', supabaseAnonKey: 'key' },
      supabase: { createClient: () => ({ auth: { getSession: async () => ({ data: { session: {} } }) }, from: () => query }) },
      addEventListener() {},
    },
    document: { querySelector: element },
    location: { search, href: 'https://example.test/scale-review' + search },
    history: { replaceState() {} }, URL, URLSearchParams,
    localStorage: { getItem: (k) => storage.get(k), setItem: (k, v) => storage.set(k, v) },
  };
  vm.runInNewContext(source, context);
  for (let i = 0; i < 5; i++) await new Promise(setImmediate);
  return { elements, ranges, records: JSON.parse(storage.get('bioimages-scale-review-v1')) };
}

test('a new browser resumes after annotations saved by other reviewers', async () => {
  const view = await openReview({ rows: [{ image_id: 'a', human_scale: 'close-up', confidence: 'high' }] });
  assert.equal(view.elements.get('#progress-count').textContent, '1 / 3');
  assert.equal(view.elements.get('#review-position').textContent, 'Image 2 of 3');
});
test('an explicit image link remains selected', async () => {
  const view = await openReview({ search: '?n=3', rows: [{ image_id: 'a', human_scale: 'close-up' }] });
  assert.equal(view.elements.get('#review-position').textContent, 'Image 3 of 3');
});
test('fractional image numbers do not produce an invalid index', async () => {
  const view = await openReview({ search: '?n=1.5' });
  assert.equal(view.elements.get('#review-position').textContent, 'Image 1 of 3');
});
test('the latest shared revision is displayed and pages beyond 1000 are loaded', async () => {
  const rows = Array.from({ length: 1000 }, () => ({ image_id: 'a', human_scale: 'distant' }));
  rows.push({ image_id: 'a', human_scale: 'close-up' }, { image_id: 'b', human_scale: 'mid-range' });
  const view = await openReview({ rows });
  assert.deepEqual(view.ranges, [[0, 999], [1000, 1999]]);
  assert.equal(view.records.a.human_scale, 'close-up');
  assert.equal(view.elements.get('#progress-count').textContent, '2 / 3');
  assert.equal(view.elements.get('#review-position').textContent, 'Image 3 of 3');
});
test('database errors retain the local backup and report unavailable shared access', async () => {
  const view = await openReview({ local: { b: { human_scale: 'close-up' } }, error: { message: 'permission denied' } });
  assert.equal(view.records.b.human_scale, 'close-up');
  assert.match(view.elements.get('#review-status').innerHTML, /permission denied/);
});
