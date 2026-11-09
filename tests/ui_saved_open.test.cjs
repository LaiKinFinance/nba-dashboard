// Opening the app: answer at once from the phone's saved copy, refresh in the background, redraw once with fresh data.
const test = require('node:test'); const assert = require('node:assert/strict');
const fs = require('node:fs'); const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8');
const src = html.slice(html.indexOf('const SAVE_OK='), html.indexOf('function api(p,o={})'));
function load(store) {
  const ls = { getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v } };
  const env = { renders: 0 };
  const f = new Function('localStorage', '$', 'document', 'memo', 'env',
    `let lastLoad=1;const render=async()=>{env.renders++};${src};return {savedFirst,setBoot:v=>{booting=v},get quiet(){return quietRender}}`);
  return [f(ls, () => null, { createElement: () => ({ style: {} }), body: { appendChild() {} }, activeElement: null }, {}, env), env];
}
const tick = () => new Promise(r => setTimeout(r, 5));
test('with a saved copy: returns it at once, then saves fresh data and redraws once', async () => {
  const store = { 'nbaSaved:best': JSON.stringify({ t: Date.now(), v: ['old'] }) };
  const [m, env] = load(store);
  let release; const fresh = new Promise(r => { release = r });
  assert.deepEqual(await m.savedFirst('best', () => fresh), ['old']);
  assert.equal(env.renders, 0);
  release(['new']); await tick();
  assert.deepEqual(JSON.parse(store['nbaSaved:best']).v, ['new']);
  assert.equal(env.renders, 1);
});
test('no saved copy, or after opening: waits for fresh data, no extra redraw', async () => {
  const store = {}; const [m, env] = load(store);
  assert.deepEqual(await m.savedFirst('best', async () => ['new']), ['new']);
  m.setBoot(false); store['nbaSaved:value'] = JSON.stringify({ t: Date.now(), v: ['old'] });
  assert.deepEqual(await m.savedFirst('value', async () => ['fresh']), ['fresh']);
  await tick(); assert.equal(env.renders, 0);
});
test('old saved copies (over 18 hours) and unlisted pages are never used', async () => {
  const store = { 'nbaSaved:best': JSON.stringify({ t: Date.now() - 19 * 3600e3, v: ['stale'] }), 'nbaSaved:picks': JSON.stringify({ t: Date.now(), v: ['x'] }) };
  const [m] = load(store);
  assert.deepEqual(await m.savedFirst('best', async () => ['new']), ['new']);
  assert.deepEqual(await m.savedFirst('picks', async () => ['mine']), ['mine']);
});
test('the one redraw never reuses saved copies, so it cannot repeat itself', async () => {
  const store = { 'nbaSaved:live': JSON.stringify({ t: Date.now(), v: ['old'] }) };
  const [m, env] = load(store);
  await m.savedFirst('live', async () => ['new']); await tick();
  assert.equal(env.renders, 1);
  assert.deepEqual(await m.savedFirst('live', async () => ['newer']), ['newer']);   // the redraw's own requests wait for fresh data
  await tick(); assert.equal(env.renders, 1);
});
