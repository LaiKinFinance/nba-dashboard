const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x=>x[1]);
const script = scripts.find(x=>x.includes('const legsPayload='));

test('the full page script parses after removing money controls', () => {
  for (const source of scripts) assert.doesNotThrow(() => new Function(source));
});

test('personal results render without a model, prices or money', async () => {
  const source = script.slice(script.indexOf('const legLine='), script.indexOf('async function res('));
  const response = {
    bets: [{id:1, user:'mark', created:'2026-10-29T00:00Z', legs:'[{"stat":"reb","line":7.5}]', result:'won', model_prob:.6, manual:0}],
    users: [{user:'mark', bets:1, wins:1, losses:0, settled:1, hit_rate:1}],
  };
  const render = new Function('api', 'localStorage', 'esc', 'pct', 'legDesc', source+';return bets')(
    async () => response, {user:'mark'}, x=>String(x), x=>`${Math.round(x*100)}%`, ()=> 'Over 7.5 rebounds');
  const app = {innerHTML:''}; await render(app);
  assert.match(app.innerHTML, /Past picks/);
  assert.match(app.innerHTML, /Over 7.5 rebounds/);
  assert.match(app.innerHTML, /Mark won/);
  assert.match(app.innerHTML, /Mark lost/);
  assert.doesNotMatch(app.innerHTML, /\$|profit|stake|refund|Sportsbet|bankroll/i);
});

test('saving a pick sends no financial fields and keeps its threshold type', async () => {
  const payloadLine = script.match(/^const legsPayload=[^\n]+/m)[0];
  const save = script.slice(script.indexOf('async function logBet()'), script.indexOf('/* ---------- live lineup/injury alerts ---------- */'));
  const original = [{game_id:'g',player_id:'p',stat:'pts',line:20,side:'over',kind:'atleast'}];
  const calls=[];
  const run = new Function('post','needName','go',`let legs=${JSON.stringify(original)},calc={r:{joint:.7}};`+payloadLine+save+';return logBet');
  await run(async (route,body)=>calls.push({route,body}), ()=>true, ()=>{})();
  assert.equal(calls[0].route, 'bets');
  assert.equal(calls[0].body.legs[0].kind, 'atleast');
  assert.equal(calls[0].body.model_prob, .7);
  for(const key of ['stake','odds','amount','returned_amount']) assert.equal(Object.hasOwn(calls[0].body,key), false);
});
