const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8');
const source = html.split('// BEGIN playerLegProgress:')[1].split('\n').slice(1).join('\n').split('// END playerLegProgress')[0];
const context = {
  statName: {pts:'PTS', reb:'REB', ast:'AST', pra:'Pts+Reb+Ast', pr:'Pts+Reb', pa:'Pts+Ast'},
};
vm.createContext(context);
vm.runInContext(html.match(/function catCount\(r\)\{[^\n]*\}/)[0], context);          // the real five-category function from the page
vm.runInContext(source, context);
const calc = context.playerLegProgress;
const box = {pts:12, reb:7, ast:5};

test('combined markets sum the box components', () => {
  for (const [stat, total] of [['pra',24], ['pr',19], ['pa',17]]) {
    const result=calc({stat, line:total-.5, side:'over'}, box);
    assert.equal(result.cur,total);
    assert.equal(result.hit,true);
    assert.equal(result.percent,100);
  }
});
test('under and over show opposite results at a half-point line', () => {
  assert.equal(calc({stat:'pra',line:24.5,side:'under'},box).hit,true);
  const over=calc({stat:'pra',line:24.5,side:'over'},box);
  assert.equal(over.hit,false);
  assert.match(over.txt,/needs 1 more/);
  assert.equal(calc({stat:'pra',line:23.5,side:'under'},box).hit,false);
});
test('missing statistics do not become NaN progress or a false result', () => {
  assert.equal(calc({stat:'pra',line:20.5,side:'over'},{pts:12,reb:7}),null);
  assert.equal(calc({stat:'pts',line:'invalid',side:'over'},box),null);
});
test('binary no legs respect the chosen side', () => {
  const double={pts:12,reb:10,ast:5};
  assert.equal(calc({stat:'dd',side:'under'},double).hit,false);
  assert.equal(calc({stat:'td',side:'under'},double).hit,true);
});

test('double and triple doubles count steals and blocks, all five categories', () => {
  const stocks={pts:12,reb:4,ast:3,stl:10,blk:11};
  assert.equal(calc({stat:'dd',side:'over'},stocks).cur,3);
  assert.equal(calc({stat:'td',side:'over'},stocks).hit,true);
  assert.equal(calc({stat:'td',side:'under'},stocks).hit,false);
  assert.match(calc({stat:'td',side:'over'},stocks).txt,/STL 10, BLK 11/);
});
test('under wording says the bet can still be lost, and shows when it already has been', () => {
  assert.match(calc({stat:'pra',line:30.5,side:'under'},box).txt,/must stay under/);
  assert.match(calc({stat:'pra',line:20.5,side:'under'},box).txt,/already over the line, this under has lost/);
});
test('exactly on a whole-number line is a refund, not a win or a loss', () => {
  for (const side of ['over','under']) {
    const r=calc({stat:'pra',line:24,side},box);
    assert.equal(r.refund,true); assert.equal(r.hit,false); assert.match(r.txt,/refunded/);
  }
  assert.equal(calc({stat:'pra',line:23,side:'over'},box).refund,false);
});
