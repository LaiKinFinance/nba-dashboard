const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8');
const source = html.split('// BEGIN playerLegProgress:')[1].split('\n').slice(1).join('\n').split('// END playerLegProgress')[0];
const context = {
  statName: {pts:'PTS', reb:'REB', ast:'AST', pra:'Pts+Reb+Ast', pr:'Pts+Reb', pa:'Pts+Ast'},
  catCount:r=>Number(r.pts>=10)+Number(r.reb>=10)+Number(r.ast>=10),
};
vm.createContext(context);
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
