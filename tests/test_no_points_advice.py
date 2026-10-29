"""A slip with any points leg gets no profit, value or stake advice; the warning stays."""
import numpy as np, pytest
from fastapi.testclient import TestClient
import app, parlay

ALPHAS = {k: 0.1 for k in parlay.ALL}; CORR = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}
PRED = dict(proj={**{k: 3.0 for k in parlay.ALL}, "pts": 20.0, "reb": 8.0}, team="A", p_dd=.1, p_td=.01, start="2999-01-01T00:00Z")
GP = dict(source="market", home="NY", away="BOS", p_home=.6, p_away=.4, start="2999-01-01T00:00Z", home_pts=112., away_pts=106., margin=5.0, total=218.0,
          sigma_margin=13., sigma_total=17., handicaps=[], totals=[], team_totals={}, bands=[])

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "projections", lambda: [dict(PRED, game_id="g", player_id="p")]); monkeypatch.setattr(app, "game_preds", lambda: {"g": GP})
    monkeypatch.setattr(app, "require_current_model", lambda preds=None: dict(alphas=ALPHAS, corr=CORR))
    return TestClient(app.app, raise_server_exceptions=False)

def leg(stat, line=20.5, **k): return dict(game_id="g", player_id="p", stat=stat, line=line, side="over", **k)

@pytest.mark.parametrize("legs", [
    [leg("pts")], [leg("pra", 30.5)], [leg("pr", 25.5)], [leg("pa", 22.5)],
    [leg("pts"), leg("reb", 6.5)],                                                  # points mixed with a non-points player leg
    [leg("pts", 19.5, price=1.9), dict(game_id="g", stat="ml", side="home", price=1.7)],       # points mixed with a team leg, with prices
    [leg("reb", 6.5), leg("pts", 20.0, price=1.9), leg("reb", 7.5, price=1.8)],     # whole-number points line too
])
def test_points_slips_have_no_value_profit_or_stake_advice(client, legs):
    for odds in (None, 2.0, 5.0):
        body = dict(legs=legs, **({"odds": odds} if odds else {})); r = client.post("/api/parlay", json=body).json()
        assert r["value"] is None and r["no_advice"] is True and "points" in r["points_warning"].lower()
        assert "kelly" not in str(r) and "ev" not in r and 0 < r["joint"] < 1                   # the chance itself is still shown

def test_slips_without_points_keep_their_advice(client):
    r = client.post("/api/parlay", json=dict(odds=3.0, legs=[leg("reb", 6.5)])).json()
    assert r["value"]["ev"] is not None and "no_advice" not in r

def test_page_hides_the_stake_guide_for_points_slips():
    import shutil, subprocess, textwrap
    if not shutil.which("node"): pytest.skip("Node.js needed")
    js = textwrap.dedent("""
      const fs=require('fs');const html=fs.readFileSync('static/index.html','utf8');
      const m=html.match(/const POINTS_STATS=[\\s\\S]*?\\nfunction stakeGuide\\(r\\)\\{[\\s\\S]*?\\n(?=function clearSlip)/)[0];
      const run=(legs,r)=>new Function('legs','calc','localStorage','drawSlip','esc','r',m+';return stakeGuide(r)')(legs,{o:3},{bankroll:'1000'},()=>{},x=>x,r);
      const pts=[{stat:'pts'}], reb=[{stat:'reb'}], pra=[{stat:'pra'}];
      if(run(pts,{value:{edge:.1}})!=='') throw new Error('points slip shows stake advice');
      if(run(pra,{value:{edge:.1}})!=='') throw new Error('combined points slip shows stake advice');
      if(run(reb,{no_advice:true})!=='') throw new Error('no_advice slip shows stake advice');
      if(!run(reb,{value:{edge:.1}}).includes('Stake guide')) throw new Error('normal slip lost its stake guide');
    """)
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30); assert r.returncode == 0, r.stderr
