"""Bets that could finish level get no advice; ordinary and "20 or more" bets still work; old records survive."""
import numpy as np, pytest
from fastapi.testclient import TestClient
import app, parlay, model, levels

ALPHAS = {k: 0.1 for k in parlay.ALL}; CORR = {"same6": np.eye(6).tolist(), "teammate": {}, "opponent": {}}
PRED = dict(proj={**{k: 3.0 for k in parlay.ALL}, "pts": 20.0, "reb": 8.0, "ast": 5.0}, team="A", p_dd=.1, p_td=.01, start="2999-01-01T00:00Z")
GP = dict(source="market", home="NY", away="BOS", p_home=.6, p_away=.4, start="2999-01-01T00:00Z", home_pts=112., away_pts=106., margin=5.0, total=218.0,
          sigma_margin=13., sigma_total=17., handicaps=[], totals=[], team_totals={}, bands=[dict(team="home", lo=1, hi=5, p=.2), dict(team="home", lo=6, hi=None, p=.3)])

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app, "ACCESS_CODE", ""); monkeypatch.setattr(app, "RATE", (10**6, 60))
    monkeypatch.setattr(app, "projections", lambda: [dict(PRED, game_id="g", player_id="p")]); monkeypatch.setattr(app, "game_preds", lambda: {"g": GP})
    monkeypatch.setattr(app, "require_current_model", lambda preds=None: dict(alphas=ALPHAS, corr=CORR))
    return TestClient(app.app, raise_server_exceptions=False)

def leg(stat="reb", line=6.5, side="over", **k): return dict(game_id="g", player_id="p", stat=stat, line=line, side=side, **k)
def team(stat, line=None, side="home", **k): return dict(game_id="g", stat=stat, side=side, **({"line": line} if line is not None else {}), **k)
def post(client, legs, **body): return client.post("/api/parlay", json=dict(legs=legs, **body)).json()

# ---- ordinary bets still work -------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("legs", [[leg("reb", 6.5)], [leg("reb", 6.5, "under")], [leg("ast", 3.5), leg("reb", 6.5)], [team("ml")], [team("spread", -4.5)],
                                  [team("total", 217.5, "over")], [team("tt", 110.5, "under", team="home")], [team("band", team="home", lo=6)],
                                  [leg("dd", 0)], [leg("reb", 6.5), team("ml")]])
def test_ordinary_bets_get_chances_and_value(client, legs):
    r = post(client, legs, odds=3.0)
    assert not r.get("unsupported") and 0 < r["joint"] < 1 and r["fair_odds"] and (r["value"] is not None) and "explanation" not in r

# ---- level-possible bets get an explanation instead of numbers -------------------------------------------------------------------------------------
LEVEL = [leg("reb", 6), leg("reb", 6, "under"), leg("pra", 30), leg("pr", 20), leg("pa", 20), leg("fg3m", 2), leg("stl", 1), leg("blk", 1.0),
         team("spread", -4), team("spread", 3, "away"), team("total", 218, "over"), team("total", 218, "under"), team("tt", 110, "over", team="home")]
@pytest.mark.parametrize("bad", LEVEL)
def test_whole_number_bets_get_no_advice_only_an_explanation(client, bad):
    for odds in (None, 2.0, 5.0):
        r = post(client, [bad], **({"odds": odds} if odds else {}))
        assert r["unsupported"] is True and r["no_advice"] is True and r["value"] is None and "level" in r["explanation"]
        for forbidden in ("joint", "fair_odds", "leg_probs", "independent", "refund_note", "push_probs", "ev", "kelly"): assert forbidden not in r, forbidden
        assert r["level_legs"] and "{" not in r["explanation"]

def test_one_whole_number_leg_removes_advice_for_the_whole_slip(client):
    r = post(client, [leg("reb", 6.5), team("spread", -4)], odds=3.0)
    assert r["unsupported"] is True and len(r["level_legs"]) == 1 and r["level_legs"][0]["stat"] == "spread" and "joint" not in r

@pytest.mark.parametrize("stat", ["dd", "td"]) 
def test_double_doubles_and_winners_are_never_level(stat):
    assert not levels.can_finish_level(dict(stat=stat, line=0, side="over")) and not levels.can_finish_level(dict(stat="ml", side="home"))
    assert not levels.can_finish_level(dict(stat="band", lo=5, hi=None, team="home", side="home", line=0))

# ---- "20 or more" is different from "over 20" ------------------------------------------------------------------------------------------------------
def test_twenty_or_more_works_and_is_the_same_bet_as_over_nineteen_and_a_half(client):
    more = post(client, [leg("pts", 20, kind="atleast")]); half = post(client, [leg("pts", 19.5)])
    exact = float(model.over_prob("pts", 20.0, ALPHAS, 19.5))
    assert not more.get("unsupported") and more["joint"] == pytest.approx(exact, abs=0.004) and more["leg_probs"][0] == pytest.approx(exact) and half["leg_probs"][0] == pytest.approx(exact)

def test_over_twenty_is_a_different_bet_and_is_not_calculated(client):
    r = post(client, [leg("pts", 20)]); assert r["unsupported"] is True and "joint" not in r
    assert post(client, [leg("pts", 20, kind="atleast")])["leg_probs"][0] > post(client, [leg("pts", 20.5)])["leg_probs"][0]     # 20+ is easier than over 20.5

def test_twenty_or_more_inside_a_combined_bet(client):
    r = post(client, [leg("reb", 7, kind="atleast"), leg("ast", 4.5)], odds=3.0); assert not r.get("unsupported") and r["value"]["ev"] is not None

@pytest.mark.parametrize("bad", [leg("pts", 20.5, kind="atleast"), leg("pts", 20, "under", kind="atleast"), leg("dd", 0, kind="atleast"), leg("pts", 0, kind="atleast")])
def test_badly_formed_or_more_bets_are_refused(client, bad):
    assert client.post("/api/parlay", json=dict(legs=[bad])).status_code == 422

def test_atleast_wording():
    import shutil, subprocess, textwrap
    if not shutil.which("node"): pytest.skip("Node.js needed")
    js = textwrap.dedent("""
      const fs=require('fs');const html=fs.readFileSync('static/index.html','utf8');
      const src=html.match(/const sgn=[^\\n]*\\n/)[0]+html.match(/const legDesc=[^\\n]*/)[0];
      const legDesc=new Function('statName','esc',src+';return legDesc')({pts:'PTS'},x=>x);
      const eq=(a,b)=>{if(a!==b)throw new Error(a+' !== '+b)};
      eq(legDesc({stat:'pts',line:20,side:'over',kind:'atleast'}),'20 or more PTS'); eq(legDesc({stat:'pts',line:20,side:'over'}),'Over 20 PTS');
    """)
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30); assert r.returncode == 0, r.stderr

# ---- prices and screens do not show whole-number lines ---------------------------------------------------------------------------------------------
def test_game_page_lists_whole_number_team_lines_in_a_note_not_as_bets():
    gp = dict(GP, handicaps=[dict(line=-4.5, p_home=.5, p_away=.5)], totals=[dict(line=217.5, p_over=.5, p_under=.5)])
    bl = {"spread": {-4.5: {"home": (1.9, "sportsbet")}, -4.0: {"home": (1.9, "sportsbet")}}, "total": {217.5: {"over": (1.9, "sportsbet")}, 218.0: {"over": (1.9, "sportsbet")}}}
    m = app.match_payload(gp, bl, 1.0, "t")
    assert [h["line"] for h in m["handicaps"]] == [-4.5] and [t["line"] for t in m["totals"]] == [217.5]
    assert m["level_lines"] == [dict(market="Handicap", line=-4.0), dict(market="Total", line=218.0)]
    assert not any("push" in h or "refund" in h for h in m["handicaps"] + m["totals"])

def test_the_page_has_no_refund_estimates():
    html = open("static/index.html").read()
    for gone in ("refund_note", "value_note", "r.value.assumption", "payout_odds", "Expected profit"): assert gone not in html, gone
    assert "No advice for this slip" in html and "whole-number line" in html

def test_no_refund_maths_left_in_the_code():
    for f in ("parlay.py", "settle.py"): assert "refund_ev" not in open(f).read() and "reduced_odds" not in open(f).read()
    src = open("app.py").read(); assert "team_leg_outcome" not in src and "push_probs" not in src and "refund_ev" not in src
