"""'Under' bets must never be described as 'over' in the slip, saved picks, saved bets or the live tab."""
import shutil, subprocess, textwrap
import pytest

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js needed for the page checks")

def run_js(body):
    js = textwrap.dedent("""
      const fs=require('fs');const html=fs.readFileSync('static/index.html','utf8');
      const src=html.match(/const sgn=[^\\n]*\\n/)[0]+html.match(/const legDesc=[^\\n]*/)[0];
      const statName={pts:'PTS',reb:'REB',pra:'Pts+Reb+Ast',dd:'Double-double',td:'Triple-double',fg3m:'3PM'};
      const legDesc=new Function('statName','esc',src+';return legDesc')(statName,x=>x);
    """) + textwrap.dedent(body)
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30); assert r.returncode == 0, r.stderr

def test_player_under_and_over_wording():
    run_js("""
      const eq=(a,b)=>{if(a!==b)throw new Error(a+' !== '+b)};
      eq(legDesc({stat:'pts',line:24.5,side:'under'}),'Under 24.5 PTS'); eq(legDesc({stat:'pts',line:24.5,side:'over'}),'Over 24.5 PTS');
      eq(legDesc({stat:'pra',line:40.5,side:'under'}),'Under 40.5 Pts+Reb+Ast'); eq(legDesc({stat:'reb',line:8.5}),'Over 8.5 REB');
    """)

def test_double_and_triple_double_no_bets_say_no():
    run_js("""
      const eq=(a,b)=>{if(a!==b)throw new Error(a+' !== '+b)};
      eq(legDesc({stat:'dd',side:'under'}),'No Double-double'); eq(legDesc({stat:'td',side:'under'}),'No Triple-double'); eq(legDesc({stat:'dd',side:'over'}),'Double-double');
    """)

def test_team_markets_keep_their_own_wording():
    run_js("""
      const eq=(a,b)=>{if(a!==b)throw new Error(a+' !== '+b)};
      eq(legDesc({stat:'total',line:218.5,side:'under'}),'Under 218.5 total points'); eq(legDesc({stat:'total',line:218.5,side:'over'}),'Over 218.5 total points');
      eq(legDesc({stat:'tt',line:110.5,side:'under'}),'Team total under 110.5');
    """)

def test_saved_picks_and_bets_use_the_shared_wording():
    html = open("static/index.html").read()
    assert html.count("legDesc(") >= 5 and "Over ${l.line} ${statName[l.stat]}`);" not in html          # the hard-coded 'Over' is gone from the shared helper
