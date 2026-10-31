"""A questionable starter that Rotowire lists in his slot AND in the 'may not play' list appears once."""
import slate

HTML = """<div class="lineup is-nba"><div class="lineup__abbr">UTA</div><div class="lineup__abbr">PHI</div>
<ul class="lineup__list"><li class="lineup__status">Expected Lineup</li>
<li class="lineup__player has-injury-status"><div class="lineup__pos">PF</div><a title="Lauri Markkanen">L</a><span class="lineup__inj">Ques</span></li>
<li class="lineup__player"><div class="lineup__pos">C</div><a title="Walker Kessler">W</a></li>
<li class="lineup__title">May not play</li>
<li class="lineup__player has-injury-status"><div class="lineup__pos">F</div><a title="Lauri Markkanen">L</a><span class="lineup__inj">Ques</span></li></ul>
<ul class="lineup__list"><li class="lineup__player"><div class="lineup__pos">C</div><a title="Joel Embiid">J</a></li></ul></div>"""

class R:  text = HTML

def test_questionable_starter_listed_once(monkeypatch):
    monkeypatch.setattr(slate.requests, "get", lambda *a, **k: R())
    side = slate.lineups_full()[0]["away"]
    names = [r["name"] for r in side["starters"] + side["out"]]
    assert names.count("Lauri Markkanen") == 1 and "Walker Kessler" in names
    assert next(r for r in side["out"] if r["name"] == "Lauri Markkanen")["pos"] == "PF"     # keeps the real position, not the generic F
