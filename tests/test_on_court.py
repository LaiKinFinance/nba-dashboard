"""Who is on the floor: starters plus substitutions in order (ESPN summary format)."""
import live

def _js(starters, subs):
    ath = lambda i, st: dict(athlete=dict(id=i), starter=st)
    box = [dict(team=dict(id="1"), statistics=[dict(athletes=[ath(i, i in starters["1"]) for i in "abcdefgh"])]),
           dict(team=dict(id="2"), statistics=[dict(athletes=[ath(i, i in starters["2"]) for i in "ABCDEFGH"])])]
    plays = [dict(type=dict(text="Substitution"), sequenceNumber=str(n), team=dict(id=t), participants=[dict(athlete=dict(id=i)), dict(athlete=dict(id=o))])
             for n, (t, i, o) in enumerate(subs, 1)] + [dict(type=dict(text="Jump Shot"), sequenceNumber="99", team=dict(id="1"), participants=[])]
    return dict(boxscore=dict(players=box), plays=plays)

def test_starters_then_substitutions_in_order():
    js = _js(dict({"1": set("abcde"), "2": set("ABCDE")}), [("1", "f", "a"), ("1", "a", "b"), ("2", "F", "E")])
    # team 1: f replaces a, then a returns for b -> a, c, d, e, f ; team 2: F replaces E
    assert live._on_court(js) == set("acdef") | set("ABCDF")

def test_a_team_without_exactly_five_gets_no_one_marked():
    js = _js(dict({"1": set("abcd"), "2": set("ABCDE")}), [])
    assert live._on_court(js) == set("ABCDE")
