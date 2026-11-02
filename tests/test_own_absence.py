"""A player on the injury list is never counted as his own missing teammate."""
import predict

def test_own_share_is_removed_for_that_player_only():
    feats = dict(abs_min=40.0, abs_pts=30.0, abs_reb=12.0, abs_ast=8.0, abs_n=1.5)
    found = [dict(player_id="embiid", weight=0.5, min=30.0, pts=17.0, reb=9.6, ast=5.5), dict(player_id="other", weight=1.0, min=25.0, pts=21.5, reb=7.2, ast=5.25)]
    mine = predict.own_absence((feats, found), "embiid")
    assert round(mine["abs_pts"], 2) == 21.5 and round(mine["abs_n"], 2) == 1.0
    assert predict.own_absence((feats, found), "maxey") == feats          # teammates still see him as possibly out
