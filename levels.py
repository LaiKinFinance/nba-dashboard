"""Bets that could finish level.

A whole-number line (over/under 20, handicap -4, total 220) can finish exactly on the line. Sportsbet then usually refunds that bet, but the
app does NOT work out chances, fair prices, value or stake advice for those lines; it says so instead.

Half-point lines (over 20.5) cannot finish level, and "20 or more" ("at least 20") cannot either: it wins on 20 and loses on 19.
These are different from "over 20", which is a level-possible bet."""

PLAYER_STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "pra", "pr", "pa")
EXPLANATION = ("This line is a whole number, so the result can finish exactly level. Sportsbet usually refunds a level bet, and this app "
               "does not work out chances, value or stake advice for it. Choose a half-point line (such as 20.5) or a \"20 or more\" bet to see numbers.")

def is_whole(x):
    return float(x).is_integer()

def can_finish_level(leg):
    """True if this leg could end exactly on its line (so the app gives no advice for it)."""
    if leg.get("kind") == "atleast": return False
    if leg["stat"] in PLAYER_STATS or leg["stat"] in ("total", "tt", "spread"): return is_whole(leg.get("line", 0))
    return False                                   # double/triple-double, winner and margin bands cannot finish level

def as_over_line(leg):
    """'20 or more' is the same bet as 'over 19.5'. Returns the leg in that plain form (other legs unchanged)."""
    if leg.get("kind") == "atleast": return dict(leg, side="over", line=float(leg["line"]) - 0.5, kind=None)
    return leg

def level_legs(legs):
    return [l for l in legs if can_finish_level(l)]
