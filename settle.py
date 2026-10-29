"""Score predictions from final scores and box scores. No bookmaker payment rules.

Only completed picks with a clear outcome count as won or lost. A player who did
not play, or an exact tie on the line, is left out of the accuracy count.
"""

from levels import as_over_line

COMBO = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast")}
CATS = ("pts", "reb", "ast", "stl", "blk")

def _cmp(value, line, side):
    """Over/under against a line: 'won', 'lost' or 'push'."""
    if value == line: return "push"
    return "won" if (value > line) == (side == "over") else "lost"

def leg_result(leg, game, player):
    """game: {'home_pts','away_pts'} for a finished game, else None.  player: that player's box-score row, or the string 'missing' if the
    game's box score is stored but he is not in it (did not play), or None if the box score is not stored yet."""
    if not game: return None
    stat, side = leg["stat"], leg.get("side", "over")
    if stat in ("ml", "spread", "total", "tt", "band"):
        h, a = game["home_pts"], game["away_pts"]; margin = h - a
        if stat == "ml": return "won" if (margin > 0) == (side == "home") and margin != 0 else "lost"
        if stat == "spread":
            m = margin + float(leg["line"])                                   # the stored line is the HOME handicap
            if m == 0: return "push"
            return "won" if (m > 0) == (side == "home") else "lost"
        if stat == "total": return _cmp(h + a, float(leg["line"]), side)
        if stat == "tt": return _cmp(h if leg.get("team") == "home" else a, float(leg["line"]), side)
        m = margin if leg.get("team") == "home" else -margin                   # band: win by lo..hi
        return "won" if m >= leg["lo"] and (leg.get("hi") is None or m <= leg["hi"]) else "lost"
    leg = as_over_line(leg); stat, side = leg["stat"], leg.get("side", "over")      # "20 or more" is judged as over 19.5
    if player is None: return None
    if player == "missing": return "void"
    if stat in ("dd", "td"):
        n = sum(1 for k in CATS if (player.get(k) or 0) >= 10); yes = n >= (2 if stat == "dd" else 3)
        return "won" if yes == (side != "under") else "lost"
    val = sum(player[k] for k in COMBO.get(stat, (stat,)))
    return _cmp(val, float(leg["line"]), side)

def settle_bet(legs, results):
    """Keep incomplete picks pending and unscorable picks out of the win/loss count."""
    if not results: return "void"
    if any(r is None for r in results): return "open"
    if any(r in ("push", "void") for r in results): return "void"
    if "lost" in results: return "lost"
    return "won"
