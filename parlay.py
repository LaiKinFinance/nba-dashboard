"""Multi-leg probability for player legs, with a Gaussian copula so legs from the same player or game are not treated as independent.

Supported player legs: pts, reb, ast, fg3m, stl, blk, pra, pr, pa (over/under a line) and dd, td (yes).
Combined markets (pra/pr/pa) are computed from the simulated component stats, so they stay consistent with single-stat legs."""
import numpy as np
from scipy import stats

DD_TD_CORR = {"dd": 0.45, "td": 0.5}   # latent link between double/triple-double legs and the player's own pts/reb/ast
CORE = ("pts", "reb", "ast")
ALL = ("pts", "reb", "ast", "fg3m", "stl", "blk")
COMBO = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast")}
# How a player's stats move with his team's result in the same game (4 seasons, top-3 players, measured against the betting market's
# expected margin and total): a big night goes with his team doing better than expected, and points/assists with a higher-scoring game.
TEAM_LINK = {"margin": {"pts": 0.136, "reb": 0.097, "ast": 0.144}, "total": {"pts": 0.198, "reb": -0.064, "ast": 0.131}}

def _psd(c):
    w, v = np.linalg.eigh(c); w = np.clip(w, 1e-6, None); c = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(c)); return c / np.outer(d, d)

from model import samples

def nb_samples(u, mu, alpha):
    """Map uniforms to negative-binomial counts with a lookup table (much faster than ppf)."""
    mu = max(float(mu), 0.05); nb = 1.0 / alpha; p = nb / (nb + mu)
    kmax = int(mu + 10 * np.sqrt(mu + alpha * mu * mu)) + 6
    cdf = stats.nbinom.cdf(np.arange(kmax + 1), nb, p)
    return np.searchsorted(cdf, u, side="left").astype(float)

def _same(corr, a, b):
    s6 = corr.get("same6")
    if s6 is not None: return float(np.array(s6)[ALL.index(a), ALL.index(b)])
    if a in CORE and b in CORE: return float(np.array(corr["same"])[CORE.index(a), CORE.index(b)])
    return 0.0

def leg_prob_single(pred, alphas, leg):
    """Exact marginal for single-stat and dd/td legs (None for combined markets, which need the simulation)."""
    s = leg["stat"]
    if s == "dd": p = pred["p_dd"]
    elif s == "td": p = pred["p_td"]
    elif s in COMBO: return None
    else:
        from model import over_prob
        if leg.get("side", "over") == "under":
            # At integer lines, equality is a push, not an under win.
            return float(1 - over_prob(s, pred["proj"][s], alphas, np.ceil(leg["line"]) - 1))
        p = float(over_prob(s, pred["proj"][s], alphas, leg["line"]))
    return p if leg.get("side", "over") == "over" else 1 - p

def _team_win(l, gp, zm, zt):
    """Did a team leg win, given the game's margin and total draws (standard normal, home view)? Each leg keeps its own exact chance."""
    from scipy.stats import norm
    k = l["stat"]; sm, st = gp["sigma_margin"], gp["sigma_total"]
    if k in ("ml", "spread"):
        q = (gp["p_home"] if k == "ml" else float(norm.cdf((gp["margin"] + l["line"]) / sm)))          # home side wins with chance q
        home = zm > norm.ppf(1 - q); return home if l["side"] == "home" else ~home
    if k == "total":
        q = float(norm.cdf((gp["total"] - l["line"]) / st)); over = zt > norm.ppf(1 - q); return over if l["side"] == "over" else ~over
    if k == "tt":
        sg = float(np.sqrt((st ** 2 + sm ** 2) / 4)); mu = gp["home_pts"] if l["team"] == "home" else gp["away_pts"]
        z = (st * zt + (sm if l["team"] == "home" else -sm) * zm) / np.sqrt(st ** 2 + sm ** 2)
        q = float(1 - norm.cdf((l["line"] - mu) / sg)); over = z > norm.ppf(1 - q); return over if l["side"] == "over" else ~over
    m = gp["margin"] + sm * zm; m = m if l["team"] == "home" else -m                                  # winning-margin band
    return (m >= l["lo"] - 0.5) & ((m <= l["hi"] + 0.5) if l.get("hi") else True)

def parlay(legs, preds, alphas, corr, n=60000, seed=7, team_legs=(), gps=None, own=None):
    """Joint chance of all legs. Team legs (winner, handicap, totals) share the game's margin and total with that game's player legs.
    own = {(game, player, stat): (average, swing)} gives a consistent player's result for that stat its OWN normal spread (the rule Best
    picks use). It replaces his simulated result once, keeping its rank in every draw (so the links to teammates and his team's result stay),
    and every pick on him reads that one shared result: any order, any mix of lines, over and under can never both win."""
    own = own or {}
    from levels import as_over_line
    legs = [as_over_line(l) for l in legs]        # "20 or more" is calculated as over 19.5
    team_legs = list(team_legs or [])
    # variables: one latent per (game, player, component)
    var = {}
    def add(pid, game, comp):
        if (game, pid, comp) not in var: var[(game, pid, comp)] = dict(i=len(var), game=game)
    for l in legs:
        pid, g, s = l["player_id"], l["game_id"], l["stat"]
        for c in (COMBO[s] if s in COMBO else (s,)): add(pid, g, c)
        if s in ("dd", "td"):
            for c in CORE: add(pid, g, c)       # needed for the latent link
    for l in team_legs:
        add("_margin", l["game_id"], "_margin"); add("_total", l["game_id"], "_total")
    for i, key in enumerate(sorted(var, key=lambda t: tuple(str(x) for x in t))): var[key]["i"] = i      # one fixed internal order: the picks' order never changes the draws
    keys = sorted(var, key=lambda k: var[k]["i"]); k = len(keys); C = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            (ga, pa, ca), (gb, pb, cb) = keys[a], keys[b]; r = 0.0
            if pa == pb and ga == gb:
                if ca in ALL and cb in ALL: r = _same(corr, ca, cb)
                elif ca in DD_TD_CORR and cb in CORE: r = DD_TD_CORR[ca]
                elif cb in DD_TD_CORR and ca in CORE: r = DD_TD_CORR[cb]
                elif ca in DD_TD_CORR and cb in DD_TD_CORR: r = 0.9
            elif ga == gb and ({ca, cb} & {"_margin", "_total"}) and ({ca, cb} & set(CORE)):
                tc, pc, pid = (ca, cb, pb) if ca in ("_margin", "_total") else (cb, ca, pa)
                r = TEAM_LINK[tc[1:]][pc] * ((1 if preds[(ga, pid)].get("home") else -1) if tc == "_margin" else 1)
            elif ga == gb and ca == cb and ca in CORE:
                pra_, prb_ = preds[(ga, pa)], preds[(gb, pb)]
                r = corr["teammate" if pra_["team"] == prb_["team"] else "opponent"].get(ca, 0.0)
            C[a, b] = C[b, a] = r
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(k), _psd(C), size=n); u = stats.norm.cdf(z)
    vals = {}
    for (game, pid, comp), v in var.items():
        if comp in ("_margin", "_total"): continue
        pred = preds[(v["game"], pid)]
        if comp in ("dd", "td"):
            p = pred["p_dd"] if comp == "dd" else pred["p_td"]; vals[(game, pid, comp)] = (u[:, v["i"]] > 1 - p).astype(float)
        else: vals[(game, pid, comp)] = samples(comp, u[:, v["i"]], pred["proj"][comp], alphas)
    totals = {}
    def total(game, pid, s):
        if (game, pid, s) not in totals:
            x = sum(vals[(game, pid, c)] for c in (COMBO[s] if s in COMBO else (s,)))
            if (game, pid, s) in own:                       # one shared result: same draw order, his own average and swing
                mu, sd = own[(game, pid, s)]
                rank = stats.rankdata(x + np.random.default_rng(seed + 991).random(len(x)) * 1e-6)      # tiny tie-break, fixed seed
                x = mu + sd * stats.norm.ppf((rank - 0.5) / len(x))
            totals[(game, pid, s)] = x
        return totals[(game, pid, s)]
    wins, probs = [], []
    for li, l in enumerate(legs):
        game, pid, s = l["game_id"], l["player_id"], l["stat"]
        if s in ("dd", "td"):
            w = vals[(game, pid, s)] > 0.5
            if l.get("side", "over") == "under": w = ~w
        else:
            x = total(game, pid, s)
            w = x > l["line"] if l.get("side", "over") == "over" else x < l["line"]
        wins.append(w)
        if (game, pid, s) in own:
            mu, sd = own[(game, pid, s)]; q = float(1 - stats.norm.cdf((float(l["line"]) - mu) / sd))
            probs.append(q if l.get("side", "over") == "over" else 1 - q)
        else:
            exact = leg_prob_single(preds[(l["game_id"], pid)], alphas, l)
            probs.append(float(exact) if exact is not None else float(w.mean()))
    for l in team_legs:
        z = lambda c: stats.norm.ppf(np.clip(u[:, var[(l["game_id"], "_" + c, "_" + c)]["i"]], 1e-9, 1 - 1e-9))
        wins.append(_team_win(l, gps[l["game_id"]], z("margin"), z("total")))
    joint = float(np.logical_and.reduce(wins).mean()) if wins else 1.0
    return dict(leg_probs=probs, independent=float(np.prod(probs)) if probs else 1.0, joint=joint, fair_odds=float(1 / joint) if joint > 0 else None)
