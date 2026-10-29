"""Multi-leg probability for player legs, with a Gaussian copula so legs from the same player or game are not treated as independent.

Supported player legs: pts, reb, ast, fg3m, stl, blk, pra, pr, pa (over/under a line) and dd, td (yes).
Combined markets (pra/pr/pa) are computed from the simulated component stats, so they stay consistent with single-stat legs."""
import numpy as np
from scipy import stats

DD_TD_CORR = {"dd": 0.45, "td": 0.5}   # latent link between double/triple-double legs and the player's own pts/reb/ast
CORE = ("pts", "reb", "ast")
ALL = ("pts", "reb", "ast", "fg3m", "stl", "blk")
COMBO = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast")}

def _psd(c):
    w, v = np.linalg.eigh(c); w = np.clip(w, 1e-6, None); c = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(c)); return c / np.outer(d, d)

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
        from model import nb_over
        p = float(nb_over(pred["proj"][s], alphas[s], leg["line"]))
    return p if leg.get("side", "over") == "over" else 1 - p

def parlay(legs, preds, alphas, corr, n=60000, seed=7):
    # variables: one latent per (player, component)
    var = {}
    def add(pid, game, comp):
        if (pid, comp) not in var: var[(pid, comp)] = dict(i=len(var), game=game)
    for l in legs:
        pid, g, s = l["player_id"], l["game_id"], l["stat"]
        for c in (COMBO[s] if s in COMBO else (s,)): add(pid, g, c)
        if s in ("dd", "td"):
            for c in CORE: add(pid, g, c)       # needed for the latent link
    keys = sorted(var, key=lambda k: var[k]["i"]); k = len(keys); C = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            (pa, ca), (pb, cb) = keys[a], keys[b]; ga, gb = var[keys[a]]["game"], var[keys[b]]["game"]; r = 0.0
            if pa == pb:
                if ca in ALL and cb in ALL: r = _same(corr, ca, cb)
                elif ca in DD_TD_CORR and cb in CORE: r = DD_TD_CORR[ca]
                elif cb in DD_TD_CORR and ca in CORE: r = DD_TD_CORR[cb]
                elif ca in DD_TD_CORR and cb in DD_TD_CORR: r = 0.9
            elif ga == gb and ca == cb and ca in CORE:
                pra_, prb_ = preds[(ga, pa)], preds[(gb, pb)]
                r = corr["teammate" if pra_["team"] == prb_["team"] else "opponent"].get(ca, 0.0)
            C[a, b] = C[b, a] = r
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(k), _psd(C), size=n); u = stats.norm.cdf(z)
    vals = {}
    for (pid, comp), v in var.items():
        pred = preds[(v["game"], pid)]
        if comp in ("dd", "td"):
            p = pred["p_dd"] if comp == "dd" else pred["p_td"]; vals[(pid, comp)] = (u[:, v["i"]] > 1 - p).astype(float)
        else: vals[(pid, comp)] = nb_samples(u[:, v["i"]], pred["proj"][comp], alphas[comp])
    wins, probs = [], []
    for l in legs:
        pid, s = l["player_id"], l["stat"]
        if s in ("dd", "td"): w = vals[(pid, s)] > 0.5
        else:
            x = sum(vals[(pid, c)] for c in (COMBO[s] if s in COMBO else (s,)))
            w = x > l["line"] if l.get("side", "over") == "over" else x < l["line"]
        wins.append(w)
        exact = leg_prob_single(preds[(l["game_id"], pid)], alphas, l)
        probs.append(float(exact) if exact is not None else float(w.mean()))
    joint = float(np.logical_and.reduce(wins).mean()) if wins else 1.0
    return dict(leg_probs=probs, independent=float(np.prod(probs)) if probs else 1.0, joint=joint, fair_odds=float(1 / joint) if joint > 0 else None)

def value(joint, decimal_odds):
    """Edge and expected value per 1 staked; kelly fraction (full) for reference."""
    ev = joint * decimal_odds - 1
    kelly = max(0.0, (joint * decimal_odds - 1) / (decimal_odds - 1)) if decimal_odds > 1 else 0.0
    return dict(implied=1 / decimal_odds, edge=joint - 1 / decimal_odds, ev=ev, kelly=kelly)
