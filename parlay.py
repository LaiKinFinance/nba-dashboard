"""Parlay probability with a Gaussian copula so same-game legs are not treated as independent."""
import numpy as np
from scipy import stats
from model import nb_over

DD_TD_CORR = {"dd": 0.45, "td": 0.5}  # latent correlation of double/triple-double legs with the player's own pts/reb/ast
STATS = ("pts", "reb", "ast")

def leg_prob(pred, alphas, leg):
    """Probability that a single leg wins. leg: {stat, side: over|under, line (for pts/reb/ast)}."""
    s = leg["stat"]
    if s == "dd": p = pred["p_dd"]
    elif s == "td": p = pred["p_td"]
    else: p = float(nb_over(pred["proj"][s], alphas[s], leg["line"]))
    return p if leg.get("side", "over") == "over" else 1 - p

def _psd(c):
    w, v = np.linalg.eigh(c); w = np.clip(w, 1e-6, None); c = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(c)); return c / np.outer(d, d)

def parlay(legs, preds, alphas, corr, n=200_000, seed=7):
    """legs: list of leg dicts incl. player_id & game_id; preds: {(game_id, player_id): projection}."""
    ps = [leg_prob(preds[(l["game_id"], l["player_id"])], alphas, l) for l in legs]
    k = len(legs); C = np.eye(k)
    same = np.array(corr["same"])
    for i in range(k):
        for j in range(i + 1, k):
            a, b = legs[i], legs[j]
            if a["game_id"] != b["game_id"]: continue
            pa, pb = preds[(a["game_id"], a["player_id"])], preds[(b["game_id"], b["player_id"])]
            sa, sb = a["stat"], b["stat"]
            r = 0.0
            if a["player_id"] == b["player_id"]:
                if sa in STATS and sb in STATS and sa != sb: r = same[STATS.index(sa), STATS.index(sb)]
                elif sa in DD_TD_CORR and sb in STATS: r = DD_TD_CORR[sa]
                elif sb in DD_TD_CORR and sa in STATS: r = DD_TD_CORR[sb]
                elif sa in DD_TD_CORR and sb in DD_TD_CORR and sa != sb: r = 0.9
                elif sa == sb: r = 1.0
            elif sa == sb and sa in STATS:
                r = corr["teammate" if pa["team"] == pb["team"] else "opponent"].get(sa, 0.0)
            # flip sign if exactly one leg is an "under"
            if (a.get("side", "over") == "under") != (b.get("side", "over") == "under"): r = -r
            C[i, j] = C[j, i] = r
    C = _psd(C)
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(k), C, size=n)
    # leg wins when its uniform score is below p  (the latent is oriented so that "win" is the low tail)
    wins = z < stats.norm.ppf(np.array(ps))
    joint = wins.all(axis=1).mean()
    return dict(leg_probs=ps, independent=float(np.prod(ps)), joint=float(joint), fair_odds=float(1 / joint) if joint > 0 else None)

def value(joint, decimal_odds):
    """Edge and expected value per 1 staked; kelly fraction (full) for reference."""
    ev = joint * decimal_odds - 1
    kelly = max(0.0, (joint * decimal_odds - 1) / (decimal_odds - 1)) if decimal_odds > 1 else 0.0
    return dict(implied=1 / decimal_odds, edge=joint - 1 / decimal_odds, ev=ev, kelly=kelly)
