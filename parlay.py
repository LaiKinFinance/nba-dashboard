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

def parlay(legs, preds, alphas, corr, n=60000, seed=7, prices=None, team_extra=None):
    # variables: one latent per (game, player, component)
    var = {}
    def add(pid, game, comp):
        if (game, pid, comp) not in var: var[(game, pid, comp)] = dict(i=len(var), game=game)
    for l in legs:
        pid, g, s = l["player_id"], l["game_id"], l["stat"]
        for c in (COMBO[s] if s in COMBO else (s,)): add(pid, g, c)
        if s in ("dd", "td"):
            for c in CORE: add(pid, g, c)       # needed for the latent link
    keys = sorted(var, key=lambda k: var[k]["i"]); k = len(keys); C = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            (ga, pa, ca), (gb, pb, cb) = keys[a], keys[b]; r = 0.0
            if pa == pb and ga == gb:
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
    for (game, pid, comp), v in var.items():
        pred = preds[(v["game"], pid)]
        if comp in ("dd", "td"):
            p = pred["p_dd"] if comp == "dd" else pred["p_td"]; vals[(game, pid, comp)] = (u[:, v["i"]] > 1 - p).astype(float)
        else: vals[(game, pid, comp)] = samples(comp, u[:, v["i"]], pred["proj"][comp], alphas)
    wins, pushes, probs = [], [], []
    for l in legs:
        game, pid, s = l["game_id"], l["player_id"], l["stat"]
        zero = np.zeros(len(u), dtype=bool)
        if s in ("dd", "td"):
            w = vals[(game, pid, s)] > 0.5
            if l.get("side", "over") == "under": w = ~w
            pu = zero
        else:
            x = sum(vals[(game, pid, c)] for c in (COMBO[s] if s in COMBO else (s,)))
            w = x > l["line"] if l.get("side", "over") == "over" else x < l["line"]
            pu = (x == l["line"]) if float(l["line"]).is_integer() else zero      # a tie on a whole-number line refunds the stake
        wins.append(w); pushes.append(pu)
        exact = leg_prob_single(preds[(l["game_id"], pid)], alphas, l)
        probs.append(float(exact) if exact is not None else float(w.mean()))
    joint = float(np.logical_and.reduce(wins).mean()) if wins else 1.0
    out = dict(leg_probs=probs, push_probs=[float(p.mean()) for p in pushes], independent=float(np.prod(probs)) if probs else 1.0, joint=joint,
               fair_odds=float(1 / joint) if joint > 0 else None)
    if prices is not None:
        out["ev"] = refund_ev(wins, pushes, prices, team_extra or [], rng_seed=seed + 1)
    return out

def refund_ev(wins, pushes, prices, team_extra, rng_seed=8):
    """Expected profit per 1 staked when a tie refunds that leg (the bet pays on the remaining legs' prices).
    wins/pushes: simulated yes/no per player leg; prices: decimal price for each player leg; team_extra: [(win_prob, push_prob, price)]
    for team legs (treated as independent of everything else). Returns None if any needed price is missing."""
    if any(p is None for p in prices) or any(t[2] is None for t in team_extra): return None
    n = len(wins[0]) if wins else 20000
    rng = np.random.default_rng(rng_seed); lost = np.zeros(n, dtype=bool); mult = np.ones(n)
    for w, pu, pr in zip(wins, pushes, prices):
        lost |= ~(w | pu); mult *= np.where(w, pr, 1.0)
    for pw, pp, pr in team_extra:
        r = rng.random(n); w = r < pw; pu = (r >= pw) & (r < pw + pp)
        lost |= ~(w | pu); mult *= np.where(w, pr, 1.0)
    return float(np.where(lost, -1.0, mult - 1.0).mean())

def value(joint, decimal_odds):
    """Edge and expected value per 1 staked; kelly fraction (full) for reference."""
    ev = joint * decimal_odds - 1
    kelly = max(0.0, (joint * decimal_odds - 1) / (decimal_odds - 1)) if decimal_odds > 1 else 0.0
    return dict(implied=1 / decimal_odds, edge=joint - 1 / decimal_odds, ev=ev, kelly=kelly)
