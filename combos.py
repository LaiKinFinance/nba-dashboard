"""Combined stat markets (points+rebounds+assists, points+rebounds, points+assists).

The three stats of one player move together (a big night tends to lift all of them), so the sum is simulated from the same
Gaussian copula the parlay engine uses, with negative-binomial marginals, instead of adding independent distributions."""
import numpy as np
from scipy import stats

KINDS = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast")}
CORE = ("pts", "reb", "ast")

def _psd(c):
    w, v = np.linalg.eigh(np.asarray(c, float)); w = np.clip(w, 1e-6, None); c = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(c)); return c / np.outer(d, d)

def simulate(mu, alphas, same, n=40000, seed=11):
    """mu: {pts,reb,ast}; returns {stat: samples}. Latents are positively oriented: high latent = high stat."""
    from parlay import nb_samples
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(3), _psd(same), size=n); u = stats.norm.cdf(z)
    return {s: nb_samples(u[:, i], mu[s], alphas[s]) for i, s in enumerate(CORE)}

def combo_lines(mu, alphas, same, lo=0.04, hi=0.97, cap=60):
    sim = simulate(mu, alphas, same); res = {}
    for kind, parts in KINDS.items():
        tot = sum(sim[s] for s in parts); lines = []
        for L in np.arange(0.5, 120.5, 1.0):
            q = float((tot > L).mean())
            if lo <= q <= hi: lines.append(dict(line=float(L), p_over=q))
        if len(lines) > cap: lines = sorted(sorted(lines, key=lambda x: abs(x["p_over"] - 0.5))[:cap], key=lambda x: x["line"])
        res[kind] = lines
    return res
