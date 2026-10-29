# Audit guide (for Codex / any reviewer)

Repo: https://github.com/LaiKinFinance/nba-dashboard  ·  Live: https://nba-dashboard.fly.dev (access code is a Fly secret, not in the repo)
Purpose: dashboard that projects NBA top-3 players per team (points, rebounds, assists, 3PM, steals, blocks, combos, double/triple-double),
prices parlays with correlation, and compares against bookmaker odds (The Odds API, `au` region, Sportsbet preferred).
Friends use it through a shared access code; picks/bets/comments are stored in SQLite on a Fly volume.

## Data flow
1. `ingest.py` — ESPN public JSON: scoreboard per day + box score per game → `data/nba.db` (`games`, `player_games`). Skips non-NBA (All-Star) games.
   `backfill_lines.py` fills ESPN/DraftKings market total/spread/moneylines per game.
2. `features.py` — leak-free rolling features (everything shifted by one game), opponent defence, teammate-absence features (`absence.py`),
   tonight's starter flag. `top3.py` picks each team's top-3 by recent production.
3. `model.py` — gradient-boosted Poisson-loss regressors per stat (+ negative-binomial dispersion), logistic triple-double model, GBM double-double classifier.
   `team_model.py` — margin-Elo + scoring-form formula; **in backtests the market line beat it, so match markets use the market as the central estimate**.
4. `predict.py` — trains, builds tonight's projections (injury/lineup feeds → absences, counterfactual "full strength" and "recent norm" projections,
   evidence from the player's own with/without history), stores JSON payloads in `projections` / `game_preds`.
5. `parlay.py` — Gaussian-copula multi-leg probability (same-player, teammate, opponent correlations estimated from held-out residuals).
   `combos.py` — pts+reb+ast style markets from the same copula.
6. `odds.py` — The Odds API client (credit-aware, per-game cooldown), stores all bookmakers; `slate.py` — ESPN schedule/injuries + Rotowire lineups (HTML scrape).
7. `app.py` (FastAPI) + `static/index.html` (single-file vanilla JS UI, PWA). `live.py` — live scores/box scores from ESPN.

## Commands
```
pip install -r requirements.txt
python tools/audit_data.py        # data-integrity checks (exit 1 on FAIL)
python model.py                   # player-model backtest (time split at 2026-02-01)
python team_model.py              # team backtest, market vs Elo
python predict.py --train         # retrain + re-project (writes data/)
DATA_DIR=data uvicorn app:app --port 8000   # APP_ACCESS_CODE=... to require a code
```

## Claims we made that you should try to break
- Player stats beat a plain 10-game average only slightly (pts MAE ≈ 5.36 vs 5.60). Teammate-absence features removed a ~1 pt under-prediction when 15+ ppg of teammates are out.
- Steals/blocks models are no better than a plain average (flagged low-confidence in the UI).
- Market margin MAE 11.2 vs Elo 11.8; market total MAE 14.2 vs 15.0 (656 test games).
- "VALUE" tags and the Value-bets card compare model chance with bookmaker price, leaning 30% toward the market; confidence labels are heuristic points.

## What we want from the audit (please report as a table: severity · file:line · evidence · suggested fix)
1. **Correctness bugs** in any module (esp. `features.py` leakage, `absence.py` bookkeeping, `parlay.py` copula orientation, `team_model.py` probability maths, odds parsing, name matching in `slate.py`).
2. **Data quality**: run `tools/audit_data.py`, then look for what it does not check (stale rows, wrong team after trades, minutes of DNP players, duplicate players with different ids, timezone/date-boundary bugs between ESPN UTC times and US dates).
3. **Statistical validity**: calibration of probabilities, whether the negative-binomial + Gaussian-copula assumptions are defensible, overfitting, sample-size problems (e.g. 76 triple-doubles in the test set), and whether the "edge" logic could mislead a bettor.
4. **Security**: shared-secret auth in `app.py`, SQL injection, XSS in `static/index.html` (template strings), CORS, rate limiting, secrets handling (GitHub Actions workflow, Fly), scraping/ToS risk (Rotowire, ESPN undocumented API).
5. **Reliability/performance**: background refresh loop, SQLite concurrency on a Fly volume, request cost of `/api/slate`, caching.
6. **Testing gaps**: there are no automated tests; propose and (if you can) add a small pytest suite for the riskiest logic.
7. **UX/accessibility** and anything confusing for non-technical users.

Please work on a branch named `codex-audit`, do not push to `main`, do not add secrets, and keep each fix small and separately commit-able. Put the findings in `AUDIT_FINDINGS.md`.
