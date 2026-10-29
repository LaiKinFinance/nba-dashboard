# Evidence report: fixes after Codex's review

Branches `review/1-…` to `review/13-…` are stacked in order (each builds on the one before). `review/13-daylight-saving-checks` holds everything.
`main` is unchanged (`1df220a`). `data/nba.db` and `data/models.pkl` have the same fingerprints as `main` on every branch (db `1a2fa6`, model `7336f8`).
Full suite on the final stack: **168 passed, 1 skipped** (`AUTO_REFRESH=0` is set before imports; every test works on a private copy of the data).
The skipped check needs a repaired copy of the data; it was run separately and **passed** (see "Repaired data").

## 1. Team-bet price age  (`review/1-team-price-age`)
- Reproduced: a 48-hour-old winner price was recommended with no age. Cause: only player prices had an age rule, and the screen decided VALUE on its own.
- Fix: `odds.line_ages` (age from the stored prices, like `fetch_ages`); one rule `STALE_H = 8` hours for player and team prices; old prices stay visible with their age but
  get `stale` / no edge, no VALUE tag (player strips, double/triple-double, winner/handicap/total), no entry in value picks; a game-level "prices are N hours old" note.
- Last good price is kept when the provider answers empty (test: 48-hour-old line still there, age still 48).
- Tests: `tests/test_team_price_age.py` (5).
- Note: with the existing 10-hour fetch cooldown, prices are often older than 8 hours, so VALUE labels will appear less often. That is the honest result; fetching more often costs Odds API credits.

## 2. Refunds and bad requests  (`review/2-…`, `review/3-refunds`)
- Codex e5d4385 cherry-picked. Reproduced refusals (HTTP 422) for: double-double with side=home, player market without player_id, team total without team, winner with an over/under side, band with upper < lower, band without team (`tests/test_request_validation.py`).
- Refunds: whole-number lines can tie. `parlay.parlay` now returns `push_probs` and, when each leg's own price is given, `ev` from a simulation where a refunded leg pays 1.00 and any loss loses the stake (`refund_ev`).
  Team legs: `team_leg_outcome` gives (win, tie) using half-point continuity corrections on whole-number lines; the smooth curve used to give the tie mass to the home/over side.
  If a bet can refund: no single fair price (`fair_odds: null`) and a note; value only if every leg has its own price (assumption stated: product of leg prices, refunded leg = 1.00); otherwise `value: null` with a reason.
  Whole-number bookmaker lines get no VALUE tag on the player strips or on the handicap/total buttons.
- Test: simulated EV matches hand calculation `(pw·price + pp)·… − 1` within 0.02 for independent legs (`tests/test_refunds.py`, 12 tests).
- Limit: the "pays the product of leg prices" assumption is not how a bookmaker prices same-game multis; value for refundable multis is only indicative.

## 3. Honest failures  (`review/4-honest-failures`)
- `refresh_odds` now returns `{problem}` from both provider calls, so `_stage` marks `odds` partial/failed instead of ok. Per-game errors are collected, kept in `odds_status.error`, and `last_ok` moves forward only on a clean run.
- Health: opens the database inside try; on failure returns HTTP 503 with a readable problem list; backup folder unreadable is a reported problem, not a crash; shows `prices.last_ok/last_error`.
- Tests: `tests/test_failure_reporting.py` (6).

## 4. No chances from an outdated model  (`review/5-outdated-calculations`)
- `guarded_projections()` runs before enrichment: upcoming games need a model that matches (schema, columns, model id); started games keep their own stored parameters (`alphas` are now saved inside every projection) and are labelled `frozen`;
  started games without stored parameters are left out rather than recomputed. Best/value skip frozen rows. Bets on started games are refused (400).
- UI: "Frozen" chip; banner also when the slate is empty because predictions are updating.
- Tests: `tests/test_outdated_calculations.py` (7).

## 5. Refresh queue  (`review/6-refresh-queue`)
- `POST /api/refresh` returns 202 at once; one job at a time (`request_refresh`, state in `_job`); `GET /api/refresh/status`; 429 within 10 minutes of a manual run. Training still runs in a low-priority child process.
- Tests: `tests/test_refresh_queue.py` (3), including "second request does not start a second job" and "nothing ran on the request thread".

## 6. Stronger checks  (`review/7-stronger-checks`)
- `validate(report, alphas)`: `n_test` must be a real whole number ≥ 1000 (NaN, inf, 1000.5, True, "5000", None rejected); every expected stat (pts, reb, ast, fg3m, stl, blk) needs a finite spread in (0, 5]; points correction values must be finite.
- Contract: bundles store `feature_columns`; `contract_problem` compares schema and exact columns; `model_stale` and `served_check` use it (an incompatible model returns `status=incompatible`, not `ok`).
- Check results are stored with `model_id`; health only uses results for the current model and counts the rest as ignored.
- Published predictions: `prediction_log` (table, written in `predict.save`) keeps what was published before tip-off; `published_check` compares it with actual results and measures uncertainty by resampling whole games (400 draws). The old history check is renamed `rebuilt_history` and labelled diagnostic.
- Not done: using bookmaker lines inside the published check (the ladders are model lines). Graded published picks with bookmaker prices are on the Track tab.
- Tests: `tests/test_model_gate.py`, `tests/test_published_check.py`, `tests/test_served_check.py`.

## 7. Points  (`review/8-points-correction`, `tools/points_study.py`)
Protocol (all by game date): base model trained before 2026-01-15; correction chosen on 2026-01-15 → 2026-03-15; **test = 2026-03-15 onward, looked at once** (4,388 player-games).
Findings (test period, lines = floor(projection)+0.5 shifted −4…+4):

| variant | Brier | log loss | mean gap | ≥75% chances: predicted / actual / gap (n) |
|---|---|---|---|---|
| original (spread from training games) | 0.2307 | 0.6565 | −0.041 | 0.821 / 0.751 / +0.070 (3757) |
| spread from calibration games | 0.2305 | 0.6547 | −0.051 | 0.818 / 0.765 / +0.053 |
| scale + spread (max. likelihood) | 0.2281 | 0.6488 | −0.022 | 0.817 / 0.753 / +0.064 |
| scale + spread + "bad night" share (kept) | 0.2275 | 0.6476 | −0.007 | 0.809 / 0.759 / +0.050 (3037) |

- Bias is real: projections were 0.5–0.7 points too low in the calibration and test periods (13.9 projected vs 14.5 actual).
- The over-confidence is not fixed by scale and spread alone. By month (≥75% gap, mixture): Mar +3.1, Apr +3.4, May +4.2, Jun +1.0 (n=26), **Oct 2026 +10.7 (n=654, first weeks of a new season)**. A 6-week rolling re-fit did not help.
- Kept correction: `q`≈5% of games the player scores ~15% of the usual level, scale ×1.09, spread 0.125 (fitted on the calibration window; accepted only if it beats the plain distribution by ≥3 log-likelihood on the later 40% of that window; calibration-fit improvement 83.8 for 4 parameters).
- One distribution everywhere: `model.over_prob` / `model.samples` are used by strips, ladders, boost/fade, single-leg exact chances, parlay simulation and PRA/PR/PA simulation. Version `pts-mix-1` is saved in the model's `alphas["pts_fix"]` and in every projection.
- Because the high-chance gap remains (and is worst early in a season), `POINTS_CLAIMS_ENABLED = False`: no VALUE for pts/pra/pr/pa, no points legs in Best picks, a note in Player Props. Rebounds, threes, steals, blocks were close earlier (gaps 0.5–2.7 points); assists about 2.7 points high; not changed.
- Not done: a season-start widening (the earlier season has no history to fit it on); this needs more data.
- Tests: `tests/test_points_correction.py` (14 incl. samples match formula within 0.004, parlay joint equals single, combined totals change).

## 8. Small improvements
- `review/9-name-matching` (Codex 4e6080a, unchanged), `review/11-dark-slip-text` (Codex fa93b00, unchanged; dark slip text now 143,161,191 on 20,32,52), `review/12-keyboard-buttons` (Codex 665a9fe; one conflict with my points note resolved by keeping both; keyboard Enter toggles `aria-expanded`, checked in headless Chromium).
- `review/10-live-progress` (Codex 5af6e56 + adaptation): under wording ("must stay under" / "already over, this under has lost"), refund state for whole-number lines (player and team legs, shown as "Refunded"), five-category `catCount` and double/triple-double text. JS tests: 7 (`tests/ui_progress.test.cjs`).
- `review/13-daylight-saving-checks`: 9 boundary cases (clock changes, midnight) for `ingest.et_date`, `repair.et_date`, `app.et_day`; no production change.

## Repaired data
`REPAIRED_DATA_DIR=<repaired copy> pytest tests/test_phase1_repair.py` **passed** (dates: no team plays twice on one Eastern date; every game has a tip-off; >500 dates differ from UTC; ≥400 players on rosters).
Upgrade rehearsal (copy of today's bundled data and old model, final code, `refresh_once`): see `RELEASE_PLAN.md` for the figures.

## Process note (a mistake, corrected)
Commit `6a1a4ca` (refunds) had accidentally included a changed `data/nba.db` (caused by importing the app without `DATA_DIR`). I removed it and rebuilt branches 3–13 on top, then force-pushed those review branches (with lease). Every branch now has the same db/model fingerprints as `main`.

## Could not verify
- A real deployment of any of this (nothing deployed). Memory use of training on Fly's 1 GB machine (see release plan for what I measured locally).
- Real Odds API behaviour with a live key (no key in this environment); the refund and failure paths are tested with simulated provider answers.
- Whether Fly keeps daily volume snapshots (I cannot see the account).
- UI beyond a headless Chromium run on a scratch copy; no phone testing.
- That points chances are fixed: they are not (see section 7).
