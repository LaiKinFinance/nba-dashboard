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

## 7. Points  (`review/8-points-correction`, corrected by `review/24-points-study-fix`)
Corrected study (`python3 tools/points_study.py 2026-01-15 2026-03-15 --fair`): base model trained before 2026-01-15; correction chosen on 2026-01-15 → 2026-03-15; **test = 2026-03-15 onward** (4,388 player-games in 381 games).
Games are never split between periods (the script stops if one game appears in two periods). Every version is scored on **the same bets**; uncertainty comes from resampling whole games.

| bets (same for every version) | version | predicted | gap (predicted − actual) | 95% range | Brier |
|---|---|---|---|---|---|
| **A**: bets the original model rated 75%+ (n=3,757, actual 0.751) | original | 0.821 | +0.070 | +0.053 to +0.089 | 0.1913 |
| | spread from calibration games | 0.785 | +0.034 | +0.014 to +0.051 | 0.1880 |
| | scale + spread | 0.785 | +0.034 | +0.016 to +0.054 | 0.1879 |
| | scale + spread + bad nights (kept) | 0.796 | +0.045 | +0.026 to +0.063 | 0.1883 |
| **B**: lines 1–4 below the projection (n=17,552, actual 0.662) | original | 0.661 | −0.002 | −0.014 to +0.009 | 0.2206 |
| | spread from calibration games | 0.629 | −0.033 | −0.044 to −0.022 | 0.2216 |
| | scale + spread | 0.643 | −0.020 | −0.033 to −0.008 | 0.2203 |
| | scale + spread + bad nights (kept) | 0.665 | +0.002 | −0.010 to +0.013 | 0.2194 |

What this changes from the first report: the earlier tables compared each version on its own "75%+" group, so the groups differed. On the same bets:
- Group **B** (chosen by line position, not by any model's chance) shows the kept version is well calibrated on average (+0.002) and has the best Brier score, but the gain over the original is small (0.2206 → 0.2194).
- Group **A** is chosen because the original model liked those bets, which partly builds in a "winner's curse" (this favours the later versions over the original). Even so, bets rated 75%+ by the kept version still run **about 4.5 points too confident** (range 2.6 to 6.3, which excludes zero). That is the case that matters for choosing bets, so points VALUE, "safe" labels and now **all profit/value/stake advice on slips containing points** stay off (`POINTS_CLAIMS_ENABLED = False`).
- The kept correction is still used everywhere (`model.over_prob` / `model.samples`); version `pts-mix-1` is saved with the model.
- In production training the fitting/checking split of the calibration window is made at **game boundaries** (`model.game_cut`; tests `test_split_between_fitting_and_checking_never_cuts_a_game`).
- Earlier month-by-month result (kept): the first weeks of a new season were worst (+10.7 points, n=654) and a 6-week rolling re-fit did not help.

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

## 9. Speed  (`review/15-speed`)
Cause (profiled): `enriched()` made ~13,000 one-at-a-time `scipy` calls (about 70% of its time); every tab (Best picks, Games, Value, Lineups) rebuilt it; `get_slate` rebuilt every game block once per player (78 times, not 13); `predict.project` ran 12 minutes **inside the web process** after the last live deploy; pages were not compressed (91 KB page, ~1.8 MB slate JSON).
- Fix: vectorised `over_prob` calls (identical numbers: every strip entry compared before/after, equal to 9 decimals); `enriched()` shared per (latest projection, latest price fetch, model file time, minute); each game block built once; `GZipMiddleware`; `project_quietly` and the daily checks run via `run_quietly` (a niced, one-core child process).
- Measured here (not on Fly): enriched 1.19 s → 0.40 s; Games page 1.27 s → 0.4 s cold and 0.07 s warm; Best/Value/Lineups ~0.07 s warm; page 95,654 → 29,980 bytes compressed. Compared full slate, best and value JSON old vs new: identical.
- Tests: `tests/test_speed.py` (cache refreshes when data changes; batched = one-at-a-time; gzip; heavy work goes to the child process).
- Not verified: speed on Fly's shared CPU.

## 10. Best picks first  (`review/16-best-picks-first`)
Tab order and landing page changed (the app always opens on Best picks, today's picks view even if the last visit was elsewhere); the games list is requested alongside best/value instead of before them. Checked in headless Chromium with old saved tab/view values set.

## 11. Automatic marking of My bets  (`review/17-auto-grade-bets`, `settle.py`)
- Rules: won / lost / push (tie on a whole-number line) / void (player did not play); a bet is lost as soon as any leg is lost; won when every leg has finished and none lost (push and void legs drop out); full refund (`void`) when every leg dropped out. A game whose box score is not stored yet stays pending (never "did not play").
- Payout: when legs dropped out and every leg has a saved price, payout odds are the product of the remaining legs' prices (`payout_odds`); otherwise the full odds are kept and a note says to check Sportsbet. New bets now save each leg's price.
- Manual marks are kept (`manual=1`); "undo" hands the bet back to automatic marking. `grade_bets()` runs in every refresh cycle and when the scorecard is opened.
- Tests: `tests/test_auto_marking.py` (47). Real-data smoke test on a copy of the repaired database: a bet on the top scorer of the latest game was marked won / lost / refunded as expected.
- Limits: same-game combos with correlated legs are still marked leg by leg (correct). Cash-outs and boosts cannot be seen by the app; that is what the manual override is for.

## 12. Fixes after Codex's second review
| # | Problem | Fix (branch) | Evidence |
|---|---|---|---|
| 1 | Shared page results could keep old prices/predictions; price age not re-checked | `review/18-stale-results`: key = database file stamps (+WAL), model file, and the minute; prices carry their time stamp (`odds_at`, `prices_at`) so the page re-ages them each draw; page refreshes the games list every 5 minutes; API answers are `no-store` | `tests/test_stale_results.py` (6): key changes on any write, new model, 61 s, 9 h; a 7.9-hour price is stale one hour later |
| 2 | Points slips showed profit/value/stake advice | `review/19-no-points-advice`: any slip with pts/pra/pr/pa returns `value: null`, `no_advice`, a points warning; the stake guide is hidden | `tests/test_no_points_advice.py` (9, incl. a node check of the page function): single, combined, mixed, whole-number lines, with and without entered odds |
| 3 | Started games still calculable / recommended | `review/20-started-games`: `has_started()`; Best picks, Value (player and team), and `/api/parlay` (player and team legs) refuse started games | `tests/test_started_games.py` (11) |
| 4 | Older version's results counted as current; record could change after start | `review/21-per-version-checks`: `published_check` returns one result per model id; each stored as its own record; Track record headline counts only the current version, earlier versions reported separately; `prediction_log` has `start`, updates only while `start` is in the future, and BEFORE UPDATE/DELETE triggers refuse changes afterwards; `save()` never logs a started game | `tests/test_published_check.py` (9) and `tests/test_track_versions.py` |
| 5 | Refund value ignored the entered combined price (2.0 and 5.0 gave the same profit) | `review/22-refund-price`: value shown only if every leg has a price AND the entered price matches the legs multiplied (±2%); otherwise hidden with the reason | `tests/test_refunds.py` (+4): 2.0 and 5.0 both hide the value; matching price shows it; better leg price gives better value |
| 6 | "Under" described as "over" | `review/23-under-wording`: shared `legDesc` says Under / No (double/triple-double); used by slip, saved picks, bets, live | `tests/test_leg_wording.py` (4, node) |
| 7 | Points study compared different bets, split rows not games | `review/24-points-study-fix` (see section 7 above) | study output; `tests/test_points_correction.py` (+2) |
| 8 | Four finished games with no box score | `review/25-missing-games`: ESPN lists them as final with scores (DEN 98–110 MIN 2026-04-30; CLE 97–107 DET 2026-05-07; CLE 117–113 DET 2026-05-13; NY 121–108 CLE 2026-05-23, all playoffs) but `summary` returns 404 on both ESPN hosts we can reach. They are recorded in `unavailable_games` with their scores and reason, re-tried on every ingest, and marked resolved (not deleted) if ESPN ever publishes them. Nothing was added to `games`/`player_games`. A weekly reconcile separates stored / known-unavailable / **unexplained** gaps; only unexplained gaps are a problem. | `tests/test_missing_games.py` (5); run on a copy of the repaired data: 4 recorded, 0 unexplained, 0 rows invented |
| 9 | Release plan, backups, going back | `review/26-backup-and-recovery`, `review/28-backup-first-release`, `RELEASE_PLAN.md` | `tests/test_backup_restore.py` (9); rehearsals below |

Known effect of item 8: those four games are missing from model training (no player rows). The affected teams' rest days and opponent-defence averages around them are slightly off. This is small and cannot be fixed without data ESPN does not publish.

### Rehearsals on copies (nothing touched live data)
- **Full upgrade** (copy of today's bundled data and old model, final code): see `RELEASE_PLAN.md`.
- **Data recovery:** backup bundle of a 5.7 MB database; a new bet, pick and comment were added; then all player points were zeroed on the live copy. `restore` produced a NEW database: points back to 318,970, the newer bet, pick and comment kept (comment re-linked to its pick), integrity ok, damaged file unchanged.
- **Code rollback:** the previous code (`1df220a`) run against data already upgraded by the new code: every page answered (200), it retrained its own model in 37 s, and bets/picks stayed readable.

## 13. Fingerprints and tests
- `data/nba.db` `1a2fa6`, `data/models.pkl` `7336f8` on every branch (same as `main`).
- Final stack `review/27-final-candidate`: **271 passed, 1 skipped** with automatic refresh off before imports and every test on a private copy of the data. The skipped repaired-data check was run separately on a repaired copy and **passed**. Full update rehearsal from a frozen copy of the final code: 189 s, every step ok, health ok.
- One rehearsal run gave a false alarm ("model out of date") because I switched code versions in the working folder while it was running; it was repeated from a frozen copy and passed.
- Not verified: speed and memory on Fly's machine; real Odds API behaviour (no key here); phones; whether Fly keeps daily volume snapshots.

## 14. Simplification: no refund calculations  (`review/29-simplify-no-refund-maths`, built on `review/27-final-candidate`)
Mark's decision: remove automatic refund calculations and estimates. This supersedes section 2 (refunds) and the refund rows in section 12.
- **Removed:** tie chances (`push_probs`), the refund-aware profit simulation (`refund_ev`), reduced payouts (`reduced_odds`, `payout_odds` is no longer written), the "refund" notes and the "Expected profit" refund box, `team_leg_outcome`.
- **Bets that cannot finish level work as before:** half-point lines (over 20.5), winners, margin bands, double/triple-doubles. Tested for chances, fair odds, value, the stake guide and the game page.
- **Bets that could finish level get no advice:** any whole-number player line (over/under 20), handicap, game total or team total. The slip returns `unsupported`, `no_advice`, `value: null`, a plain explanation and the list of affected lines, and **none** of: chance, fair odds, leg chances, value, stake guide. One such leg removes advice for the whole slip. The game page lists such bookmaker lines in a one-line note instead of offering them as bets; player cards do the same (`level_lines`).
- **"20 or more" (`kind: "atleast"`)** is a separate bet type: wins on 20, loses on 19, calculated as over 19.5 (tests show its chance equals over 19.5 and is higher than over 20.5), judged correctly by the automatic marking and the live tab, worded "20 or more PTS". "Over 20" stays a level-possible bet (no advice). Badly formed "or more" bets (half-point line, under, double-double, zero) are refused.
- **Old records kept:** no table or column was dropped (`payout_odds`, `leg_results` with push/void remain readable; a test loads a legacy record).
- **Money actually returned:** if a finished bet has a level or cancelled leg and nothing lost, the app does not decide the result or the payout. The bet stays "open", flagged `needs_amount`, and My bets shows "Tell us what Sportsbet paid back". `POST /api/bets/{id}/returned` stores the amount typed (0 or more); the result follows from it (more than the stake: won; equal: stake returned; less: lost) and profit uses that amount only. Bets still waiting are left out of profit and counted (`awaiting_amount`). A bet with a lost leg is still marked lost at once (its stake is lost whatever happened to the other legs).
- **Tests:** `tests/test_no_level_advice.py` (37), `tests/test_enriched_levels.py` (2), changes in `tests/test_auto_marking.py` and `tests/ui_progress.test.cjs`; `tests/test_refunds.py` removed. Suite: **304 passed, 1 skipped**. Real-data check on a copy: over 11.5 rebounds gives chance 0.435/value; over 11.0 gives the explanation and no numbers; "11 or more" gives 0.520 (equal to over 10.5); a slip with handicap -3.5 works, with -4 it shows no numbers.
- **Not verified:** how Sportsbet actually settles each kind of level bet (the app no longer claims to know).
