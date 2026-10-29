# Independent review of Codex's audit (AUDIT_FINDINGS.md)

Reviewer: Claude · scope: findings checked against `main` (7833bde) and the bundled data. **The `codex-audit` branch was not available to me (it exists only on Codex's machine),
so I have NOT reviewed its code changes or its 44 tests.** Everything below about the audit's *findings* was re-tested independently; its *fixes* are still unreviewed.

## 1. Verdicts on the findings (what I reproduced)

| # | Finding (Codex) | Verdict | My evidence |
|---|---|---|---|
| 1 | `ingest.py` INSERT supplies 10 values to a 20-column table | **Confirmed, high, and live-impacting** | Reproduced on the real schema: `table games has 20 columns but 10 values were supplied`. The refresh loop runs ingestion first, so one new game aborts the whole cycle (no new box scores, projections, pick logging or grading). Introduced by my own migration; the one earlier test run found "0 new games", so the insert was never exercised. Fixed on `hotfix/ingest-insert-and-key-redaction` (e7474fe); that fix ingested the 4 missing Oct 28 games on a scratch copy. |
| 2 | API key can reach logs / `odds_status` / `/api/meta` through `requests` error text | **Confirmed, high (latent)** | `HTTPError` text contains `?apiKey=...`; `odds.refresh` stored `str(e)` and `/api/meta` returned it. I cannot inspect production, so I do not know whether it has happened. Fixed on the same branch (58cfd91): sanitized errors, scrubbing on write and read. |
| 3 | Upcoming-game placeholder rows contaminate each other's features | **Confirmed, high (for back-to-backs)** | Jalen Brunson, second of two upcoming games: `min_m5` 36.0 → 27.8, `pts_m5` 27.2 → 18.2, rest 2 → 1 day, because the first game's zero placeholder is read as history. Affects every player whose team plays twice in the 2-day window. |
| 4 | Parlay engine keys latents by (player, component), not game | **Confirmed, medium** | Same player in two games, means 30 and 12 pts, both "over 24.5": leg probabilities 0.658 and 0.020 but joint 0.657, which is impossible (joint can never exceed the smaller leg). |
| 5 | UTC calendar dates vs US basketball dates | **Confirmed, high** | 130 player/date groups and 15 team/date groups contain two games (e.g. ATL, 2025-11-23: NO@ATL and ATL@CHA). Rest days, same-day form and the 2026-02-01 train/test cut are affected. |
| 6 | Fixed UTC-5 "Eastern" in `slate.py` / `live.py` | **Confirmed, low-medium** | Code reading; wrong all of the DST period (October included). `app.py` already uses `ZoneInfo`. |
| 7 | Dispersion fitted on training residuals understates real variance | **Confirmed, high** | Same numbers as Codex: alpha training vs later games: points 0.085 vs 0.134, rebounds 0.016 vs 0.068, assists 0.001 vs 0.081. Assist legs the model rated 62–86% (n=21) hit 57%. Small sample, but the direction matches. |
| 8 | "Current team" = last team a player played for | **Confirmed, high** | Compared with ESPN's current rosters: 25 of 497 players differ (e.g. Nic Claxton BKN→CHI, Santi Aldama MEM→DAL, Brandon Ingram TOR→LAC). Aldama has appeared in a published best pick as a Grizzly. |
| 9 | Historical absences use hindsight (who played later) | **Confirmed by design, high for claims** | It was a documented design choice (a player counts as absent only if he plays for that team later in the season), but it means "leak-free" was overstated and the measured benefit (bias 0.86 → 0.23 pts) is not a pregame result. |
| 10 | Four completed Oct 28 games missing from the database | **Confirmed, cause partly uncertain** | ESPN lists them as final, our DB lacks them. Consistent with finding 1 (ingestion failing), but they may also simply have finished after my last snapshot. |
| 11 | DD/TD should use five categories | **Rule confirmed; practical effect small, so I rate it low** | Whole database: 147 official triple-doubles vs 146 by our rule (Wembanyama, 2026-05-05: 11/15/5 plus 12 blocks); double-doubles 2,152 vs 2,152. Correct to change, not urgent. |
| 12 | DD/TD built from separate latents can price impossible combinations | **Confirmed (code and Codex repro), medium** | Binary latents with fixed 0.45/0.5 links; "TD yes + DD no" gets non-zero probability. Product choice: derive from simulated counts, or block overlapping legs. |
| 13 | Live-tab progress for combined markets is broken | **Confirmed, medium** | `cur=r[l.stat]` for `pra/pr/pa` reads a field the box score never has. Under-legs also treated as overs. |
| 14 | `.slip .mute` colour unreadable in dark theme | **Confirmed, low** | CSS rule `.slip .mute{color:#3a3a3a}` on a dark card. |
| 15 | "30% market blend" does nothing | **Confirmed, low but misleading** | adjusted edge = 0.7 × raw edge exactly; ranking unchanged. It is a haircut, not shrinkage toward the market. |
| 16 | Team moneyline vs bands vs handicap use different distributions | **Confirmed, medium** | Winner chance from no-vig moneyline, bands/handicaps from a normal; they do not add up consistently. Integer lines / pushes are not priced. |
| 17 | Same-game team and player legs multiplied as independent | **Confirmed limitation; product choice** | Already warned for mixed slips; EV is still shown. |
| 18 | Selection prefers Sportsbet even when another book pays more | **Disagree that it is a defect** | Deliberate (you bet on Sportsbet); `alt` badge shows better prices. Needs clearer labelling only. |
| 19 | Stored-XSS through saved pick legs; shared code in localStorage; no rate limits; open if no code set | **Accept as real, severity depends on trust model** | Friends-only app, but a stored script would run for every friend and could read the stored code. Cheap to fix. `/api/refresh` is callable by any signed-in user. |
| 20 | Refresh loop: retrains on every start, daemon thread per process, non-atomic pickle writes, no single-flight lock, stale rows | **Accept, medium** | Code reading. One Fly machine, so risk is moderate. |
| 21 | Prices have no age; cooldowns up to 10 h; an empty response can erase prices | **Accept, high for "value" claims** | A VALUE tag on a price that may be hours old is misleading. Free-tier credits force a trade-off. |
| 22 | Rotowire/ESPN terms restrict automated extraction | **Accept as a risk, not verified** | I cannot judge the legal text; it is a business decision (authorized feeds vs personal-scale use). |
| 23 | Referee, Elo carry-over, NBA Cup final (neutral site, 83 games) etc. | **Accept, low** | Cup final is labelled regular season; team model now uses the market, so Elo issues are minor. |

Nothing in the audit I would call wrong. Differences are severity (item 11), framing (item 18) and urgency (items 1–2 are more urgent than ranked).

## 2. Phased plan (small, separately committed; nothing to main without your say-so)

**Phase 0: stop the bleeding (no model change, can ship with the current saved model).**
1. Merge the two hotfix commits (ingest INSERT, key redaction). Check the live app's logs for the `20 columns` error. **If the Odds API key may ever have appeared in an error or status text, rotate it.**
2. After review of `codex-audit`, cherry-pick only commits that do **not** change model inputs: stored-XSS validation, parlay game key, odds parsing, name matching, DST, slip colour, accessibility buttons, live combo progress, tests. Keep them off any branch that also changes features.

**Phase 1: correct dates, availability history, rosters (data repair first; no retraining until done).**
1. Add `tip_utc` and `et_date` columns (keep the old `date`), backfill from ESPN scoreboards (about 350 cheap calls), recompute rest days and the train/test cut on tip time.
2. Reconcile completed schedule IDs against ingested IDs; retry and persist failures; list the 9 incomplete box scores as quarantined. Label the NBA Cup final (neutral site).
3. Current rosters from ESPN's roster endpoint, stored with dates; top-3 and absences use it; players not on the roster are excluded.
4. Start archiving injury and lineup snapshots (time-stamped, before tip-off). Until months of real snapshots exist, treat the absence and starter features as **hindsight features**: stop calling the model pregame-validated, and retrain a baseline without them for comparison.
5. Five-category DD/TD.

**Phase 2: calibration and consistent pricing (after Phase 1 data).**
1. Estimate dispersion from out-of-fold predictions; separate calibration period from the final untouched test; rolling origins.
2. Calibrate probabilities per market; report calibration at real bookmaker lines once timestamped prop lines exist (forward archive; none today).
3. One coherent joint model: DD/TD derived from simulated counts, one discrete no-tie margin distribution for winner/handicap/bands, push handling, or block unsupported combinations and hide their EV.
4. Replace the "30% market blend", "Strong/Medium" and "safe-ish" wording with an evidence checklist; the Track record tab becomes the forward test.

**Phase 3: security.** Fail closed if no access code in production; per-friend invite tokens instead of a typed name; remove the stored code from localStorage; rate limits and body limits; `/api/refresh` admin-only; replace inline handlers and add a CSP; pin the GitHub Action, `.dockerignore`, non-root container.

**Phase 4: reliability.** One scheduler started on app start (single-flight lock, persisted last-train time, one failing stage no longer skips the rest); atomic, versioned model files with the previous good one kept; SQLite WAL and busy timeout; price timestamps, staleness limits and last-good prices; cached enriched slate; off-host backups with a tested restore.

**Phase 5: labels, accessibility, tests.** "Model fair price" vs "bookmaker price" labelled everywhere; no "safe" language; dependence warnings that also suppress EV; keyboard and screen-reader support; pytest as a required deployment gate with archived provider fixtures.

## 3. Release rule for model-affecting changes
Do **not** deploy changed feature logic (absence, placeholders, dates, rosters) with the saved model: the bundled `models.pkl` and stored projections encode the old feature definitions, and the app currently retrains on every start without validation. Required first: (1) data repairs above, (2) offline retrain, (3) a validation report (rolling origins, calibration, simple baselines, no "better average error = profitable betting" claims), (4) feature-schema version stored with the model and checked on load, (5) your sign-off. Average error improvements are never evidence of betting profit.

## 4. What stays uncertain
Whether production has actually been failing since the migration deploy (logs not inspected); whether the key was ever exposed; true roster accuracy beyond the 497 compared players; whether the four missing games are only timing; any claim about Codex's fixes (branch not seen); legal status of the data sources.
