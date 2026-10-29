# Release, backup and going-back plan (for Mark's approval — NOTHING here has been done yet)

Written in plain words. The detailed evidence is in `REVIEW_REPORT.md`.

**Mark's latest decision:** money tracking is removed. The replacement is prepared on `codex/simple-pick-tracking`, based on `review/29-simplify-no-refund-maths`; see `SIMPLE_PICK_TRACKING.md`. The older release target and money-entry instructions below are superseded. Publish only the final combined copy that Codex reviews and Mark approves. This document still needs the final backup/recovery steps agreed before it is used.

## What would go live
The whole set on `review/27-final-candidate` (the 17 earlier steps plus the second round of fixes: stale results, no points advice, started games, per-version checks, refund prices, "under" wording, points study, missing games, backups). Today's website is the version on `main` (`1df220a`).

The refund calculations from the earlier step were removed again (Mark's decision to simplify): see `REVIEW_REPORT.md` section 14.

Not included: any fix that makes points chances accurate. Points VALUE labels, "safe" labels and all profit/stake advice on slips with points stay switched off.

## The order (each stage waits for Mark's "yes")
**Stage 0 – Reviews and approval.** Codex reviews the final set. Claude fixes anything real. Mark says yes in writing.

**Stage 1 – Mark creates the backup code (2 minutes).**
1. Go to the GitHub page of the project → *Settings* → *Secrets and variables* → *Actions* → *New repository secret*.
2. Name: `BACKUP_CODE`. Value: a long phrase only you know (for example four unrelated words and a number). Do not send it to Claude or Codex or put it in chat.
3. Save. (The website's publish step copies it into the server by itself, the same way it does for the access code.)

**Stage 2 – Publish the "backup first" version.** This is a very small version (`review/28-backup-first-release`): today's website plus ONLY a backup download page. Nothing else changes. It is the safe place to go back to.

**Stage 3 – Take the backups (before anything else changes).**
1. Open `https://nba-dashboard.fly.dev/backup.html`, type your backup code, press **Download data backup**, then **Download prediction file**. Two files are saved on your phone or computer.
2. Put a copy of both in your own Google Drive (this is the copy that lives OUTSIDE the live server).
3. Claude checks the files with the checking tool (`tools/backup_restore.py verify`) and tells you "intact" or what is wrong. Nothing continues until this says intact.
4. In the Fly dashboard: *nba-dashboard* → *Volumes* → *Snapshots*. Check that a snapshot from today exists. (Fly normally makes them daily; Claude cannot see your Fly account, so please look and tell Claude.)
5. Claude saves two markers (tags) in GitHub: the code that is live today (`1df220a`) and the "backup first" version. Code is public on GitHub; **data is never put on GitHub.**

**Stage 4 – Publish the full set** (`review/27-final-candidate`), in one go.

**Stage 5 – Watch it.** Open `https://nba-dashboard.fly.dev/api/health`. For the first 5–20 minutes it says **problem**: that is the model being retrained (expected; the Games tab shows "predictions are being updated"). When it says **ok**, check: Best picks, Games, Value, Lineups, Live, a saved pick, the bet slip and My bets on your phone. Look at the health page each morning (Melbourne time) for a week.

## What you will notice after the release
- Best picks is the first page. Pages load faster.
- Fewer VALUE labels (prices older than 8 hours get none; no VALUE for points).
- No profit/stake advice on any slip that includes points; the points warning stays.
- My picks shows won / lost when games finish. A player who did not play or a result that finished level stays out of that count. No money entry is needed.
- Whole-number lines (such as over 20 or handicap -4) get no chances or advice, only a short explanation. Half-point lines and "20 or more" bets work as before.
- Notes such as "prices are N hours old", "Frozen", "predictions are being updated".

## Going back (rehearsed on copies)
**A. The website misbehaves (most likely case) — put the old code back.** GitHub → *Actions* → *Deploy to Fly.io* → open the successful run for the "backup first" version → *Re-run all jobs*. Takes about a minute. **Your data stays as it is** (the new version only adds columns and tables; it never deletes history).
  *Rehearsed:* the previous code was run against data already upgraded by the new code. Every page answered, it trained its own model again in 37 seconds, and bets and picks were still readable.

**B. The prediction file.** The old code makes its own prediction file again within minutes, so restoring the old file is normally not needed. Your saved copy exists anyway (Stage 3).

**C. The data is damaged (unlikely).** Claude runs `tools/backup_restore.py restore` on the Stage 3 backup. It writes a NEW database file = the backup + every newer bet, pick, comment, vote, published prediction and price record found in the damaged file. It never overwrites or deletes the damaged file.
  *Rehearsed:* on a real-sized copy, after deliberately zeroing all player points. The recovered file had the points back, the newer bet, pick and comment kept (comment still attached to its pick), and passed its integrity check; the damaged file was untouched.
  *What is not possible from here:* putting the recovered file onto the live Fly disk needs Fly's command-line tool or Fly's snapshot restore, which neither Claude nor Codex can use in this environment. If that day comes, Claude gives you the exact short steps to follow with Fly, and the recovered file contains your newest bets.

## Things this plan does not cover
- A second automatic off-site copy every day. Today's off-site copy is the one you download by hand at Stage 3 (and any time later from the same page). If you want an automatic daily copy sent somewhere private, say so and it becomes a separate step. (GitHub is not suitable: the project is public.)
- Fly outages. Earlier today the publish step failed several times because Fly's own servers returned errors; nothing changed on the website until it succeeded. If that happens, wait and retry.

## What was tested (copies only, nothing live)
- Full test suite on the final set: **271 passed, 1 skipped**. The skipped check needs a repaired copy of the data; it was run separately on one and **passed**.
- Full update rehearsal from today's bundled data and old model with the final code: the whole update took **189 seconds** here and every step finished "ok" (repair, weekly check of every finished game, download of 9 missing games, retraining, projections, grading, marking of bets, checks, backup). The health page ended at **ok**, and listed the four finished playoff games that ESPN has no box score for. (A first attempt looked wrong only because I switched code versions while it was running; the repeat was run from a frozen copy.)
- Backup download needs the right code, is switched off when no code is set, and is limited to one download per 30 seconds (tests). The backup page never stores the code.
- Data recovery and code rollback rehearsals: above.

## Known risks I cannot remove
- Training used about 284 MB at peak here (Fly's machine has 1 GB), but Fly's shared processor may be several times slower than my test machine.
- Prices are fetched at most every 10 hours (Odds API credits), so VALUE will often be switched off for old prices.
- Points chances remain too confident for bets chosen because the model likes them (about 4.5 points even after correction).
- No profit is proven. The Track record needs weeks of saved prices and results before it means anything.
- Four finished playoff games have no box score on ESPN (listed in `REVIEW_REPORT.md`); they cannot be used for training.
