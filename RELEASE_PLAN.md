# Release, backup and recovery plan  (for Mark's approval — nothing here has been done)

## What would go live
Everything on `review/13-daylight-saving-checks` (13 small steps, each its own copy of the code). The website is currently the version from `main` commit `1df220a`.
Not included: a fix for points chances being too confident (points VALUE and "safe" labels are switched off instead).

## Before anything goes live (needs Codex's review first)
1. Codex reviews the 13 steps and reports in plain words. Claude fixes anything real it finds.
2. Mark says **yes** to the release in writing.

## Backups (do these first, on the day)
1. **Saved copy of the code that works today.** Tag the current live version (`1df220a`) in GitHub as `last-good-before-release`. Costs nothing and takes a minute.
2. **Saved copy of the data.** The app already keeps its own daily copy of the database inside the Fly storage volume (`/data/backups`, last 7 days) once the reliability fix is live; today's live version does not have it yet, so before releasing:
   - In Fly: Dashboard → *nba-dashboard* → **Volumes** → pick the volume → **Snapshots** → create/confirm a snapshot made on the release day. (Fly normally takes daily snapshots; I could not check your account, so please look.)
   - Claude also downloads a copy of the live database through the app's backup step after release and keeps it OUTSIDE the volume (a file in a private place Mark chooses). **Not built yet**: today the backups sit on the same volume as the data. If you want an off-site copy, say so and I will build it as a separate step.
3. **Saved copy of today's prediction model** (the old `models.pkl` stays inside the volume as `models.pkl`; the new one replaces it only if it passes the checks, and a rejected one is kept as `models.pkl.rejected`).

## Release steps (Claude does these; Mark approves each stage)
1. Put the 13 steps on the website in one go (one publish).
2. Open `https://nba-dashboard.fly.dev/api/health`. For the first 10–20 minutes it will say **problem** because the model is being retrained (expected). The Games tab shows "predictions are being updated".
3. After about 15 minutes it should say **ok**. Rehearsal (copy of today's data, old model): see the figures below.
4. Check the Games, Best, Value, Lineups and Live tabs, a saved pick and the bet slip on a phone.
5. For a week: look at the health page each morning (Melbourne time).

## What you will notice
- Fewer VALUE labels (prices older than 8 hours get none; points markets get none).
- Notes such as "prices are N hours old" and "predictions are being updated".
- Parlays that can be refunded on a tie show "Refunded" rules instead of a single fair price.

## If something goes wrong (recovery)
- **Website broken or wrong numbers:** in GitHub → Actions, re-run the deploy of the last good version (`1df220a`) — the website goes back to today's behaviour within about a minute. The data in the volume is kept (the new version only adds columns and tables; it never deletes history).
- **Data damaged:** restore the volume snapshot from the release day (Fly → Volumes → Snapshots → restore), or copy the newest file from `/data/backups` back to `nba.db`. Claude does this with Mark's approval.
- **Model rejected or missing:** the site keeps showing the previous projections and says predictions are being updated; bet building stays paused until a model passes. Nothing is trained during a page request.
- **Fly or GitHub outage during release:** nothing changes on the website until the publish succeeds (this happened earlier today; retry later).

## Known risks I cannot remove
- Training needs memory on a 1 GB machine; the rehearsal figures below are from this environment, not from Fly.
- Prices are fetched at most every 10 hours (credit limits), so VALUE will often be switched off for old prices.
- Points chances remain too confident; first weeks of a season are worst.
- No profit is proven. The Track record tab needs weeks of saved prices and results before it means anything.
