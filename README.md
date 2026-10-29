# NBA Top-3 Dashboard

Tracks each team's top-3 producers, projects points/rebounds/assists, double- and triple-double odds,
and prices 2-3 leg parlays (with same-game correlation) against bookmaker odds.

## Run
    pip install -r requirements.txt
    python ingest.py 2025-10-21 2026-06-30   # one-off backfill (last season, ~10 min)
    python refresh.py --train                # pull new games, retrain, project upcoming games
    APP_ACCESS_CODE=choose-a-code uvicorn app:app --port 8000

Run `python refresh.py` daily (cron / scheduler) and `--train` weekly.

## Data sources
- Box scores, schedule, injuries: ESPN public JSON (stats.nba.com blocks cloud IPs).
- Starting lineups / may-not-play: Rotowire lineups page.
- Odds: set `ODDS_API_KEY` (The Odds API, `au` region, bookmaker `sportsbet`); see `odds.py`. Without it, enter parlay odds by hand.

## Sharing
Friends open the link, enter a name and the shared access code (`APP_ACCESS_CODE`), then add picks, comments, votes and bets.
Set `NBA_DB` to a path on a persistent disk when hosting.

## Deploy to Fly.io
    fly launch --no-deploy --copy-config            # app name must be globally unique; edit fly.toml if taken
    fly volumes create nba_data --size 1 --region syd
    fly secrets set APP_ACCESS_CODE=NBA2026 ODDS_API_KEY=<your key>
    fly deploy
The app refreshes data every 4 hours and retrains weekly (AUTO_REFRESH=1). The access code lives only in Fly secrets, not in git.
