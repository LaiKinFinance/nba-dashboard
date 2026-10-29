# NBA Top-3 Dashboard

Tracks each team's top-3 producers, projects points/rebounds/assists, double- and triple-double odds,
and estimates how often combined picks will be right. Bookmaker prices provide context.

This is a prediction app. It tracks won/lost picks, not stakes, payments, refunds or profit.
The priorities are correct dates and rosters, honest chances, careful VALUE labels,
a fast website and an honest record of prediction results.

## Run
    pip install -r requirements.txt
    python ingest.py 2025-10-21 2026-06-30   # one-off backfill (last season, ~10 min)
    python refresh.py --train                # pull new games, retrain, project upcoming games
    APP_ACCESS_CODE=choose-a-code uvicorn app:app --port 8000

Run `python refresh.py` daily (cron / scheduler) and `--train` weekly.

## Data sources
- Box scores, schedule, injuries: ESPN public JSON (stats.nba.com blocks cloud IPs).
- Starting lineups / may-not-play: Rotowire lineups page.
- Odds: set `ODDS_API_KEY` (The Odds API, `au` region, bookmaker `sportsbet`); see `odds.py`. Picks can be saved without any price or money entry.

## Sharing
Friends open the link, enter a name and the shared access code (`APP_ACCESS_CODE`), then save picks, comments and votes.
Set `NBA_DB` to a path on a persistent disk when hosting.

## Deploy to Fly.io
    fly launch --no-deploy --copy-config            # app name must be globally unique; edit fly.toml if taken
    fly volumes create nba_data --size 1 --region syd
    fly secrets set APP_ACCESS_CODE=NBA2026 ODDS_API_KEY=<your key>
    fly deploy
The app refreshes data every 4 hours and retrains weekly (AUTO_REFRESH=1). The access code lives only in Fly secrets, not in git.

## Odds API budget
Each game's player-prop odds cost 5 credits (5 markets x 1 region) and are fetched once per game (about 10h cooldown), only for games starting within 30h.
The free plan's 500 credits/month covers roughly 100 games. Tune with `ODDS_MARKETS`, `ODDS_GAMES_PER_DAY`, `ODDS_MIN_CREDITS`.

## Match betting and alternate lines
- Team model (`team_model.py`): margin-Elo for the winner/handicap, scoring-form formula for the total. Check results with `python team_model.py`; the app's Track record shows published pick results.
- Winner/handicap/total prices come from ONE Odds API call (3 credits) at most every `ODDS_LINES_HOURS` (default 8).
- Alternate player lines are off by default because they cost extra credits per game. To enable, set `ODDS_MARKETS` to
  `player_points,player_rebounds,player_assists,player_double_double,player_triple_double,player_points_alternate,player_rebounds_alternate,player_assists_alternate`.
  Lines with no bookmaker price still show the model's fair odds.

## Audit
`python tools/audit_data.py` checks data integrity; see `AUDIT.md` for the architecture map and review checklist.
