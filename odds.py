"""Bookmaker odds: The Odds API (au region, Sportsbet) when ODDS_API_KEY is set; otherwise manual entry through the app."""
import os, requests

BASE = "https://api.the-odds-api.com/v4"
MARKETS = "player_points,player_rebounds,player_assists,player_double_double,player_triple_double"

def available():
    return bool(os.environ.get("ODDS_API_KEY"))

def events():
    r = requests.get(f"{BASE}/sports/basketball_nba/events", params={"apiKey": os.environ["ODDS_API_KEY"]}, timeout=30)
    r.raise_for_status(); return r.json(), r.headers.get("x-requests-remaining")

def player_props(event_id, bookmakers="sportsbet"):
    r = requests.get(f"{BASE}/sports/basketball_nba/events/{event_id}/odds",
                     params={"apiKey": os.environ["ODDS_API_KEY"], "regions": "au", "bookmakers": bookmakers,
                             "markets": MARKETS, "oddsFormat": "decimal"}, timeout=30)
    r.raise_for_status()
    out = []
    for b in r.json().get("bookmakers", []):
        for m in b["markets"]:
            for o in m["outcomes"]:
                out.append(dict(book=b["key"], market=m["key"], player=o.get("description"), side=o["name"].lower(),
                                line=o.get("point"), odds=o["price"]))
    return out, r.headers.get("x-requests-remaining")
