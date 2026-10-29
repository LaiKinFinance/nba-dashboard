"""Optional, bounded ESPN cross-check; no database writes or paid API calls."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import sqlite3
import sys
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ingest


def check(game):
    gid, stored_date = game
    js = requests.get(ingest.SUM, params={'event': gid}, timeout=30).json()
    h = js['header']
    c = h['competitions'][0]
    teams = [x['team']['abbreviation'] for x in c['competitors']]
    played = []
    zero = []
    dnp = 0
    for team in js.get('boxscore', {}).get('players', []):
        for group in team['statistics']:
            for a in group['athletes']:
                if a.get('didNotPlay') or not a.get('stats'):
                    dnp += 1
                    continue
                s = dict(zip(group['keys'], a['stats']))
                row = dict(pid=a['athlete'].get('id'), name=a['athlete'].get('displayName'), team=team['team']['abbreviation'], minutes=s.get('minutes'), points=s.get('points'))
                played.append(row)
                if s.get('minutes') == '0': zero.append(row)
    with sqlite3.connect('file:data/nba.db?mode=ro', uri=True) as con:
        stored = {x[0] for x in con.execute('SELECT player_id FROM player_games WHERE game_id=?', (gid,))}
    return dict(game_id=gid, stored_date=stored_date, start=c['date'], teams=teams,
                season_type=h['season']['type'], neutral=c.get('neutralSite'), notes=c.get('notes'),
                missing_played=[x for x in played if x['pid'] not in stored], zero_minutes=zero, dnp=dnp)


if __name__ == '__main__':
    with sqlite3.connect('file:data/nba.db?mode=ro', uri=True) as con:
        games = con.execute("SELECT game_id,date FROM games WHERE (home='NY' AND away='SA') OR (home='SA' AND away='NY') OR game_id IN ('401810560','401810633','401810760') OR date=(SELECT max(date) FROM games)").fetchall()
    with ThreadPoolExecutor(3) as pool:
        for row in pool.map(check, games):
            print(json.dumps(row))
    with sqlite3.connect('file:data/nba.db?mode=ro', uri=True) as con:
        ids = {r[0] for r in con.execute('SELECT game_id FROM games')}
    for date in ['20251216', '20260131', '20261027', '20261028']:
        js = requests.get(ingest.SB, params={'dates': date, 'limit': 100}, timeout=30).json()
        events = [e for e in js['events'] if e['status']['type']['name'] == 'STATUS_FINAL']
        real = [e for e in events if all(c['team']['abbreviation'] in ingest.REAL_TEAMS for c in e['competitions'][0]['competitors'])]
        print(json.dumps(dict(scoreboard_date=date, finished=len(real), missing_ids=[e['id'] for e in real if e['id'] not in ids],
                              different_utc_day=[dict(game_id=e['id'], utc=e['date'], eastern=dt.datetime.fromisoformat(e['date'].replace('Z', '+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()) for e in real if e['date'][:10].replace('-', '') != date])))
