"""Upcoming games (ESPN), starting lineups + may-not-play lists (Rotowire), injuries (ESPN)."""
import datetime as dt, re
import requests
from bs4 import BeautifulSoup

H = {"User-Agent": "Mozilla/5.0"}
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
INJ = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
RW = "https://www.rotowire.com/basketball/nba-lineups.php"
TEAM_NICK = {"ATL":"Hawks","BOS":"Celtics","BKN":"Nets","CHA":"Hornets","CHI":"Bulls","CLE":"Cavaliers","DAL":"Mavericks","DEN":"Nuggets","DET":"Pistons","GS":"Warriors","HOU":"Rockets","IND":"Pacers","LAC":"Clippers","LAL":"Lakers","MEM":"Grizzlies","MIA":"Heat","MIL":"Bucks","MIN":"Timberwolves","NO":"Pelicans","NY":"Knicks","OKC":"Thunder","ORL":"Magic","PHI":"76ers","PHX":"Suns","POR":"Trail Blazers","SAC":"Kings","SA":"Spurs","TOR":"Raptors","UTAH":"Jazz","WSH":"Wizards"}
ALIAS = {"NYK": "NY", "SAS": "SA", "GSW": "GS", "NOP": "NO", "WAS": "WSH", "UTA": "UTAH", "PHO": "PHX"}  # rotowire -> espn

def upcoming(days=2):
    """Scheduled (not started) games for the next `days` calendar days (US Eastern dates, as ESPN keys them)."""
    out, today = [], dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).date()
    for i in range(days):
        d = today + dt.timedelta(days=i)
        js = requests.get(SB, params={"dates": d.strftime("%Y%m%d")}, headers=H, timeout=30).json()
        for e in js["events"]:
            c = e["competitions"][0]["competitors"]
            h = next(x for x in c if x["homeAway"] == "home"); a = next(x for x in c if x["homeAway"] == "away")
            out.append(dict(game_id=e["id"], start=e["date"], status=e["status"]["type"]["name"], date=e["date"][:10],
                            home=h["team"]["abbreviation"], away=a["team"]["abbreviation"],
                            home_record=(h.get("records") or [{}])[0].get("summary"), away_record=(a.get("records") or [{}])[0].get("summary")))
    seen = set(); res = []
    for g in out:
        if g["game_id"] not in seen: seen.add(g["game_id"]); res.append(g)
    return res

def injuries():
    js = requests.get(INJ, headers=H, timeout=30).json()
    res = {}
    for t in js["injuries"]:
        for i in t["injuries"]:
            res[i["athlete"]["displayName"]] = dict(status=i["status"], team=i["athlete"]["team"]["abbreviation"],
                                                    detail=i.get("shortComment", "")[:200])
    return res

def lineups():
    """{espn_team_abbr: {status, starters:[names], out:[(name,status)]}} from Rotowire (names are abbreviated)."""
    s = BeautifulSoup(requests.get(RW, headers=H, timeout=30).text, "lxml")
    res = {}
    for b in s.select("div.lineup.is-nba"):
        abbrs = [ALIAS.get(a.get_text(strip=True), a.get_text(strip=True)) for a in b.select(".lineup__abbr")]
        for ab, ul in zip(abbrs, b.select("ul.lineup__list")):
            st = ul.select_one(".lineup__status")
            starters, out = [], []
            for li in ul.select("li.lineup__player"):
                a = li.select_one("a"); name = (a.get("title") or a.get_text(" ", strip=True)) if a else li.get_text(" ", strip=True)
                inj = li.select_one(".lineup__inj")
                if "has-injury-status" in li.get("class", []):
                    out.append((name, inj.get_text(strip=True) if inj else "Out"))
                else:
                    starters.append(name)
            res[ab] = dict(status=(st.get_text(strip=True) if st else ""), starters=starters, out=out)
    return res

def norm(n):
    return re.sub(r"[^a-z ]", "", n.lower().replace(".", " ")).split()

def name_match(full, abbreviated):
    """'Desmond Bane' vs 'D. Bane' / 'Desmond Bane' (last name must match, first initial must match)."""
    f, a = norm(full), norm(abbreviated)
    if not f or not a: return False
    return f[-1] == a[-1] and f[0][0] == a[0][0]

if __name__ == "__main__":
    for g in upcoming(): print(g["date"], g["away"], "@", g["home"], g["status"])
    L = lineups(); print(len(L), sorted(L))
    print(L.get("ORL"))
