# How to check the app is healthy (no coding needed)

## 1. Is the data refreshing? (after the `reliability` branch is deployed)
Open this address in any browser. No password is needed and it contains no secrets:

    https://nba-dashboard.fly.dev/api/health

You will see plain text. What matters:
- `data_through` should be today's or yesterday's date during the season.
- `stages`: each step (`ingest`, `project`, `odds`, ...) shows `ok_at` (last time it worked) and, if it failed, `fail_at` and `error`.
  Anything with an `error` is a problem. Copy the whole page and send it to Claude.
- `model_current` should say `true` once the retrained model is in place.

## 2. Looking at the Fly logs yourself (only if asked)
1. Go to https://fly.io/dashboard and sign in.
2. Click the app **nba-dashboard**.
3. In the left menu click **Monitoring** (or **Live Logs**).
4. Press Ctrl+F and search for `20 columns`. If it appears, that is the old ingest bug. After the hotfix it should not.

## 3. Replacing (rotating) the Odds API key
Only needed if you think the key was ever shown in an error message.
1. Sign in at https://the-odds-api.com, open your account, and request a new API key. The old one stops working.
2. In the Fly dashboard open **nba-dashboard**, then **Secrets**, and replace `ODDS_API_KEY` with the new key. The app restarts by itself.
3. Never paste the key into chat or a file.
