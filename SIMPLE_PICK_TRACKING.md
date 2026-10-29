# Simple pick tracking

Mark asked to remove money tracking and focus on prediction quality.

Prepared on `codex/simple-pick-tracking`, starting from `review/29-simplify-no-refund-maths`.

## What changed

- My bets is now My picks.
- Saving a pick needs no stake, price or returned amount.
- Money entry, profit/loss totals, payout routes, bankroll and stake advice, expected profit calculations and hypothetical returns are removed.
- Results show won, lost or waiting. An unplayed player or exact tie is not counted as a win or loss. Combined picks wait for their complete results.
- The results page works without loading a prediction model.
- Old records and retired money fields stay stored. They are not used or shown. No history is deleted.
- Existing points warnings, stale-model checks, started-game checks and VALUE restrictions remain.

No prediction features, training, probability calculations, game dates or rosters were changed. No retraining is needed for this removal.
The bundled `data/nba.db` and `data/models.pkl` match main.

## Checks

The suite covers saving without money, actual won/lost counts, excluding unscorable picks, manual corrections, ownership, keeping old records, and the page's real save/render functions. The page's full JavaScript also passes its syntax check.

Full run: **305 passed**, including the repaired-data check using `REPAIRED_DATA_DIR` on an existing private repaired copy. One existing library warning remains. No check was skipped.

Nothing has been deployed. This change does not claim to fix unrelated backup recovery problems or approve the full release.
