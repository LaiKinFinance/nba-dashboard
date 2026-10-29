# Audit branch integration and release hold

29 October 2026. Review branch: `codex-audit`. Base: `hotfix/ingest-insert-and-key-redaction` at `58cfd911bb21516e8aa5bb5bfcba1ccf6f4b2c2d`, which contains main through `7833bde`.

The audit commits were replayed onto that hotfix base, split where necessary and classified below. Claude's newer track-record/referee/lineup work on the base was preserved. The separate `ui/clear-slip-and-header` branch was not merged. This is a review branch, not a release candidate for wholesale deployment.

Read the original evidence in [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) and Claude's [independent review](https://github.com/LaiKinFinance/nba-dashboard/blob/audit-review/AUDIT_REVIEW.md). That review assessed findings and the hotfix; it explicitly did not review the then-unpublished audit code. Historical report line references and numerical results refer to the original audit snapshot, not this integration.

## Hotfix decisions

| Area | Kept | Reason |
|---|---|---|
| Ingestion | Claude's `e7474fe` unchanged | Its named-column `INSERT ... ON CONFLICT DO UPDATE` supports the migrated 20-column table and preserves extra market/foul/stat columns. My original named-column `INSERT OR REPLACE` fixed the value count but could erase extra fields during a conflicting insert. My production ingestion change was dropped; its end-to-end regression test was retained and expanded. |
| Key redaction | Claude's `58cfd91` unchanged in production | Both implementations suppressed Requests URL exceptions. Claude's additionally scrubs error text before storage/return and when reading previously stored status. My duplicate `_get` replacement was dropped. My malformed-odds parser and regression tests were retained because they address separate behavior. |

`git diff 58cfd91 -- ingest.py` is empty. The production difference in `odds.py` is limited to count/yes-no market parsing and finite-price validation; the hotfix request/error/status code remains intact. The hotfix redaction test now uses pytest's environment restoration and an in-memory status database so a synthetic test key does not remain in the test process environment.

## Model compatibility safeguard

- Feature schema **2** identifies the changed chronological absence bookkeeping and completed-history-only per-event placeholder logic. Changing feature meanings/history requires another version increment even when column names stay the same.
- The training writer saves `feature_schema_version` and the ordered `feature_columns` for every stat/classifier in the model bundle.
- Every production model load checks the exact integer version and ordered feature-column manifest. Missing, old, future, malformed or column-mismatched schemas are rejected. The old bundled model is deliberately **not** relabelled or rewritten.
- Missing artifacts no longer silently trigger training during a load. Prediction APIs return HTTP 503 with an explanation when the model is incompatible. Automatic refresh and the API refresh command check compatibility before ingestion/training writes; the background loop stops on a schema failure rather than silently replacing the model.
- The safeguard proves compatibility, **not** repaired data, calibrated probabilities, or release approval. A matching schema alone is never evidence that a model is ready to ship.

The existing bundled model is unversioned and is therefore intentionally rejected by this branch. The data audit now reports this as a failure: **21 passes, 0 warnings, 1 expected compatibility failure**. Do not deploy the entire branch against the existing artifact. Do not bypass the guard by changing metadata on an old model. No real-data training, reprojection, data repair or historical deletion was performed in this integration.

## Regression tests and final checks

The original 44-test suite included these two specific regressions:

| Original test | Coverage now |
|---|---|
| `tests/test_ingest.py::test_ingest_inserts_game_after_market_columns_added` | Calls the actual `ingest.run()` path with mocked provider data and verifies the saved game/market total. Now parametrized for both the 14-column and 20-column migrated schemas. |
| `tests/test_odds.py::test_provider_failure_never_returns_request_url` | Simulates a Requests HTTP 401 carrying a synthetic marker; checks the returned exception contains the status but does not contain the request marker. It now exercises Claude's retained redaction code. |

Inherited hotfix regressions also remain: `test_insert_new_game_on_migrated_schema`, `test_reingest_keeps_market_lines`, `test_get_never_leaks_key`, and `test_scrub_and_status_clean_old_leaks`. The latter checks previously stored status, beyond my original redaction test's scope.

Final verification on the combined code:

- `python -m pytest -q`: **63 passed**, no skips, one existing Starlette/httpx deprecation warning. This includes the original coverage, four inherited hotfix cases, the extra 20-column variant, model-schema/refresh regressions, and one Python wrapper that runs four JavaScript progress checks through Node.js.
- The four JavaScript checks cover component sums for PRA/P+R/P+A, under/over selection, missing-data handling and binary no legs. They do not change the three-category DD/TD definition; the official five-category correction remains Phase 1 work.
- The cross-game parlay example was strengthened to put the high-mean game first, so the original bug would clearly violate the smaller leg probability instead of escaping a loose simulation tolerance.
- `git diff --check` passed. Hotfix ancestry was verified.
- Bundled `data/nba.db` and `data/models.pkl` match the hotfix and fetched main exactly. **No audit commit changes either file.** Newer data already inherited from main is not an audit data repair.
- Tests use synthetic fixtures or temporary database/model copies. They make no paid odds calls, use no real credentials and do not deploy.

Reproduce with Python 3.12: `python -m pip install -r requirements-test.txt`, then `python -m pytest -q`. Install Node.js to run the JavaScript checks; their Python wrapper explicitly skips if Node is unavailable. Node was available and those checks ran in this verification. The historical timing tool requires a compatible validated model and will now correctly reject the bundled legacy artifact instead of bypassing the guard.

## Commit classification

Every new audit commit has one of the requested labels in its subject. The two inherited hotfix commits are model-independent but retain their original authors/titles.

| Commit | Label | Scope |
|---|---|---|
| `a90ae3a` | model-independent | Baseline probability/copula tests. |
| `f0cf902` | model-independent | Actual ingestion-path regression test. |
| `adc82e7` | model-independent | Parlay draws keyed by game, player and component. |
| `3ff4fa9` | model-independent | Player under outcomes and strict count under tails. |
| `876b114` | model-independent | Preferred/best-price and redaction tests. |
| `d36172b` | model-independent | Malformed count odds and binary yes/no parsing. |
| `87bd0fb` | model-independent | Correct bookmaker on each displayed outcome. |
| `ddd86ad` | model-independent | Stored-pick XSS escaping and saved-input validation. |
| `e5d4385` | model-independent | Market-specific required fields/sides. |
| `108a007` | model-independent | Auth and SQL-search tests. |
| `4e6080a` | model-independent | Name normalization/full-name matching. |
| `986adf3` | model-independent | Eastern daylight saving selection. |
| `665a9fe` | model-independent | Keyboard/expanded-state market controls. |
| `fa93b00` | model-independent | Slip text colour. |
| `173ffac` | model-independent | Hotfix redaction test isolation. |
| `1462865` | model-independent | Explicit 14/20-column ingestion cases. |
| `5af6e56` | model-independent | Live combined-stat/under display and JavaScript checks. |
| `47c885c` | changes model inputs | Absence history/roster bookkeeping; held for Phase 1. |
| `75c8ec9` | changes model inputs | Per-event placeholder features; held for Phase 1. |
| `8128701` | changes model inputs | Schema-2 compatibility contract and training/load checks; coordinated with those definitions. |
| `ffc5ed0` | model-independent | Safe API/background handling of incompatible model errors; depends on the compatibility guard. |
| `f568826` | model-independent | Read-only historical audit tools. |
| `d3e37be` | model-independent | Historical findings report and integration pointer. |
| `6824b92` | model-independent | Data audit uses the guarded model loader; depends on the compatibility guard. |
| `7344b3d` | model-independent | Stronger cross-game parlay regression example. |

The final integration-notes commit is also labelled model-independent. “Model-independent” means it does not redefine training features, not that every such commit can be cherry-picked without its dependencies. In particular, the API/data-audit guard callers need the schema module and are part of the coordinated Phase 1 artifact transition. The model-independent product changes through `5af6e56` can be reviewed separately from the feature-definition commits. Dates, roster reconciliation, official five-category DD/TD and dispersion changes are **not implemented** by this integration.

## Review and release rule

Only `codex-audit` is to be pushed. No main push, pull-request merge, deployment, historical data deletion or real model regeneration is authorized by this integration.

Phase 1 must first repair/reconcile data and dates, resolve roster/availability definitions, then explicitly train offline and validate against suitable later games and simple baselines. Publish matching versioned artifacts only after review/sign-off. Further date, roster or five-category changes must increment the schema again. The whole audit branch stays on hold until that process is complete; individual model-independent fixes can be reviewed separately with their dependencies.
