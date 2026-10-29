"""Model schema + validation gate. Run: python3 tests/test_model_gate.py"""
import os, sys, pickle, tempfile
d = tempfile.mkdtemp(); os.environ["DATA_DIR"] = d; sys.path.insert(0, ".")
import predict, features
# an unversioned (old) model must be refused for projections, but readable for display
pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="2026-01-01", report={}), open(predict.MODELS, "wb"))
assert predict.model_stale()
try: predict.load_models(); raise SystemExit("old model was accepted")
except predict.StaleModel: pass
assert predict.load_models(strict=False)["trained_through"] == "2026-01-01"
pickle.dump(dict(models={}, alphas={}, corr={}, trained_through="x", report={}, schema=features.FEATURE_SCHEMA), open(predict.MODELS, "wb"))
assert not predict.model_stale() and predict.load_models()["trained_through"] == "x"
# gate: a model worse than the plain 10-game average is rejected
good = {"stats": {t: dict(mae_model=4.0, mae_10game_avg=4.1) for t in predict.TARGETS}}
bad = {"stats": {**good["stats"], "pts": dict(mae_model=6.0, mae_10game_avg=4.1)}}
assert predict.validate(good) == (True, []) and predict.validate(bad) == (False, ["pts"])
print("model gate tests passed")
