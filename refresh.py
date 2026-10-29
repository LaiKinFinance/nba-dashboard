"""Daily job: pull new box scores, (optionally) retrain, re-project upcoming games.  python refresh.py [--train]"""
import sys, datetime as dt
import ingest, predict
t = dt.date.today()
ingest.run(t - dt.timedelta(days=4), t)
if "--train" in sys.argv: predict.train()
p = predict.project(); predict.save(p); print(len(p), "projections")
