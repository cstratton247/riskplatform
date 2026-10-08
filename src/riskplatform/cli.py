"""Command-line entry point: `riskplatform <command>`."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from riskplatform.config import load_config
from riskplatform.db import get_engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="riskplatform")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("init-calendar", help="load NYSE trading days into ref.trading_day")
    cal.add_argument("--start", type=date.fromisoformat)
    cal.add_argument("--end", type=date.fromisoformat)

    lp = sub.add_parser("load-prices", help="full-history price refresh with DQ checks")
    lp.add_argument("--tickers", nargs="+")
    lp.add_argument("--start", type=date.fromisoformat)
    lp.add_argument("--end", type=date.fromisoformat, help="exclusive; defaults to today")

    lm = sub.add_parser("load-macro", help="load FRED/ALFRED series into macro.observation_vintage")
    lm.add_argument("--series", nargs="+")

    sub.add_parser("build-features", help="compute features and write them to feat.feature_value")
    bl = sub.add_parser("build-labels", help="compute labels and write them to ml.label")
    bl.add_argument("--quantile", type=float, help="override target.threshold_quantile (e.g. 0.90)")

    wf = sub.add_parser("walk-forward", help="run the benchmark ladder over purged walk-forward folds")
    wf.add_argument("--no-db", action="store_true", help="print results without writing to the database")
    wf.add_argument("--quantile", type=float, help="override target.threshold_quantile (e.g. 0.90)")
    wf.add_argument("--ladder-only", action="store_true", help="skip tuned models (fast)")

    cp = sub.add_parser("compare", help="significance tests and fold-averaged summary from stored predictions")
    cp.add_argument("--quantile", type=float, help="override target.threshold_quantile (e.g. 0.90)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    if getattr(args, "quantile", None):
        cfg["target"]["threshold_quantile"] = args.quantile
    engine = get_engine()

    if args.command == "init-calendar":
        from riskplatform.ingest.calendar import load_calendar

        start = args.start or date.fromisoformat(cfg["data"]["start_date"])
        end = args.end or date.today()
        n = load_calendar(engine, start, end)
        print(f"Loaded {n} trading days ({start} to {end})")
        return 0

    if args.command == "load-macro":
        import os

        from riskplatform.ingest.macro import load_macro

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            print("FRED_API_KEY is not set in .env", file=sys.stderr)
            return 2
        ok = load_macro(engine, cfg, args.series, api_key)
        print("All series loaded." if ok else "Finished with failures; see ops.pipeline_run.")
        return 0 if ok else 1

    if args.command == "build-features":
        from riskplatform.features.compute import compute_features
        from riskplatform.features.store import (
            finish_run, load_feature_inputs, start_run, write_features,
        )

        run_id = start_run(engine, "build_features")
        try:
            feats = compute_features(load_feature_inputs(engine))
            n = write_features(engine, feats, cfg["target"]["asset"], run_id)
        except Exception as exc:
            finish_run(engine, run_id, 0, error=f"{type(exc).__name__}: {exc}")
            raise
        finish_run(engine, run_id, n)
        print(f"Wrote {n} feature values ({feats.shape[1]} features, {feats.shape[0]} days)")
        return 0

    if args.command == "build-labels":
        from riskplatform.features.store import build_and_write_labels, finish_run, start_run

        run_id = start_run(engine, "build_labels")
        try:
            n = build_and_write_labels(engine, cfg)
        except Exception as exc:
            finish_run(engine, run_id, 0, error=f"{type(exc).__name__}: {exc}")
            raise
        finish_run(engine, run_id, n)
        print(f"Wrote {n} label rows")
        return 0

    if args.command == "walk-forward":
        import pandas as pd

        from riskplatform.features.compute import BASE_FEATURES
        from riskplatform.features.store import finish_run, start_run
        from riskplatform.models.benchmarks import default_ladder
        from riskplatform.models.learners import full_ladder
        from riskplatform.validation.report import fold_mean_table
        from riskplatform.validation.store import frame_hash, load_modeling_frame, write_walk_forward
        from riskplatform.validation.walkforward import evaluate, make_folds, run_walk_forward

        v = cfg["validation"]
        frame = load_modeling_frame(engine, cfg)
        specs = default_ladder() if args.ladder_only else full_ladder()
        folds = make_folds(v["test_years"][0], v["test_years"][1])
        result = run_walk_forward(frame, specs, folds, list(BASE_FEATURES),
                                  v["modeling_start"], v["final_holdout_start"])
        metrics = evaluate(result.preds)

        pd.set_option("display.width", 160)
        print("\nPRIMARY: metrics averaged over folds (%d-%d), each year weighted equally:" % tuple(v["test_years"]))
        print(fold_mean_table(metrics).round(4).to_string())
        print(f"\nConfigurations tried (inner CV): {len(result.trials)} fits logged "
              f"across {result.trials['model'].nunique()} tuned models")

        if not args.no_db:
            run_id = start_run(engine, "walk_forward")
            try:
                n = write_walk_forward(engine, cfg, specs, result.preds, metrics, result.fold_info,
                                       result.trials, frame_hash(frame), run_id)
            except Exception as exc:
                finish_run(engine, run_id, 0, error=f"{type(exc).__name__}: {exc}")
                raise
            finish_run(engine, run_id, n)
            print(f"Saved {n} predictions, metrics and trials. Next: riskplatform compare")
        return 0

    if args.command == "compare":
        import pandas as pd

        from riskplatform.validation.report import fold_mean_table, pairwise_tests
        from riskplatform.validation.store import load_predictions
        from riskplatform.validation.walkforward import evaluate

        preds = load_predictions(engine, cfg)
        pd.set_option("display.width", 200)
        print("\nPRIMARY: fold-averaged metrics:")
        print(fold_mean_table(evaluate(preds)).round(4).to_string())
        print("\nPre-declared comparisons (ML vs HAR / VIX benchmarks); Holm-adjusted p-values.")
        print("pr_auc_diff = fold-mean PR-AUC difference (positive favors `model`); 95% block-bootstrap CI.")
        print("logloss_diff < 0 favors `model`; DM test with HAC variance.\n")
        tests = pairwise_tests(preds, cfg["stats"])
        print(tests.round(4).to_string(index=False) if not tests.empty else "(no ML models found)")
        return 0

    from riskplatform.ingest.prices import load_prices
    from riskplatform.ingest.yahoo import YahooSource

    start = args.start or date.fromisoformat(cfg["data"]["start_date"])
    end = args.end or date.today()
    ok = load_prices(engine, YahooSource(), cfg, args.tickers, start, end)
    print("All tickers loaded." if ok else "Finished with failures; see ops.pipeline_run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
