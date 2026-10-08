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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    engine = get_engine()

    if args.command == "init-calendar":
        from riskplatform.ingest.calendar import load_calendar

        start = args.start or date.fromisoformat(cfg["data"]["start_date"])
        end = args.end or date.today()
        n = load_calendar(engine, start, end)
        print(f"Loaded {n} trading days ({start} to {end})")
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
