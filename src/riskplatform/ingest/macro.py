"""Macro loading pipeline: fetch -> DQ -> replace the series' vintage rows.

Each series is replaced wholesale in one transaction (delete + insert), which is
idempotent and keeps the exclusion constraint on overlapping vintages satisfied,
since ALFRED re-issues the full, consistent vintage history on every request.
"""
from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import Engine, text

from .fred import apply_rule_lag, fetch_observations, macro_checks
from .prices import _config_hash, _git_sha, _store_dq

log = logging.getLogger(__name__)

INSERT_SQL = text(
    "insert into macro.observation_vintage "
    "(series_id, observation_date, valid_from, valid_to, value) "
    "values (:series_id, :observation_date, :valid_from, :valid_to, :value)"
)


def load_macro(engine: Engine, cfg: dict, series_ids: list[str] | None, api_key: str) -> bool:
    start = date.fromisoformat(cfg["data"]["start_date"])
    wanted = series_ids or cfg["data"]["fred_series"]
    today = date.today()

    with engine.begin() as conn:
        run_id = conn.execute(
            text(
                "insert into ops.pipeline_run (pipeline_name, git_sha, config_hash) "
                "values ('load_macro', :sha, :cfg) returning run_id"
            ),
            {"sha": _git_sha(), "cfg": _config_hash()},
        ).scalar_one()
        meta = {
            r.series_id: (r.availability_method, r.frequency)
            for r in conn.execute(
                text("select series_id, availability_method, frequency from macro.series")
            )
        }

    rows_read = rows_written = 0
    failures: list[str] = []

    for sid in wanted:
        if sid not in meta:
            failures.append(f"{sid}: not in macro.series")
            continue
        method, frequency = meta[sid]
        try:
            rows = fetch_observations(sid, api_key, start, all_vintages=(method == "alfred_vintage"))
        except RuntimeError as exc:
            failures.append(f"{sid}: {exc}")
            log.error("%s", exc)
            continue
        if method == "rule_lag":
            rows = apply_rule_lag(rows)
        rows_read += len(rows)

        results = macro_checks(rows, sid, method, frequency, today)
        with engine.begin() as conn:
            _store_dq(conn, run_id, results)
            blocking = [r for r in results if r.blocking]
            if blocking:
                failures.append(f"{sid}: blocked by {', '.join(r.check_name for r in blocking)}")
                log.error("%s rejected", sid)
                continue
            conn.execute(text("delete from macro.observation_vintage where series_id = :s"), {"s": sid})
            conn.execute(INSERT_SQL, [{"series_id": sid, **r} for r in rows])
            rows_written += len(rows)
            warned = [r.check_name for r in results if not r.passed]
            log.info("%s loaded: %d rows%s", sid, len(rows),
                     f" (warnings: {', '.join(warned)})" if warned else "")

    with engine.begin() as conn:
        conn.execute(
            text(
                "update ops.pipeline_run set finished_at = now(), status = :st, "
                "rows_read = :rr, rows_written = :rw, error_message = :err where run_id = :id"
            ),
            {"st": "failed" if failures else "succeeded", "rr": rows_read, "rw": rows_written,
             "err": "; ".join(failures) or None, "id": run_id},
        )
    return not failures
