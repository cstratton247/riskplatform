from datetime import date

from riskplatform.ingest.fred import OPEN_ENDED, apply_rule_lag, macro_checks, parse_observations


def _obs(d, rs, re, v):
    return {"date": d, "realtime_start": rs, "realtime_end": re, "value": v}


def test_parse_skips_missing_and_reads_vintages():
    payload = {"observations": [
        _obs("2020-03-01", "2020-04-03", "2020-05-07", "4.4"),
        _obs("2020-03-01", "2020-05-08", "9999-12-31", "4.5"),
        _obs("2020-04-01", "2020-05-08", "9999-12-31", "."),
    ]}
    rows = parse_observations(payload)
    assert len(rows) == 2
    assert rows[1]["valid_to"] == OPEN_ENDED and rows[1]["value"] == 4.5


def _daily(d):
    return {"observation_date": d, "valid_from": d, "valid_to": OPEN_ENDED, "value": 1.0}


def test_rule_lag_next_federal_business_day():
    out = {r["observation_date"]: r["valid_from"] for r in apply_rule_lag([
        _daily(date(2020, 1, 3)),     # Friday -> Monday
        _daily(date(2020, 11, 25)),   # day before Thanksgiving -> Friday
        _daily(date(2020, 12, 31)),   # Thursday; Jan 1 is a holiday -> Monday Jan 4
    ])}
    assert out[date(2020, 1, 3)] == date(2020, 1, 6)
    assert out[date(2020, 11, 25)] == date(2020, 11, 27)
    assert out[date(2020, 12, 31)] == date(2021, 1, 4)


def test_first_vintage_lag_check_flags_implausible_release_dates():
    good = {"observation_date": date(2020, 3, 1), "valid_from": date(2020, 4, 3),
            "valid_to": OPEN_ENDED, "value": 4.4}
    bad = {"observation_date": date(2020, 4, 1), "valid_from": date(2020, 4, 2),   # same-month
           "valid_to": OPEN_ENDED, "value": 4.5}
    today = date(2020, 6, 1)
    ok = {r.check_name: r for r in macro_checks([good], "UNRATE", "alfred_vintage", "monthly", today)}
    assert ok["first_vintage_lag"].passed
    flagged = {r.check_name: r for r in macro_checks([good, bad], "UNRATE", "alfred_vintage", "monthly", today)}
    assert not flagged["first_vintage_lag"].passed and not flagged["first_vintage_lag"].blocking


def test_staleness_and_empty():
    row = {**_daily(date(2020, 1, 3))}
    r = {x.check_name: x for x in macro_checks([row], "DGS10", "rule_lag", "daily", date(2020, 3, 1))}
    assert not r["staleness"].passed
    assert macro_checks([], "DGS10", "rule_lag", "daily", date(2020, 3, 1))[0].blocking
