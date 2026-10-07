-- =============================================================================
-- 001_initial_schema.sql
-- Financial Risk & Predictive Analytics Platform
-- Companion to docs/milestone1_spec.md (v0.2)
--
-- Design principles
--   * Point-in-time correctness: nothing stored here lets a query "see" data
--     that was not known on the as-of date (macro vintages, label_known_date).
--   * Reproducibility: every derived row traces to a pipeline run, feature
--     version, and model version.
--   * Idempotent loads: natural primary keys allow INSERT ... ON CONFLICT.
--   * Asset-keyed everywhere, so Tier 2 (asset panel) needs no schema change.
--   * Outcomes are NOT stored on prediction rows; they are joined from ml.label,
--     so a prediction row can never be edited after the fact.
--
-- NOTE: Not yet executed against a live PostgreSQL instance. Run it in the
-- Docker container and fix any dialect issues before building on it.
-- Target: PostgreSQL 15+.
-- =============================================================================

create extension if not exists btree_gist;   -- needed for the vintage overlap constraint

create schema if not exists ref;     -- reference data
create schema if not exists raw;     -- immutable raw pulls
create schema if not exists mkt;     -- cleaned market data
create schema if not exists macro;   -- economic data with vintages
create schema if not exists feat;    -- engineered features
create schema if not exists ml;      -- labels, models, predictions, evaluation
create schema if not exists ops;     -- pipeline runs and data-quality results

-- -----------------------------------------------------------------------------
-- ops: pipeline observability
-- -----------------------------------------------------------------------------
create table ops.pipeline_run (
    run_id        bigint generated always as identity primary key,
    pipeline_name text        not null,
    started_at    timestamptz not null default now(),
    finished_at   timestamptz,
    status        text        not null default 'running'
                  check (status in ('running', 'succeeded', 'failed')),
    git_sha       text,
    config_hash   text,
    rows_read     bigint,
    rows_written  bigint,
    error_message text,
    check (finished_at is null or finished_at >= started_at)
);

create table ops.dq_check_result (
    check_result_id bigint generated always as identity primary key,
    run_id          bigint      not null references ops.pipeline_run (run_id),
    check_name      text        not null,
    severity        text        not null check (severity in ('info', 'warn', 'error')),
    passed          boolean     not null,
    entity          text,                       -- e.g. ticker or series_id
    details         jsonb,
    checked_at      timestamptz not null default now()
);
create index dq_check_result_run_idx  on ops.dq_check_result (run_id);
create index dq_check_result_name_idx on ops.dq_check_result (check_name, checked_at desc);

-- -----------------------------------------------------------------------------
-- ref: reference data
-- -----------------------------------------------------------------------------
create table ref.data_source (
    source_id              smallint generated always as identity primary key,
    name                   text    not null unique,
    terms_url              text,
    redistribution_allowed boolean not null default false,
    notes                  text
);

create table ref.asset (
    asset_id         integer generated always as identity primary key,
    ticker           text    not null unique,
    name             text,
    asset_type       text    not null
                     check (asset_type in ('index', 'volatility_index', 'etf', 'equity')),
    sector           text,
    currency         char(3) not null default 'USD',
    first_trade_date date,                       -- derived from loaded data
    is_active        boolean not null default true
);

-- One row per NYSE trading day (populated from a market calendar library).
create table ref.trading_day (
    trade_date     date primary key,
    is_early_close boolean not null default false
);

-- -----------------------------------------------------------------------------
-- raw: immutable record of what each vendor pull returned
-- -----------------------------------------------------------------------------
create table raw.price_pull (
    pull_id         bigint generated always as identity primary key,
    run_id          bigint      not null references ops.pipeline_run (run_id),
    source_id       smallint    not null references ref.data_source (source_id),
    asset_id        integer     not null references ref.asset (asset_id),
    requested_start date        not null,
    requested_end   date        not null,
    fetched_at      timestamptz not null default now(),
    payload_hash    text        not null,        -- detects vendor revisions between pulls
    row_count       integer     not null,
    payload         jsonb       not null,
    check (requested_end >= requested_start)
);
create index price_pull_asset_idx on raw.price_pull (asset_id, fetched_at desc);

-- -----------------------------------------------------------------------------
-- mkt: cleaned daily prices (latest accepted version; history lives in raw)
-- Business-rule checks (close within [low, high], outliers, staleness) are done
-- in the pipeline and recorded in ops.dq_check_result; only hard invariants are
-- enforced here.
-- -----------------------------------------------------------------------------
create table mkt.price_daily (
    asset_id    integer       not null references ref.asset (asset_id),
    trade_date  date          not null references ref.trading_day (trade_date),
    open        numeric(18,6),
    high        numeric(18,6),
    low         numeric(18,6),
    close       numeric(18,6) not null,
    adj_close   numeric(18,6),
    volume      bigint,
    source_id   smallint      not null references ref.data_source (source_id),
    pull_id     bigint        references raw.price_pull (pull_id),
    ingested_at timestamptz   not null default now(),
    primary key (asset_id, trade_date),
    check (close > 0),
    check (open      is null or open      > 0),
    check (adj_close is null or adj_close > 0),
    check (high is null or low is null or high >= low),
    check (volume is null or volume >= 0)
);
create index price_daily_date_idx on mkt.price_daily (trade_date);   -- cross-sectional scans

-- -----------------------------------------------------------------------------
-- macro: economic series with point-in-time availability
--
-- Each row says: "observation_date's value was `value`, and this was the
-- publicly known figure from valid_from through valid_to (inclusive)."
--   * alfred_vintage series: valid_from/valid_to come from ALFRED
--     (realtime_start / realtime_end), so revisions create new rows.
--   * rule_lag series (daily, unrevised): valid_from = observation_date plus
--     one business day, valid_to = 9999-12-31. Scoring runs after the daily
--     release, and the overlay applies positions with a further execution lag.
-- -----------------------------------------------------------------------------
create table macro.series (
    series_id           text primary key,                -- FRED id, e.g. 'DGS10'
    title               text     not null,
    frequency           text     not null check (frequency in ('daily', 'monthly', 'quarterly')),
    units               text,
    seasonally_adjusted boolean,
    source_id           smallint not null references ref.data_source (source_id),
    availability_method text     not null check (availability_method in ('alfred_vintage', 'rule_lag')),
    rule_lag_note       text
);

create table macro.observation_vintage (
    series_id        text        not null references macro.series (series_id),
    observation_date date        not null,
    valid_from       date        not null,
    valid_to         date        not null default '9999-12-31',
    value            numeric(20,6),                      -- null if published as missing
    ingested_at      timestamptz not null default now(),
    primary key (series_id, observation_date, valid_from),
    check (valid_to >= valid_from),
    -- a given observation can have only one known value on any given date
    exclude using gist (
        series_id        with =,
        observation_date with =,
        daterange(valid_from, valid_to, '[]') with &&
    )
);
create index observation_vintage_pit_idx on macro.observation_vintage (series_id, valid_from);

-- The ONLY sanctioned way for feature code to read macro data.
-- Returns the most recent observation whose value was known on p_as_of.
create function macro.latest_known_as_of(p_series text, p_as_of date)
returns table (observation_date date, value numeric)
language sql stable as $$
    select v.observation_date, v.value
    from macro.observation_vintage v
    where v.series_id  = p_series
      and v.valid_from <= p_as_of
      and v.valid_to   >= p_as_of
      and v.value is not null
    order by v.observation_date desc
    limit 1;
$$;

-- -----------------------------------------------------------------------------
-- feat: feature definitions and values (long format, versioned)
-- Long format lets features be added without schema changes; wide matrices are
-- built with pivot queries or materialized views for dashboards.
-- -----------------------------------------------------------------------------
create table feat.feature_definition (
    feature_id    integer generated always as identity primary key,
    name          text    not null,
    version       integer not null default 1,
    feature_group text    not null,
    description   text    not null,
    lookback_days integer check (lookback_days is null or lookback_days >= 0),
    formula       text,
    code_hash     text,
    unique (name, version)
);

create table feat.feature_set (
    feature_set_id integer generated always as identity primary key,
    name           text    not null,
    version        integer not null,
    description    text,
    unique (name, version)
);

create table feat.feature_set_member (
    feature_set_id integer not null references feat.feature_set (feature_set_id),
    feature_id     integer not null references feat.feature_definition (feature_id),
    primary key (feature_set_id, feature_id)
);

create table feat.feature_value (
    asset_id    integer          not null references ref.asset (asset_id),
    as_of_date  date             not null references ref.trading_day (trade_date),
    feature_id  integer          not null references feat.feature_definition (feature_id),
    value       double precision,
    run_id      bigint           references ops.pipeline_run (run_id),
    computed_at timestamptz      not null default now(),
    primary key (asset_id, as_of_date, feature_id)       -- fast per-asset matrix reads
);
create index feature_value_feature_idx on feat.feature_value (feature_id, as_of_date);

-- -----------------------------------------------------------------------------
-- ml: labels, models, predictions, evaluation
-- -----------------------------------------------------------------------------
create table ml.label_definition (
    label_def_id           integer generated always as identity primary key,
    name                   text     not null,
    version                integer  not null,
    horizon_days           smallint not null check (horizon_days > 0),
    threshold_quantile     numeric(4,3) not null
                           check (threshold_quantile > 0 and threshold_quantile < 1),
    threshold_lookback_obs integer  not null check (threshold_lookback_obs > 0),
    spec                   jsonb,
    unique (name, version)
);

-- label_known_date = the trading day on which y becomes fully determined
-- (as_of_date + horizon). Training queries must filter label_known_date <= the
-- training cutoff; this makes label leakage a query-level impossibility.
create table ml.label (
    label_def_id     integer          not null references ml.label_definition (label_def_id),
    asset_id         integer          not null references ref.asset (asset_id),
    as_of_date       date             not null references ref.trading_day (trade_date),
    rv_fwd           double precision,
    threshold        double precision,
    y                smallint         check (y in (0, 1)),
    label_known_date date,
    primary key (label_def_id, asset_id, as_of_date),
    check ((y is null) = (label_known_date is null)),
    check (label_known_date is null or label_known_date > as_of_date)
);
create index label_known_idx on ml.label (label_def_id, asset_id, label_known_date);

create table ml.model_version (
    model_version_id integer generated always as identity primary key,
    name             text    not null,
    family           text    not null
                     check (family in ('prevalence', 'persistence', 'har', 'garch', 'vix',
                                       'logistic', 'gradient_boosting', 'random_forest')),
    track            text    not null check (track in ('classification', 'regression')),
    purpose          text    not null
                     check (purpose in ('walk_forward', 'final_holdout', 'production')),
    fold_label       text,                                -- e.g. 'wf_2018'
    label_def_id     integer not null references ml.label_definition (label_def_id),
    feature_set_id   integer references feat.feature_set (feature_set_id),  -- null for feature-free benchmarks
    train_start      date    not null,
    train_end        date    not null,
    hyperparameters  jsonb   not null default '{}'::jsonb,
    data_hash        text    not null,
    git_sha          text    not null,
    artifact_uri     text,
    status           text    not null default 'candidate'
                     check (status in ('candidate', 'champion', 'retired')),
    created_at       timestamptz not null default now(),
    check (train_end >= train_start)
);
-- at most one deployed champion per label definition and track
create unique index one_champion_idx on ml.model_version (label_def_id, track)
    where status = 'champion' and purpose = 'production';

-- Every configuration tried, kept for the multiple-testing disclosure.
create table ml.model_trial (
    trial_id        bigint generated always as identity primary key,
    run_id          bigint references ops.pipeline_run (run_id),
    family          text   not null,
    hyperparameters jsonb  not null,
    outer_fold      text   not null,
    metric_name     text   not null,
    inner_cv_score  double precision,
    created_at      timestamptz not null default now()
);
create index model_trial_family_idx on ml.model_trial (family, outer_fold);

create table ml.prediction (
    model_version_id integer          not null references ml.model_version (model_version_id),
    asset_id         integer          not null references ref.asset (asset_id),
    as_of_date       date             not null references ref.trading_day (trade_date),
    probability      double precision check (probability is null or probability between 0 and 1),
    rv_forecast      double precision check (rv_forecast is null or rv_forecast >= 0),
    scored_at        timestamptz      not null default now(),
    primary key (model_version_id, asset_id, as_of_date)
);
create index prediction_asset_date_idx on ml.prediction (asset_id, as_of_date desc);

create table ml.evaluation_result (
    evaluation_id    bigint generated always as identity primary key,
    model_version_id integer          not null references ml.model_version (model_version_id),
    scope            text             not null,           -- 'fold:2018', 'pooled', 'regime:stress'
    metric_name      text             not null,
    value            double precision not null,
    ci_low           double precision,
    ci_high          double precision,
    ci_method        text,                                -- e.g. 'stationary_bootstrap'
    n_obs            integer,
    created_at       timestamptz      not null default now(),
    unique (model_version_id, scope, metric_name),
    check (ci_low is null or ci_high is null or ci_low <= ci_high)
);

-- Outcomes are joined from ml.label rather than stored on predictions.
create view ml.v_prediction_outcome as
select p.model_version_id,
       p.asset_id,
       p.as_of_date,
       p.probability,
       p.rv_forecast,
       l.y,
       l.rv_fwd,
       l.label_known_date
from ml.prediction p
join ml.model_version m on m.model_version_id = p.model_version_id
join ml.label l
  on l.label_def_id = m.label_def_id
 and l.asset_id     = p.asset_id
 and l.as_of_date   = p.as_of_date;

-- Risk-engine tables (portfolios, VaR/ES estimates, backtests, stress scenarios)
-- are deliberately deferred to the next migration, after the model track.
