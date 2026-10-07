-- =============================================================================
-- 002_seed_reference.sql
-- Reference data for Tier 1. Re-runnable (ON CONFLICT DO NOTHING).
-- first_trade_date is left NULL and derived from loaded data, not hard-coded.
-- =============================================================================

insert into ref.data_source (name, terms_url, redistribution_allowed, notes) values
    ('yahoo_finance', 'https://finance.yahoo.com', false,
     'Development only. Unofficial scraper via yfinance; terms do not permit redistribution. Replace via the DataSource adapter before any public site.'),
    ('fred', 'https://fred.stlouisfed.org/docs/api/terms_of_use.html', true,
     'Free API. Series may carry their own copyright terms; confirm per series and attribute before publishing.'),
    ('alfred', 'https://alfred.stlouisfed.org', true,
     'Vintage (point-in-time) data from the St. Louis Fed. Same terms notes as FRED.')
on conflict (name) do nothing;

insert into ref.asset (ticker, name, asset_type, sector) values
    ('^GSPC', 'S&P 500 Index',               'index',            null),
    ('^VIX',  'CBOE Volatility Index',       'volatility_index', null),
    ('^IXIC', 'NASDAQ Composite',            'index',            null),
    ('^DJI',  'Dow Jones Industrial Average','index',            null),
    ('SPY',   'SPDR S&P 500 ETF',            'etf',              null),
    ('QQQ',   'Invesco QQQ Trust',           'etf',              null),
    ('DIA',   'SPDR Dow Jones Industrial Average ETF', 'etf',    null),
    ('TLT',   'iShares 20+ Year Treasury Bond ETF',    'etf',    'Treasuries'),
    ('IEF',   'iShares 7-10 Year Treasury Bond ETF',   'etf',    'Treasuries'),
    ('SHY',   'iShares 1-3 Year Treasury Bond ETF',    'etf',    'Treasuries'),
    ('LQD',   'iShares iBoxx Investment Grade Corporate Bond ETF', 'etf', 'Credit'),
    ('HYG',   'iShares iBoxx High Yield Corporate Bond ETF',       'etf', 'Credit'),
    ('XLF',   'Financial Select Sector SPDR',          'etf',    'Financials'),
    ('XLK',   'Technology Select Sector SPDR',         'etf',    'Technology'),
    ('XLE',   'Energy Select Sector SPDR',             'etf',    'Energy'),
    ('XLV',   'Health Care Select Sector SPDR',        'etf',    'Health Care'),
    ('XLY',   'Consumer Discretionary Select Sector SPDR', 'etf','Consumer Discretionary'),
    ('XLP',   'Consumer Staples Select Sector SPDR',   'etf',    'Consumer Staples'),
    ('XLI',   'Industrial Select Sector SPDR',         'etf',    'Industrials'),
    ('XLU',   'Utilities Select Sector SPDR',          'etf',    'Utilities'),
    ('XLB',   'Materials Select Sector SPDR',          'etf',    'Materials')
on conflict (ticker) do nothing;

insert into macro.series
    (series_id, title, frequency, units, seasonally_adjusted, source_id, availability_method, rule_lag_note)
select v.series_id, v.title, v.frequency, v.units, v.sa, s.source_id, v.method, v.note
from (values
    ('DFF',      'Effective Federal Funds Rate',       'daily',   'percent',           null::boolean, 'fred',   'rule_lag',       'Available 1 business day after observation date'),
    ('DGS10',    '10-Year Treasury Constant Maturity', 'daily',   'percent',           null,          'fred',   'rule_lag',       'Available 1 business day after observation date'),
    ('DGS2',     '2-Year Treasury Constant Maturity',  'daily',   'percent',           null,          'fred',   'rule_lag',       'Available 1 business day after observation date'),
    ('T10Y2Y',   '10Y minus 2Y Treasury Spread',       'daily',   'percent',           null,          'fred',   'rule_lag',       'Available 1 business day after observation date'),
    ('UNRATE',   'Unemployment Rate',                  'monthly', 'percent',           true,          'alfred', 'alfred_vintage', null),
    ('CPIAUCSL', 'CPI for All Urban Consumers (SA)',   'monthly', 'index 1982-84=100', true,          'alfred', 'alfred_vintage', null)
) as v(series_id, title, frequency, units, sa, source_name, method, note)
join ref.data_source s on s.name = v.source_name
on conflict (series_id) do nothing;
