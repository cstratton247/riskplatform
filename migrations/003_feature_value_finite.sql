-- 003_feature_value_finite.sql
-- Missing feature values must be absent (or NULL), never stored as NaN/Infinity.
-- Removes any NaN/Infinity rows written by an earlier loader, then enforces the rule.
delete from feat.feature_value
where value in ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision);

alter table feat.feature_value
    add constraint feature_value_finite
    check (value is null
           or value not in ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision));
