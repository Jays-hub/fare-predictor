-- Schema for the Neon `snapshots` table (the source of truth for collected fares).
--
-- STATUS: reconstructed 2026-06-10 from the collector's column list — the live
-- table was created ad hoc before this file existed. Verify against the live DB:
--
--     psql "$DATABASE_URL" -c '\d snapshots'
--
-- and replace these definitions with the real output if they differ. The unique
-- index matters most: it is the write-time dedup contract that the collector's
-- ON CONFLICT DO NOTHING relies on (the INSERT names no conflict target, so it
-- depends on a unique index existing).

CREATE TABLE IF NOT EXISTS snapshots (
    observed_at   timestamptz NOT NULL,   -- one shared timestamp per collection cycle (UTC)
    origin        text        NOT NULL,
    dest          text        NOT NULL,
    dep_date      date        NOT NULL,
    ret_date      date        NOT NULL,
    carrier       text        NOT NULL,
    cabin         text,
    price         integer     NOT NULL,   -- round-trip TOTAL for the outbound option, whole dollars
    stops         integer,
    nonstop       boolean,
    duration_min  integer,
    dep_time_raw  text,
    dep_hour      integer,                -- 0-23; buckets are derived at modeling time
    dep_minute    integer,
    dep_dow       text,
    source        text,
    price_level   text                    -- Google's low/typical/high verdict for the query
);

-- Write-time dedup. Column set chosen to identify one observed flight option
-- within one cycle; re-running a cycle's insert is then a no-op.
CREATE UNIQUE INDEX IF NOT EXISTS snapshots_dedup_idx ON snapshots (
    observed_at, origin, dest, dep_date, ret_date, carrier,
    dep_time_raw, price, stops, duration_min
);

-- Micro-migrations (each also self-applied by collect.py on every run):
-- 2026-06-10: ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS price_level text;
