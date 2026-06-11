-- Schema for the Neon `snapshots` table (the source of truth for collected fares).
--
-- STATUS: indexes verified against live pg_indexes on 2026-06-11. Column
-- names/types are still reconstructed from the collector's column list (the
-- live table was created ad hoc before this file existed); to double-check:
--
--     SELECT column_name, data_type, is_nullable FROM information_schema.columns
--       WHERE table_name = 'snapshots' ORDER BY ordinal_position;
--
-- The unique dedup index matters most: it is the write-time dedup contract that
-- the collector's ON CONFLICT DO NOTHING relies on (the INSERT names no conflict
-- target, so it fires on any unique violation).

CREATE TABLE IF NOT EXISTS snapshots (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- surrogate key (flavor inferred from snapshots_pkey; confirm via the columns query)
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

-- Write-time dedup (verified live 2026-06-11): one row per outbound option
-- per cycle, keyed by carrier + departure time + price. Re-running a cycle's
-- insert is a no-op. NULL dep_hour/dep_minute rows are never dedup'd here
-- (NULLS DISTINCT default), but the parse layer already drops/dedups those.
CREATE UNIQUE INDEX IF NOT EXISTS snapshots_dedup ON snapshots (
    observed_at, origin, dest, dep_date, ret_date, carrier,
    dep_hour, dep_minute, price
);

-- Micro-migrations (each also self-applied by collect.py on every run):
-- 2026-06-10: ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS price_level text;
