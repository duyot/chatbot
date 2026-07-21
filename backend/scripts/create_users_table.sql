-- Creates the users table for username/password authentication.
--
-- Preferred path in normal operation is Alembic (migration 0007):
--   alembic upgrade head
-- This raw DDL exists for environments without Alembic. Run e.g.:
--   psql "$DATABASE_URL" -f backend/scripts/create_users_table.sql
--
-- gen_random_uuid() is built in on PostgreSQL 13+.

CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR     NOT NULL,
    password_hash VARCHAR     NOT NULL,
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username);
