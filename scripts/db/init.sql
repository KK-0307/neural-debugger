-- Runs once, when the Postgres data volume is first created (docker-entrypoint-initdb.d).
-- Schema management proper lives in the application's migrations (Milestone 2).
CREATE EXTENSION IF NOT EXISTS vector;
