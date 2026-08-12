"""SQLite settings so admin traffic does not block the analyzer worker."""

import logging

from sqlalchemy import event

logger = logging.getLogger(__name__)


def configure_sqlite(engine):
    """Enable WAL + busy timeout for concurrent reads/writes."""
    url = str(engine.url)
    if not url.startswith("sqlite"):
        return

    busy_ms = int(__import__("os").environ.get("SQLITE_BUSY_TIMEOUT_MS", "5000"))

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={busy_ms}")
        cursor.close()

    logger.info("SQLite WAL enabled (busy_timeout=%sms)", busy_ms)
