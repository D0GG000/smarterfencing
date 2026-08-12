"""Lightweight SQLite/Postgres column adds for existing deployments."""

import logging
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)


def _ensure_table_schema(conn, inspector, table, create_sql, indexes, columns, required):
    """Reconcile a table with the desired schema.

    - Missing table -> create it.
    - Exists but missing a *required* column (legacy/wrong schema): rebuild it when
      empty (no usable data to lose), otherwise best-effort add the columns as
      nullable.
    - Add any other missing columns.

    `table` values are hardcoded literals (no user input), so f-string SQL is safe.
    """
    if table not in inspector.get_table_names():
        conn.execute(text(create_sql))
        for idx in indexes:
            conn.execute(text(idx))
        logger.info("Created %s table", table)
        return

    existing = {c["name"] for c in inspector.get_columns(table)}
    missing_required = [c for c in required if c not in existing]
    if missing_required:
        try:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        except Exception:
            row_count = None
        if row_count == 0:
            conn.execute(text(f"DROP TABLE {table}"))
            conn.execute(text(create_sql))
            for idx in indexes:
                conn.execute(text(idx))
            logger.info(
                "Rebuilt %s table (legacy schema missing %s)",
                table, ", ".join(missing_required),
            )
            return
        logger.warning(
            "%s is missing required columns %s but has %s rows; "
            "adding them as nullable", table, missing_required, row_count,
        )

    for col, ddl in columns.items():
        if col not in existing:
            conn.execute(text(ddl))
            logger.info("Added %s.%s", table, col)


def ensure_extended_schema(engine):
    """Add auth and correction columns if missing (idempotent)."""
    try:
        inspector = inspect(engine)
    except Exception as e:
        logger.warning("Schema inspect failed: %s", e)
        return

    with engine.begin() as conn:
        if "site_user" in inspector.get_table_names():
            cols = {c["name"] for c in inspector.get_columns("site_user")}
            if "google_sub" not in cols:
                conn.execute(text("ALTER TABLE site_user ADD COLUMN google_sub VARCHAR(255)"))
                logger.info("Added site_user.google_sub")
            if "password_hash" not in cols:
                conn.execute(text("ALTER TABLE site_user ADD COLUMN password_hash TEXT"))
                logger.info("Added site_user.password_hash")
            if "display_name" not in cols:
                conn.execute(text("ALTER TABLE site_user ADD COLUMN display_name VARCHAR(255)"))
                logger.info("Added site_user.display_name")
            if "username" not in cols:
                conn.execute(text("ALTER TABLE site_user ADD COLUMN username VARCHAR(32)"))
                logger.info("Added site_user.username")
                try:
                    conn.execute(
                        text(
                            "CREATE UNIQUE INDEX IF NOT EXISTS ix_site_user_username "
                            "ON site_user (username)"
                        )
                    )
                except Exception as idx_err:
                    logger.warning("username index: %s", idx_err)

        if "user_job" in inspector.get_table_names():
            cols = {c["name"] for c in inspector.get_columns("user_job")}
            if "prediction_corrections_json" not in cols:
                conn.execute(
                    text("ALTER TABLE user_job ADD COLUMN prediction_corrections_json TEXT")
                )
                logger.info("Added user_job.prediction_corrections_json")
            if "touch_deletions_json" not in cols:
                conn.execute(text("ALTER TABLE user_job ADD COLUMN touch_deletions_json TEXT"))
                logger.info("Added user_job.touch_deletions_json")
            if "share_token" not in cols:
                conn.execute(text("ALTER TABLE user_job ADD COLUMN share_token VARCHAR(64)"))
                logger.info("Added user_job.share_token")
            if "job_type" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE user_job ADD COLUMN job_type VARCHAR(20) "
                        "NOT NULL DEFAULT 'analysis'"
                    )
                )
                logger.info("Added user_job.job_type")
            if "highlight_reel_key" not in cols:
                conn.execute(
                    text("ALTER TABLE user_job ADD COLUMN highlight_reel_key VARCHAR(255)")
                )
                logger.info("Added user_job.highlight_reel_key")
            if "macro_corrections_json" not in cols:
                conn.execute(
                    text("ALTER TABLE user_job ADD COLUMN macro_corrections_json TEXT")
                )
                logger.info("Added user_job.macro_corrections_json")
            if "llm_analysis_json" not in cols:
                conn.execute(
                    text("ALTER TABLE user_job ADD COLUMN llm_analysis_json TEXT")
                )
                logger.info("Added user_job.llm_analysis_json")

        if "touch_share" in inspector.get_table_names():
            cols = {c["name"] for c in inspector.get_columns("touch_share")}
            if "touch_ref" not in cols:
                conn.execute(text("ALTER TABLE touch_share ADD COLUMN touch_ref VARCHAR(32)"))
                logger.info("Added touch_share.touch_ref")
            if "kind" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE touch_share ADD COLUMN kind VARCHAR(20) "
                        "NOT NULL DEFAULT 'touch'"
                    )
                )
                logger.info("Added touch_share.kind")

        if "app_setting" not in inspector.get_table_names():
            conn.execute(
                text(
                    """
                    CREATE TABLE app_setting (
                        key VARCHAR(128) PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
            )
            logger.info("Created app_setting table")

        # Community Hub: public posts + comments. db.create_all() handles fresh
        # databases, but older deployments may have a legacy/incomplete schema
        # (create_all never alters existing tables), so reconcile explicitly.
        _ensure_table_schema(
            conn,
            inspector,
            "community_post",
            """
            CREATE TABLE community_post (
                id VARCHAR(36) PRIMARY KEY,
                author_user_id VARCHAR(36) NOT NULL,
                job_id VARCHAR(20) NOT NULL,
                kind VARCHAR(20) NOT NULL DEFAULT 'touch',
                touch_id VARCHAR(512),
                touch_ref VARCHAR(32),
                caption TEXT,
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                created_at TIMESTAMP NOT NULL
            )
            """,
            [
                "CREATE INDEX IF NOT EXISTS ix_community_post_status ON community_post (status)",
                "CREATE INDEX IF NOT EXISTS ix_community_post_created_at ON community_post (created_at)",
            ],
            columns={
                "author_user_id": "ALTER TABLE community_post ADD COLUMN author_user_id VARCHAR(36)",
                "job_id": "ALTER TABLE community_post ADD COLUMN job_id VARCHAR(20)",
                "kind": "ALTER TABLE community_post ADD COLUMN kind VARCHAR(20) NOT NULL DEFAULT 'touch'",
                "touch_id": "ALTER TABLE community_post ADD COLUMN touch_id VARCHAR(512)",
                "touch_ref": "ALTER TABLE community_post ADD COLUMN touch_ref VARCHAR(32)",
                "caption": "ALTER TABLE community_post ADD COLUMN caption TEXT",
                "status": "ALTER TABLE community_post ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'active'",
                "created_at": "ALTER TABLE community_post ADD COLUMN created_at TIMESTAMP",
            },
            required=["id", "author_user_id", "job_id", "kind", "status", "created_at"],
        )

        _ensure_table_schema(
            conn,
            inspector,
            "community_comment",
            """
            CREATE TABLE community_comment (
                id VARCHAR(36) PRIMARY KEY,
                post_id VARCHAR(36) NOT NULL,
                author_user_id VARCHAR(36) NOT NULL,
                body TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
            """,
            [
                "CREATE INDEX IF NOT EXISTS ix_community_comment_post_id ON community_comment (post_id)",
            ],
            columns={
                "post_id": "ALTER TABLE community_comment ADD COLUMN post_id VARCHAR(36)",
                "author_user_id": "ALTER TABLE community_comment ADD COLUMN author_user_id VARCHAR(36)",
                "body": "ALTER TABLE community_comment ADD COLUMN body TEXT",
                "created_at": "ALTER TABLE community_comment ADD COLUMN created_at TIMESTAMP",
            },
            required=["id", "post_id", "author_user_id", "body", "created_at"],
        )
