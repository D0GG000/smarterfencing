"""
Delete anonymous site_user rows (no email, no auth, no jobs, no comments).

Runs in small batches with pauses so SQLite stays responsive for the analyzer.
"""

import os
import time
import logging

from sqlalchemy import text

from models import db

logger = logging.getLogger(__name__)

CLEANED_COUNTER_KEY = "anonymous_users_cleaned_total"

# Users matching this query are cookie-only visitors safe to remove.
_ELIGIBLE_SQL = """
    SELECT u.id
    FROM site_user u
    WHERE (u.email IS NULL OR TRIM(u.email) = '')
      AND u.google_sub IS NULL
      AND (u.password_hash IS NULL OR TRIM(u.password_hash) = '')
      AND NOT EXISTS (
          SELECT 1 FROM user_job j WHERE j.user_id = u.id
      )
      AND NOT EXISTS (
          SELECT 1 FROM comments c WHERE c.user_id = u.id
      )
    LIMIT :limit
"""


def count_eligible_anonymous_users():
    """Return how many site_user rows would be deleted."""
    row = db.session.execute(
        text(
            """
            SELECT COUNT(*) FROM site_user u
            WHERE (u.email IS NULL OR TRIM(u.email) = '')
              AND u.google_sub IS NULL
              AND (u.password_hash IS NULL OR TRIM(u.password_hash) = '')
              AND NOT EXISTS (
                  SELECT 1 FROM user_job j WHERE j.user_id = u.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM comments c WHERE c.user_id = u.id
              )
            """
        )
    ).scalar()
    return int(row or 0)


def get_cleaned_total():
    """Return cumulative anonymous users removed by idle cleanup."""
    row = db.session.execute(
        text("SELECT value FROM app_setting WHERE key = :key"),
        {"key": CLEANED_COUNTER_KEY},
    ).scalar()
    try:
        return int(row or 0)
    except (TypeError, ValueError):
        return 0


def record_cleaned(count):
    """Add to the cumulative cleaned counter."""
    if count <= 0:
        return get_cleaned_total()

    db.session.execute(
        text(
            """
            INSERT INTO app_setting (key, value) VALUES (:key, :value)
            ON CONFLICT(key) DO UPDATE SET
                value = CAST(CAST(app_setting.value AS INTEGER) + :delta AS TEXT)
            """
        ),
        {"key": CLEANED_COUNTER_KEY, "value": str(count), "delta": count},
    )
    db.session.commit()
    return get_cleaned_total()


def reset_cleaned_total():
    """Reset the cumulative cleaned counter to zero."""
    db.session.execute(
        text(
            """
            INSERT INTO app_setting (key, value) VALUES (:key, '0')
            ON CONFLICT(key) DO UPDATE SET value = '0'
            """
        ),
        {"key": CLEANED_COUNTER_KEY},
    )
    db.session.commit()
    return 0


def get_cleanup_stats():
    """Stats for admin dashboard."""
    return {
        "cleaned_total": get_cleaned_total(),
        "eligible_remaining": count_eligible_anonymous_users(),
    }


def cleanup_anonymous_users(
    batch_size=None,
    max_batches=None,
    sleep_ms=None,
    dry_run=False,
):
    """
    Delete anonymous users in small batches.

    Returns:
        int: Total rows deleted (or that would be deleted if dry_run).
    """
    from job_queue_models import SiteUser

    batch_size = batch_size or int(os.environ.get("USER_CLEANUP_BATCH_SIZE", "25"))
    max_batches = max_batches or int(os.environ.get("USER_CLEANUP_MAX_BATCHES", "4"))
    sleep_ms = sleep_ms or int(os.environ.get("USER_CLEANUP_SLEEP_MS", "200"))

    batch_size = max(1, min(batch_size, 500))
    max_batches = max(1, min(max_batches, 100))
    sleep_ms = max(0, min(sleep_ms, 5000))

    total = 0

    for batch_num in range(max_batches):
        ids = db.session.execute(
            text(_ELIGIBLE_SQL),
            {"limit": batch_size},
        ).scalars().all()

        if not ids:
            break

        if dry_run:
            total += len(ids)
            logger.info(
                "Dry run batch %s: would delete %s user(s)",
                batch_num + 1,
                len(ids),
            )
            break

        deleted = (
            SiteUser.query.filter(SiteUser.id.in_(list(ids)))
            .delete(synchronize_session=False)
        )
        db.session.commit()
        total += deleted

        logger.info(
            "User cleanup batch %s: deleted %s row(s), %s total this run",
            batch_num + 1,
            deleted,
            total,
        )

        if batch_num + 1 < max_batches and deleted > 0 and sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    if total > 0 and not dry_run:
        record_cleaned(total)

    return total
