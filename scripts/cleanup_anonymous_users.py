#!/usr/bin/env python3
"""Manual run: delete anonymous site_user rows in small batches."""

import os
import sys

_APP_PYTHON = os.environ.get("APP_PYTHON", "/opt/conda/envs/mmpose-env/bin/python3")
if (
    os.path.isfile(_APP_PYTHON)
    and os.path.realpath(sys.executable) != os.path.realpath(_APP_PYTHON)
):
    os.execv(_APP_PYTHON, [_APP_PYTHON, *sys.argv])

import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_SCRIPT_DIR)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

import app as flask_app  # noqa: E402
from user_cleanup import cleanup_anonymous_users, count_eligible_anonymous_users  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Remove site_user rows with no email, no auth, no jobs, no comments."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("USER_CLEANUP_BATCH_SIZE", "25")),
        help="Rows per batch (default: 25)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=int(os.environ.get("USER_CLEANUP_MAX_BATCHES", "4")),
        help="Max batches per invocation (default: 4)",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=int(os.environ.get("USER_CLEANUP_SLEEP_MS", "200")),
        help="Pause between batches in ms (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count eligible users; delete at most one batch without committing",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print how many users are eligible for cleanup",
    )
    args = parser.parse_args()

    with flask_app.app.app_context():
        eligible = count_eligible_anonymous_users()
        print(f"Eligible anonymous users: {eligible}")

        if args.count_only:
            return

        deleted = cleanup_anonymous_users(
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            sleep_ms=args.sleep_ms,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            print(f"Dry run: would delete up to {deleted} user(s) in first batch")
        else:
            print(f"Deleted {deleted} user(s)")


if __name__ == "__main__":
    main()
