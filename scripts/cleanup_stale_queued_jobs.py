#!/usr/bin/env python3
"""Manual run: remove pending/queued jobs older than a threshold (not automatic)."""

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
from job_cleanup import (  # noqa: E402
    cleanup_stale_queued_jobs,
    count_stale_queued_jobs,
    get_stale_job_cleanup_stats,
)


def main():
    default_hours = int(os.environ.get("STALE_JOB_HOURS", "48"))
    parser = argparse.ArgumentParser(
        description="Delete pending/queued jobs waiting longer than the threshold."
    )
    parser.add_argument("--hours", type=int, default=default_hours, help="Age threshold in hours")
    parser.add_argument("--days", type=int, help="Age threshold in days (overrides --hours)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("STALE_JOB_CLEANUP_BATCH_SIZE", "5")),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=int(os.environ.get("STALE_JOB_CLEANUP_MAX_BATCHES", "2")),
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=int(os.environ.get("STALE_JOB_CLEANUP_SLEEP_MS", "500")),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    with flask_app.app.app_context():
        stats = get_stale_job_cleanup_stats()
        if args.days is not None:
            age_label = f"{args.days} days"
            eligible = count_stale_queued_jobs(days=args.days)
            age_kw = {"days": args.days}
        else:
            age_label = f"{args.hours} hours"
            eligible = count_stale_queued_jobs(hours=args.hours)
            age_kw = {"hours": args.hours}

        print(f"Age threshold: {age_label}")
        print(f"Eligible jobs: {eligible}")
        print(f"Previously cleaned on startup (counter): {stats['stale_jobs_cleaned_total']}")

        if args.count_only:
            return

        removed = cleanup_stale_queued_jobs(
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            sleep_ms=args.sleep_ms,
            dry_run=args.dry_run,
            flask_app=flask_app.app,
            **age_kw,
        )

        if args.dry_run:
            print(f"Dry run: would remove up to {removed} job(s) in first batch")
        else:
            print(f"Removed {removed} stale job(s)")


if __name__ == "__main__":
    main()
