#!/usr/bin/env python3
"""List all rows in all user tables of blog.db (large columns truncated)."""

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime

DEFAULT_DB = "/workspace/blog/blog.db"
TRUNCATE_AT = 200
LARGE_COL_HINTS = (
    "json",
    "html",
    "markdown",
    "body",
    "hash",
    "results",
    "selections",
    "corrections",
    "deletions",
    "error_message",
)


def fmt_val(val, col_name="", truncate_at=TRUNCATE_AT):
    if val is None:
        return None
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, bytes):
        return f"<bytes len={len(val)}>"
    s = str(val)
    col_lower = col_name.lower()
    if any(h in col_lower for h in LARGE_COL_HINTS) and len(s) > truncate_at:
        return s[:truncate_at] + f"... [truncated, total {len(s)} chars]"
    if len(s) > truncate_at:
        return s[:truncate_at] + f"... [truncated, total {len(s)} chars]"
    return val


def dump_db(db_path, truncate_at=TRUNCATE_AT, out=sys.stdout):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    tables = [
        r[0]
        for r in conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
    ]

    print(f"Database: {db_path}\n", file=out)

    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
        print("=" * 80, file=out)
        print(f"TABLE: {table}  ({count} rows)", file=out)
        print("-" * 80, file=out)

        if count == 0:
            print("(empty)\n", file=out)
            continue

        cols = [r[1] for r in conn.execute(f"PRAGMA table_info([{table}])")]
        for i, row in enumerate(conn.execute(f"SELECT * FROM [{table}]"), 1):
            obj = {col: fmt_val(row[col], col, truncate_at) for col in cols}
            print(f"[{table} #{i}]", file=out)
            print(json.dumps(obj, indent=2, default=str), file=out)
            print(file=out)

    conn.close()
    print("Done.", file=out)


def main():
    parser = argparse.ArgumentParser(description="Dump all rows from blog.db tables.")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"SQLite database path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write dump to this file instead of stdout",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=TRUNCATE_AT,
        help=f"Max chars for large columns (default: {TRUNCATE_AT})",
    )
    args = parser.parse_args()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            dump_db(args.db, truncate_at=args.truncate, out=f)
        print(f"Wrote dump to {args.output}", file=sys.stderr)
    else:
        dump_db(args.db, truncate_at=args.truncate, out=sys.stdout)


if __name__ == "__main__":
    main()
