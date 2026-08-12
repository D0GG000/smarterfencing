# Print to terminal
python3 /app/scripts/dump_db.py
# Save to file (recommended — site_user is huge)
python3 /app/scripts/dump_db.py -o /workspace/blog/db_dump.txt
# Custom DB path
python3 /app/scripts/dump_db.py --db /workspace/blog/blog.db -o /workspace/blog/db_dump.txt
# Longer snippets for JSON columns
python3 /app/scripts/dump_db.py --truncate 500 -o /workspace/blog/db_dump.txt

# How many eligible?
python3 /app/scripts/cleanup_anonymous_users.py --count-only
# Dry run (first batch only, no delete)
python3 /app/scripts/cleanup_anonymous_users.py --dry-run
# Run once manually
python3 /app/scripts/cleanup_anonymous_users.py
 
# Count eligible stale jobs
python3 /app/scripts/cleanup_stale_queued_jobs.py --count-only
# Dry run
python3 /app/scripts/cleanup_stale_queued_jobs.py --dry-run
# Run once now
python3 /app/scripts/cleanup_stale_queued_jobs.py

