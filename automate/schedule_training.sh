#!/usr/bin/env bash
#
# Run add-on training every 24h (use with cron or systemd).
# Logs to automate/logs/run_training_YYYY-MM-DD.log
#
# Cron (run daily at 2:00 AM):
#   0 2 * * * /path/to/agentic-energy-ml/automate/schedule_training.sh
#
# Or run manually:
#   ./automate/schedule_training.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/run_training_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# Load .env if present (cron often has no env)
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

echo "=== $(date -Iseconds) Starting scheduled training ===" >> "$LOG_FILE"
python automate/run_training.py >> "$LOG_FILE" 2>&1
EXIT=$?
echo "=== $(date -Iseconds) Finished (exit $EXIT) ===" >> "$LOG_FILE"
exit $EXIT
