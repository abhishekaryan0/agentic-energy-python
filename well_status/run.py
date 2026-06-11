#!/usr/bin/env python3
"""CLI entrypoint for well status updates."""

import json
import logging
import sys

from dotenv import load_dotenv

from well_status.service import update_all_well_statuses

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    result = update_all_well_statuses()
    print(json.dumps(result, indent=2, default=str))
    status = result.get("status", "error")
    sys.exit(0 if status in ("success", "skipped") else 1)


if __name__ == "__main__":
    main()
