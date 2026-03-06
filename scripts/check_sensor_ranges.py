#!/usr/bin/env python3
"""
Check well sensor readings against SENSOR_DEFINITIONS min/max ranges.

Counts how many values in well_sensor_readings fall outside the configured
range for each sensor (per lift type). Null/NaN values are ignored (not
counted in checks or out-of-range). Uses config.sensor_config which reads
sensor_definitions (min_value, max_value) from the database.

Usage:
  python scripts/check_sensor_ranges.py
  python scripts/check_sensor_ranges.py --days 30
  python scripts/check_sensor_ranges.py --well-id WELL_001
  python scripts/check_sensor_ranges.py --limit 10000 --verbose
"""

import argparse
import logging
import sys
from collections import defaultdict

# Allow running from repo root or from scripts/
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_repo_root = _os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from config.sensor_config import (
    get_db_connection,
    get_lift_types,
    get_sensors_for_lift_type,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_sensor_ranges_by_lift_type():
    """
    Build a map: lift_type_name -> { field_name: (min, max) }
    Only includes sensors that have both min and max defined.
    """
    lift_types = get_lift_types()
    ranges_by_lift = {}
    for lift_type_name in lift_types.keys():
        sensors = get_sensors_for_lift_type(lift_type_name)
        ranges = {}
        for s in sensors:
            min_val, max_val = s.get("min"), s.get("max")
            if min_val is not None and max_val is not None:
                ranges[s["field_name"]] = (min_val, max_val)
        ranges_by_lift[lift_type_name] = ranges
    return ranges_by_lift


def _is_null_or_nan(value):
    """True if value should be ignored (null or NaN)."""
    if value is None:
        return True
    try:
        v = float(value)
        return v != v  # NaN
    except (TypeError, ValueError):
        return True


def is_out_of_range(value, min_val, max_val):
    """Return True if value is outside [min_val, max_val]. Null/NaN are ignored (not checked)."""
    if _is_null_or_nan(value):
        return False
    v = float(value)
    return v < min_val or v > max_val


def fetch_readings(conn, days=None, well_id=None, limit=None):
    """Fetch well_sensor_readings. Columns normalized to lowercase."""
    cursor = conn.cursor()
    params = []
    where_parts = ["1=1"]
    if well_id:
        where_parts.append("well_id = %s")
        params.append(well_id)
    if days is not None:
        where_parts.append("timestamp >= DATEADD(day, -%s, CURRENT_TIMESTAMP())")
        params.append(days)
    where_sql = " AND ".join(where_parts)
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    query = f"""
        SELECT *
        FROM well_sensor_readings
        WHERE {where_sql}
        ORDER BY well_id, timestamp
        {limit_sql}
    """
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cols = [c[0].lower() for c in cursor.description] if cursor.description else []
    cursor.close()
    return [dict(zip(cols, row)) for row in rows]


def run_check(days=None, well_id=None, limit=None, verbose=False):
    """Check all readings against sensor ranges and return stats."""
    ranges_by_lift = get_sensor_ranges_by_lift_type()
    total_sensors_with_range = sum(len(r) for r in ranges_by_lift.values())
    logger.info(
        "Sensor ranges loaded for lift types: %s (total fields with min/max: %s)",
        list(ranges_by_lift.keys()),
        total_sensors_with_range,
    )

    conn = get_db_connection()
    try:
        readings = fetch_readings(conn, days=days, well_id=well_id, limit=limit)
    finally:
        conn.close()

    n_readings = len(readings)
    logger.info("Fetched %s readings", n_readings)
    if n_readings == 0:
        return {
            "total_readings": 0,
            "total_checks": 0,
            "total_out_of_range": 0,
            "by_well": {},
            "by_field": defaultdict(int),
            "by_lift_type": defaultdict(int),
            "examples": [],
        }

    total_checks = 0
    total_out_of_range = 0
    by_well = defaultdict(int)
    by_field = defaultdict(int)
    by_lift_type = defaultdict(int)
    examples = []  # (well_id, timestamp, field_name, value, min, max)

    for row in readings:
        lift_type = row.get("lift_type")
        if not lift_type:
            continue
        lift_type = str(lift_type).strip()
        ranges = ranges_by_lift.get(lift_type)
        if not ranges:
            continue
        well_id = row.get("well_id")
        ts = row.get("timestamp")
        for field_name, (min_val, max_val) in ranges.items():
            value = row.get(field_name)
            if _is_null_or_nan(value):
                continue  # ignore null/NaN – don't count in checks or out-of-range
            total_checks += 1
            if is_out_of_range(value, min_val, max_val):
                total_out_of_range += 1
                by_well[well_id] += 1
                by_field[field_name] += 1
                by_lift_type[lift_type] += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "well_id": well_id,
                            "timestamp": str(ts),
                            "field_name": field_name,
                            "value": value,
                            "min": min_val,
                            "max": max_val,
                        }
                    )

    return {
        "total_readings": n_readings,
        "total_checks": total_checks,
        "total_out_of_range": total_out_of_range,
        "by_well": dict(by_well),
        "by_field": dict(by_field),
        "by_lift_type": dict(by_lift_type),
        "examples": examples,
        "verbose": verbose,
    }


def print_report(stats):
    """Print a human-readable report."""
    print("\n" + "=" * 60)
    print("WELL SENSOR READINGS — OUT OF RANGE CHECK")
    print("(vs SENSOR_DEFINITIONS min_value / max_value)")
    print("=" * 60)
    print(f"Total readings processed:     {stats['total_readings']:,}")
    print(f"Total value checks (sensor):  {stats['total_checks']:,}")
    print(f"Total out-of-range values:    {stats['total_out_of_range']:,}")
    if stats["total_checks"] > 0:
        pct = 100.0 * stats["total_out_of_range"] / stats["total_checks"]
        print(f"Out-of-range percentage:       {pct:.2f}%")
    print()

    if stats["by_well"]:
        print("--- By well_id (top 20) ---")
        sorted_wells = sorted(
            stats["by_well"].items(), key=lambda x: -x[1]
        )[:20]
        for wid, count in sorted_wells:
            print(f"  {wid}: {count:,} out-of-range")
        print()

    if stats["by_field"]:
        print("--- By sensor field (all) ---")
        for field, count in sorted(stats["by_field"].items(), key=lambda x: -x[1]):
            print(f"  {field}: {count:,}")
        print()

    if stats["by_lift_type"]:
        print("--- By lift_type ---")
        for lt, count in sorted(
            stats["by_lift_type"].items(), key=lambda x: -x[1]
        ):
            print(f"  {lt}: {count:,}")
        print()

    if stats["examples"]:
        print("--- Sample out-of-range rows (up to 20) ---")
        for ex in stats["examples"]:
            print(
                f"  well={ex['well_id']} ts={ex['timestamp']} "
                f"field={ex['field_name']} value={ex['value']} "
                f"range=[{ex['min']}, {ex['max']}]"
            )
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check well sensor readings against sensor definition min/max ranges"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only consider readings from the last N days",
    )
    parser.add_argument(
        "--well-id",
        type=str,
        default=None,
        help="Restrict to a single well_id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of readings to fetch (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include more detail in output",
    )
    args = parser.parse_args()

    stats = run_check(
        days=args.days,
        well_id=args.well_id,
        limit=args.limit,
        verbose=args.verbose,
    )
    print_report(stats)

    # Exit with non-zero if any out-of-range values found (useful for CI)
    if stats["total_out_of_range"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
