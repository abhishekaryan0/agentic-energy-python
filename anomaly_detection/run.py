import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from anomaly_detection.bulk_detect import run_bulk_anomaly_detection
from anomaly_detection.bulk_detect import run_incremental_anomaly_detection
from anomaly_detection.bulk_detect import run_recent_rows_anomaly_detection


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    mode = (os.getenv("ANOMALY_DETECTION_MODE") or "incremental").strip().lower()
    max_wells = os.getenv("ANOMALY_BULK_MAX_WELLS")
    max_wells_int = int(max_wells) if max_wells else None

    if mode == "bulk":
        bulk_default = str(max(int(os.getenv("ANOMALY_LSTM_SEQ_LEN", "24") or 24) * 2, 48))
        last_n = int(os.getenv("ANOMALY_BULK_LAST_N", bulk_default))
        result = run_bulk_anomaly_detection(last_n=last_n, per_well=True, max_wells=max_wells_int)
    elif mode == "recent":
        last_rows = int(os.getenv("ANOMALY_LAST_ROWS", "100"))
        result = run_recent_rows_anomaly_detection(last_rows=last_rows, max_wells=max_wells_int)
    else:
        window_minutes = int(os.getenv("ANOMALY_WINDOW_MINUTES", "15"))
        safety_lag = int(os.getenv("ANOMALY_SAFETY_LAG_SECONDS", "90"))
        result = run_incremental_anomaly_detection(
            window_minutes=window_minutes,
            safety_lag_seconds=safety_lag,
            max_wells=max_wells_int,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
