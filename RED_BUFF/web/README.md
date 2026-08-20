# Red Bluff Data Studio (web UI)

Local FastAPI UI to run the incremental Snowflake pipeline and inspect dataset stats. **Dev only** — not used by the production `red-buff` Lambda.

## Setup

From project root (uses root `.env` for Snowflake + `ORGANIZATION_ID`):

```bash
.venv/bin/pip install -r RED_BUFF/web/requirements.txt
```

## Run

```bash
.venv/bin/uvicorn RED_BUFF.web.app:app --reload --host 0.0.0.0 --port 8765
```

Open **http://localhost:8765**

## Usage

1. Wait for **Snowflake connected** (health check).
2. Click **Generate** — runs `run_red_buff_pipeline` in the background (incremental from watermark through today, or full backfill if empty).
3. Use **Reset** to wipe this org’s Snowflake data (then regenerate).
4. Review job status and dataset summary (dates, row counts, sample wells).

Generation is **window-based** (checkpoint → today), not a configurable “target daily rows” count.

## API

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/` | UI |
| GET | `/api/health` | Snowflake connectivity |
| GET | `/api/stats` | Counts, date ranges, preview |
| POST | `/api/generate` | Background pipeline → `job_id` |
| POST | `/api/reset` | Clear org Snowflake data |
| GET | `/api/jobs/{job_id}` | Job status |
