# Red Bluff Data Studio (web UI)

AI-style dashboard to run the incremental Snowflake pipeline and view dataset stats.

## Setup

From project root (uses root `.env` for Snowflake + `ORGANIZATION_ID`):

```bash
.venv/bin/pip install -r RED_BUFF/web/requirements.txt
```

## Run

```bash
cd /home/abhishek/Desktop/projects/agentic-energy-ml
.venv/bin/uvicorn RED_BUFF.web.app:app --reload --host 0.0.0.0 --port 8765
```

Open **http://localhost:8765**

## Usage

1. Wait for **Snowflake connected** (top right).
2. Set **target daily rows** (default 1000 ≈ 17 days × 60 wells).
3. Click **Generate batch** — runs `run_red_buff_pipeline` in the background (~2–5 min).
4. Review **Last run** metrics and **Snowflake dataset summary** (dates, row counts, sample wells).
