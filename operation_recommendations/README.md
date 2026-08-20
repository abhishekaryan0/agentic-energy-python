# Operation recommendations

ML + rules operational recommendations (Production / Optimization / Maintenance). Builds metrics from sensor + production (+ invoice) parquet, scores category/action models, estimates impact and implementation cost, prices uplift with org oil settings, and writes recommendation parquet to S3.

## Layout

| Path | Purpose |
|------|---------|
| `run.py` | Weekly checkpointed detection (`python -m operation_recommendations.run`) |
| `run_pipeline.py` | Broader local train + recommend pipeline (dev) |
| `train_models.py` | Train action/category models (`train_action_s3`, `train_action_parquet`) |
| `sagemaker_train.py` | SageMaker training entrypoint |
| `recommendations/analyzer.py` | `TrendAnalyzer` metrics from parquet |
| `recommendations/rules.py` | `RuleEngine`, `ImpactEstimator`, `ACTION_MAPPING` |
| `recommendations/persistence.py` | Recommend from metrics, economics, S3 write |
| `org_settings.py` | Read `{ORG}.org_settings` oil price type / manual value |
| `oil_price.py` | Resolve $/bbl (Manual / Brent / WTI) |
| `eia_prices.py` | EIA spot fetch for Brent / WTI |
| `tenant_config.py` | Per-tenant sensor lists (Apex subset vs Red Buff / STEP) |
| `s3_inputs.py` / `s3_models.py` | Parquet + model tarball loaders |
| `suggestions.py` | Optional OpenAI suggestion text |
| `config.py` | Thresholds, lift sensors, categories |
| `deploy/` | ECS Fargate detection + SageMaker train (Red Buff / Apex / STEP) |

## How it works

1. Load a date window of sensor + daily production partitions (default last **7** days; checkpoint under `{ORG}/detections/operations/checkpoints/`).
2. Build per-well metrics (slopes, rolling means, optional invoice rolls).
3. Apply rules + RF classifiers (`predict_category` / action models).
4. Estimate impact; cost from financial joblib (amortized); revenue upside from oil $/bbl.
5. Filter by confidence / min uplift; write `{ORG}/detections/operations/run_ts=…/operations_recommendations.parquet`.

### Oil price

`oil_price.resolve_oil_price_usd_per_bbl` reads Postgres `{ORGANIZATION_ID}.org_settings`:

| `oil_price_type` | Behavior |
|------------------|----------|
| `Manual` | Use `manual_value` |
| `Brent` / `WTI` | EIA via `EIA_API_KEY` (`eia_prices.py`) |
| Missing / DB error | Fallback Brent, then `OPERATIONS_OIL_PRICE_USD` (default **70**) |

Requires `DATABASE_URL` or `DATABASE_URL_SECRET_ARN` for org settings.

## Run locally

```bash
# Weekly detection (needs S3 models + partitions)
python -m operation_recommendations.run

# Train from S3 partitions
python -m operation_recommendations.train_models train_action_s3 --output-dir /tmp/ops_models

# Optional local full pipeline
python -m operation_recommendations.run_pipeline
```

**Useful env**

| Env | Default / notes |
|-----|-----------------|
| `S3_BUCKET`, `ORGANIZATION_ID` | Required |
| `OPS_DETECTION_DAYS` | `7` |
| `OPS_DETECTION_FORCE_WINDOW` | Force full window (ignore checkpoint) |
| `OPS_RESET_CHECKPOINT` | Treat as first run |
| `OPERATIONS_CONFIDENCE_THRESHOLD` | `0.6` (ECS templates may use `0.3`) |
| `OPERATIONS_MODEL_TARBALL_S3_URI` | Optional override; else latest `operations-train-*` |
| `FINANCIAL_MODEL_TARBALL_S3_URI` / `FINANCIAL_MODEL_PATH` | Cost model |
| `DATABASE_URL` / `DATABASE_URL_SECRET_ARN` | Org oil settings |
| `EIA_API_KEY` | Brent / WTI |
| `OPERATIONS_OIL_PRICE_USD` | Fallback `70` |
| `IMPLEMENTATION_LUMP_DAYS` / `IMPLEMENTATION_AMORTIZATION_DAYS` | Cost amortization |
| `OPS_MIN_PRODUCTION_INCREASE_BBL_DAY` | Min uplift filter |
| `OPERATIONS_TENANT_PROFILE` | Force `apex_permian` / `red_buff` / `step` |
| `OPENAI_*` | Optional suggestion enrichment |
| `OPERATIONS_RESEARCH_MODEL` | Citations model (default `perplexity/sonar-pro` via OpenRouter) |

## Deploy

```bash
# Weekly detection (ECS + EventBridge)
./operation_recommendations/deploy/schedule_detection.sh
./operation_recommendations/deploy/schedule_detection_apex.sh
./operation_recommendations/deploy/schedule_detection_step.sh

# SageMaker training
./operation_recommendations/deploy/run_sagemaker_training.sh
./operation_recommendations/deploy/schedule_sagemaker_training.sh
./operation_recommendations/deploy/run_sagemaker_training_apex.sh
./operation_recommendations/deploy/schedule_sagemaker_training_apex.sh
./operation_recommendations/deploy/run_sagemaker_training_step.sh
./operation_recommendations/deploy/schedule_sagemaker_training_step.sh
```

Detection loads joblibs from `s3://…/{ORG}/model/operations-train-…/output/model.tar.gz` (or `OPERATIONS_MODEL_TARBALL_S3_URI`). Training reads partitioned sensor data from S3.

## S3 layout

| Kind | Path |
|------|------|
| Input | `{ORG}/sensor_data/well_sensor_readings/date=…/` |
| Input | `{ORG}/sensor_data/daily_production/date=…/` |
| Input (optional) | `{ORG}/financial_data/run_ts=*/invoices_*.parquet` |
| Models | `{ORG}/model/operations-train-…/output/model.tar.gz` |
| Financial model | `{ORG}/model/financial-train-…/output/model.tar.gz` |
| Output | `{ORG}/detections/operations/run_ts=…/operations_recommendations.parquet` |
| Checkpoint | `{ORG}/detections/operations/checkpoints/last_processed.json` |

## Tenants

| Tenant | Org env | ECS scripts |
|--------|---------|-------------|
| Red Bluff | `RED_BUFF_ORGANIZATION_ID` | `schedule_detection.sh` |
| Apex Permian | `APEX_PERMIAN_ORGANIZATION_ID` | `*_apex.sh` (subset sensors, e.g. `freq_hz`) |
| South Texas Energy Partners | `STEP_ORGANIZATION_ID` | `*_step.sh` (Red Buff–style columns; `Pumping Unit` → Rod Pump) |

Downstream sync: `s3_to_postgres` → `{ORG}.operation_recommendations`.
