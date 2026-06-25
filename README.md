# Agentic Energy ML — ML models overview

This document summarizes which **Python models** are used under **anomaly detection**, **anomaly review**, and **operation recommendations**, and **why** they fit each use case.

---

## `anomaly_detection/`

Unified real-time-style anomaly scoring on well sensor data: rules from configuration, isolation forest, and an LSTM autoencoder, combined with weighted ensemble scoring.

| Component | Library / model | Role |
|-----------|-----------------|------|
| Rules | Sensor ranges from S3 parquet (`anomaly_detection.sensor_config`) | Domain min/max and bands; not a learned model |
| Outliers + scaling | **scikit-learn** `IsolationForest`, `StandardScaler` | Multivariate outliers vs typical operation |
| Sequence behavior | **PyTorch** LSTM autoencoder (`LSTMAutoencoder` in `anomaly_detection/detector.py`) | Pattern / reconstruction deviation on short windows |
| Baseline stats | Mean/std per feature (saved as `statistical_model.joblib`) | Supports z-score style checks with IF |

**Why:** Engineering rules encode known limits. Isolation Forest finds point-wise multivariate anomalies without labels. The LSTM autoencoder targets **temporal** patterns that scalar rules and isolation forest can miss. Training is described in `anomaly_detection/train_models.py` (per-well artifacts under `models/anomaly_detection/`).

Default ensemble weights (configurable in `AnomalyConfig`): rules, isolation forest, and LSTM each contribute to a single anomaly score.

---

## `anomaly_review/`

Middle layer for “unusual / worth human review” (forecast vs actual, reconstruction error, checkers). Not the same as a hard equipment alarm.

| Component | Library / model | Role |
|-----------|-----------------|------|
| Production baseline | **scikit-learn** `RandomForestRegressor` + `SimpleImputer` + `StandardScaler` | Tabular prediction of expected oil (and related flow) |
| Optional forecast | **PyTorch** `LSTMRegressor` | Sequence-based forecast when a trained LSTM is available |
| Pattern anomaly | **PyTorch** `LSTMAutoencoder` | High reconstruction error → unusual multivariate dynamics |
| Feature selection | Random Forest feature importance (`select_top_features_rf`) | Prioritize strong tabular drivers |

**Why:** Random Forest gives a robust **tabular** baseline with missing-value handling. LSTM regression uses **history** when data supports it. The LSTM autoencoder flags **normal-vs-abnormal pattern** divergence without requiring labeled anomalies. Orchestration lives in `anomaly_review/service.py`; model definitions in `anomaly_review/models.py`.

---

## `operation_recommendations/`

Operational **action / recommendation** models trained from parquet metrics (sensor + production), with labels derived from cohort z-score deviations mapped to action classes.

| Component | Library / model | Role |
|-----------|-----------------|------|
| Primary classifier | **scikit-learn** `RandomForestClassifier` | Multiclass recommended actions |
| Probabilities | Optional `CalibratedClassifierCV` around the RF | Better-calibrated probabilities when enough samples |
| Fallback | `DecisionTreeClassifier` | Used only if Random Forest is unavailable |

**Why:** Labels are **synthetic** (statistical deviations → action classes). Random Forest handles noisy tabular inputs, multiple classes, and imbalance (`class_weight='balanced'`). Calibration helps when downstream logic uses predicted probabilities. See `operation_recommendations/train_models.py`.

---

## Summary

| Area | Main Python models | Purpose |
|------|-------------------|---------|
| `anomaly_detection/` | Isolation Forest, LSTM autoencoder (PyTorch), rules + stats | Sensor-level anomaly ensemble |
| `anomaly_review/` | Random Forest regressor, LSTM regressor, LSTM autoencoder | Review-layer forecast vs actual and pattern anomalies |
| `operation_recommendations/` | Random Forest classifier (+ optional calibration) | Recommend operational actions from metrics |

There is no single model for the whole repository: each stack matches a different decision (alarm-style sensor score vs human review vs recommended action).

---

## S3 Parquet datasets (folder layout + fields)

This repo uses **org-scoped** S3 prefixes (tenant isolation).

- **Common S3 prefix**: `s3://$S3_BUCKET/$ORGANIZATION_ID/`
- **Primary sensor/prod export writer**: `transformations/` (see `transformations/README.md`, entrypoint `python -m transformations.run`)
- **Primary readers**
  - **Anomaly detection**: `anomaly_detection/sensor_data.py`
  - **Anomaly review (weekly, S3 only)**: `python -m anomaly_review.run`
  - **Operations detection (weekly, S3 only)**: `python -m operation_recommendations.run`
- **Master data (digital twin) writer**: `APEX_PERMIAN/generate_master.py`

### Sensor data (Snowflake → S3 parquet)

#### Folder layout

- **Partitioned (preferred)**
  - `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/well_sensor_readings/date=YYYY-MM-DD/part-0000.parquet`
  - `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/daily_production/date=YYYY-MM-DD/part-0000.parquet`
- **Legacy root snapshots (optional / fallback readers)**
  - `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/well_sensor_readings.parquet`
  - `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/daily_production.parquet`
- **Export checkpoint**
  - `s3://$S3_BUCKET/$ORGANIZATION_ID/sensor_data/_state/export_checkpoint.json`

#### `well_sensor_readings` parquet fields (columns)

These are the canonical columns in `transformations/columns.py` (`WELL_SENSOR_READINGS_COLUMNS`) plus one alias used by the synthetic twin scripts.

- **Identity / time**
  - `organization_id`
  - `well_id`
  - `timestamp`
  - `lift_type`
  - `well_status`
- **Rod pump / mechanical**
  - `strokes_per_minute`
  - `torque`
  - `polish_rod_load`
  - `pump_fillage`
  - `surface_stroke_length`
  - `downhole_gross_stroke`
  - `downhole_net_stroke`
  - `runtime`
  - `cycles`
  - `structural_load`
  - `inferred_production`
  - `pump_intake_pressure`
  - `dry_rod_weight`
  - `buoyant_rod_weight`
  - `pump_friction`
  - `pump_diameter`
  - `barrel_length`
  - `pump_depth`
  - `idle_time_setpoint`
- **ESP / electrical**
  - `motor_temp`
  - `motor_current`
  - `discharge_pressure`
  - `motor_voltage`
  - `intake_fluid_temp`
  - `vibration_x`
  - `vibration_y`
  - `discharge_temp`
  - `downhole_flow_rate`
  - `drive_frequency`
  - `tubing_pressure`
  - `casing_pressure`
  - `drive_input_voltage`
  - `output_voltage`
  - `input_current`
  - `motor_speed`
  - `vsd_temp`
  - `total_harmonic_distortion`
  - `motor_load`
  - `set_frequency`
  - `acceleration_ramp_time`
- **Gas lift / plunger-lift style**
  - `injection_rate`
  - `injection_temperature`
  - `bottomhole_pressure`
  - `injection_pressure`
  - `cycle_time`
  - `bottomhole_temp`
  - `plunger_arrival_time`
  - `plunger_velocity`
  - `arrival_count`
  - `missed_arrivals`
  - `shutin_time`
  - `afterflow_time`
  - `flow_time`
  - `plunger_drop_time`
  - `min_shutin_pressure`
  - `max_shutin_pressure`
  - `open_diff_pressure`
  - `well_open`
  - `well_close`
  - `velocity_limit`
  - `wellhead_temp`
  - `surface_pressure`
- **Script compatibility alias**
  - `freq_hz`

`freq_hz` is written/expected by `APEX_PERMIAN/generate_digital_twin.py` and is treated as an alias of `drive_frequency` in `transformations/schemas.py`.

#### `daily_production` parquet fields (columns)

From `transformations/columns.py` (`WELL_DAILY_PRODUCTION_COLUMNS`):

- `organization_id`
- `well_id`
- `date`
- `oil_volume`
- `gas_volume`
- `water_volume`
- `boe`
- `lift_type`
- `status`
- `notes`
- `test_oil`
- `test_water`
- `test_gas`
- `facility_id`

### Master data (digital twin → S3 parquet)

Generated by `APEX_PERMIAN/generate_master.py` (intended as “master / reference” data used by demo UIs and some downstream workflows).

#### Folder layout

- `s3://$S3_BUCKET/$ORGANIZATION_ID/digital_twin/master_data/run_ts=YYYYMMDDThhmmssZ/master_assets.parquet`
- `s3://$S3_BUCKET/$ORGANIZATION_ID/digital_twin/master_data/run_ts=YYYYMMDDThhmmssZ/master_history.parquet`

#### `master_assets.parquet` fields

Written by `_upload_master_parquet_to_s3()` in `APEX_PERMIAN/generate_master.py`:

- `organization_id`
- `asset_id`
- `asset_type` (e.g. `Well`, `Facility`)
- `lift_type` (only for wells)
- `assigned_personnel_json` (JSON string)
- `financial_ledger_json` (JSON string)
- `raw_asset_json` (full JSON blob as string)

#### `master_history.parquet` fields

Written by `_upload_master_parquet_to_s3()` in `APEX_PERMIAN/generate_master.py`:

- **Identity**
  - `organization_id`
  - `asset_id`
  - `asset_type`
  - `lift_type` (only for wells)
  - `date`
  - `status`
  - `notes`
- **Facility history fields**
  - `oil_prod`
- **Well history fields**
  - `allocated_oil`
  - `test_oil`
  - `avg_tubing_pressure`
  - `avg_casing_pressure`
  - `avg_motor_amps`
  - `gross_stroke_len`
  - `net_stroke_len`
  - `spm`
  - `pump_fillage_pct`
  - `freq_hz`
  - `pump_intake_pressure`
  - `motor_temp_f`
  - `injection_rate_mcf`
- **Raw history payload**
  - `raw_history_json` (full JSON blob as string)
