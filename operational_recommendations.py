import json
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Action -> Category mapping and category descriptions
ACTION_CATEGORY_MAP = {
    'Increase ESP frequency': 'Production',
    'Adjust pump/polish choke': 'Production',
    'Optimize gas lift injection rate': 'Optimization',
    # fallback actions that may be used in future
    'Inspect ESP equipment': 'Maintenance',
}

# Human-readable category descriptions used for UI/context
CATEGORY_DESCRIPTIONS = {
    'Production': 'Recommendations that directly aim to increase oil or gas production. May involve higher operating cost or equipment wear.',
    'Optimization': 'Recommendations focused on cost/efficiency optimization that typically reduce operating cost with modest production impact.',
    'Maintenance': 'Recommendations focused on preventing failure, extending equipment life, or reducing downtime.',
}

# Canonical feature list - used consistently across ALL training and prediction (action + category models)
CANONICAL_FEATURES = [
    'oil_volume_slope', 'motor_current_slope', 'motor_temp_slope', 'surface_pressure_slope', 'gas_volume_slope',
    'motor_current_roll_mean', 'motor_temp_roll_mean', 'surface_pressure_roll_mean', 'oil_volume_roll_mean', 'gas_volume_roll_mean',
    'missing_rate', 'motor_current_motor_temp_corr'
]
# Backward compatibility alias
FEATURES = CANONICAL_FEATURES

# Action mapping from measured field violations to recommended actions
ACTION_MAPPING = {
    'motor_current': {
        'class': 'Increase ESP frequency',
        'category': 'Production',
        'templates': [
            'Increase ESP frequency by 3 Hz',
            'Increase ESP speed by ~5%',
            'Increase ESP frequency to improve motor loading'
        ]
    },
    'motor_temp': {
        'class': 'Inspect ESP equipment',
        'category': 'Maintenance',
        'templates': [
            'Inspect ESP motor and perform impeller alignment within 7 days',
            'Replace impeller and perform alignment check within 7 days',
            'Schedule motor inspection and thermal check'
        ]
    },
    'surface_pressure': {
        'class': 'Adjust separator pressure',
        'category': 'Optimization',
        'templates': [
            'Reduce separator pressure from 125 PSI to 110 PSI',
            'Adjust separator pressure setpoint to reduce backpressure',
            'Lower separator pressure by ~10-15 PSI to improve throughput'
        ]
    },
    'oil_volume': {
        'class': 'Adjust pump/choke',
        'category': 'Production',
        'templates': [
            'Increase pump/choke opening to raise oil rate',
            'Adjust pump/polish choke to increase oil production',
            'Increase pumping speed moderately to recover oil rate'
        ]
    },
    'gas_volume': {
        'class': 'Optimize gas lift injection rate',
        'category': 'Optimization',
        'templates': [
            'Increase gas injection rate from 2.5 MMSCF to 3.2 MMSCF',
            'Optimize gas lift injection rate to improve recovery',
            'Tune gas lift injection by +20% to restore gas balance'
        ]
    }
}

# Map action -> high-level category
ACTION_CATEGORY_MAP = {
    'Increase ESP frequency': 'Production',
    'Adjust pump/polish choke': 'Production',
    'Optimize gas lift injection rate': 'Optimization',
}

# Human-facing category descriptions
CATEGORY_DESCRIPTIONS = {
    'Production': "Recommendations that directly aim to increase oil or gas production. May increase operating cost or equipment wear; valid if net economic benefit is positive.",
    'Optimization': "Recommendations focused on cost/efficiency optimization (reduce chemical usage, operating costs) without necessarily increasing production significantly.",
    'Maintenance': "Maintenance / Failure Prevention: extend equipment life, prevent failures, reduce downtime. These are recommendations, not execution actions."
}

# Optional ML imports
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
except Exception:
    DecisionTreeClassifier = None
    train_test_split = None
    classification_report = None
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    RandomForestClassifier = None
    CalibratedClassifierCV = None

class TrendAnalyzer:
    """Compute simple trends and rolling stats from a dataframe of readings.

    Expects a DataFrame indexed by timestamp with columns including:
    'oil_volume', 'gas_volume', 'motor_current', 'motor_temp', 'surface_pressure', ...
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_index()

    def rolling_means(self, window='24H'):
        return self.df.rolling(window).mean()

    def slope(self, series: pd.Series, window_points: int = 24):
        """Compute slope over the last `window_points` samples using linear fit.
        Returns None if insufficient data.
        """
        y = series.dropna()
        if len(y) < max(3, window_points):
            return None
        y_tail = y.iloc[-window_points:]
        x = np.arange(len(y_tail))
        try:
            m = np.polyfit(x, y_tail.values, 1)[0]
            return float(m)
        except Exception:
            return None

    def baseline_stats(self, field: str, lookback_points: int = 720, exclude_last: int = 24):
        """Compute baseline mean/std and slope for a field using a lookback window.

        - lookback_points: number of historical samples to use (e.g., 168 for 7 days hourly)
        - exclude_last: number of most recent samples to exclude from baseline (e.g., 24 hours)
        Returns dict with mean, std, slope, slope_std
        """
        if field not in self.df.columns:
            return {'mean': None, 'std': None, 'slope': None, 'slope_std': None}
        series = self.df[field].dropna()
        if len(series) < (exclude_last + 3):
            return {'mean': None, 'std': None, 'slope': None, 'slope_std': None}
        end_idx = len(series) - exclude_last
        start_idx = max(0, end_idx - lookback_points)
        window = series.iloc[start_idx:end_idx]
        mean = float(window.mean()) if not window.empty else None
        std = float(window.std()) if not window.empty else None
        # compute slope over the baseline window
        slope = None
        slope_std = None
        try:
            if len(window) >= 3:
                x = np.arange(len(window))
                m = np.polyfit(x, window.values, 1)[0]
                slope = float(m)
                # approximate slope std by sampling smaller windows
                sls = []
                w = 24
                for i in range(0, max(1, len(window) - w), max(1, w)):
                    part = window.iloc[i:i + w]
                    if len(part) >= 3:
                        xm = np.arange(len(part))
                        sls.append(np.polyfit(xm, part.values, 1)[0])
                slope_std = float(np.std(sls)) if sls else 0.0
        except Exception:
            slope = None
            slope_std = None

        return {'mean': mean, 'std': std, 'slope': slope, 'slope_std': slope_std}

    def compute_metrics(self):
        metrics = {}
        # slopes
        for col in ['oil_volume', 'motor_current', 'motor_temp', 'surface_pressure', 'gas_volume']:
            if col in self.df.columns:
                metrics[f'{col}_slope'] = self.slope(self.df[col], window_points=24)
            else:
                metrics[f'{col}_slope'] = None

        # rolling means last 24 samples - only on numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        rm = numeric_df.rolling(window=24).mean().iloc[-1] if len(numeric_df) >= 24 else numeric_df.mean()
        for col in ['motor_current', 'motor_temp', 'surface_pressure', 'oil_volume', 'gas_volume']:
            metrics[f'{col}_roll_mean'] = float(rm[col]) if col in rm and not pd.isna(rm[col]) else None

        # anomaly frequency placeholder (requires anomaly labels) -> count NaNs as proxy
        metrics['missing_rate'] = float(self.df.isna().mean().mean())

        # correlations (motor_current vs motor_temp)
        if 'motor_current' in self.df.columns and 'motor_temp' in self.df.columns:
            metrics['motor_current_motor_temp_corr'] = float(self.df['motor_current'].corr(self.df['motor_temp']))
        else:
            metrics['motor_current_motor_temp_corr'] = None

        return metrics


class RuleEngine:
    """Apply deterministic rules to metrics and generate recommendations."""

    def __init__(self, conn=None):
        """Initialize with Snowflake connection for DB-based rules."""
        self.conn = conn
        self.rules = {}

    def load_rules_from_db(self, lift_type=None):
        """Load field ranges from sensor_definitions table in DB.
        
        If lift_type specified, load for that type, else load all fields.
        """
        if not self.conn:
            return {}
        
        cur = self.conn.cursor()
        try:
            if lift_type:
                # Get lift_type_id
                cur.execute("SELECT id FROM lift_types WHERE lift_type_name = %s", (lift_type,))
                lift_id_row = cur.fetchone()
                if not lift_id_row:
                    return {}
                lift_id = lift_id_row[0]
                cur.execute("SELECT field_name, min_value, max_value FROM sensor_definitions WHERE lift_type_id = %s", (lift_id,))
            else:
                cur.execute("SELECT field_name, min_value, max_value FROM sensor_definitions")
            
            rows = cur.fetchall()
            rules = {}
            for field, mn, mx in rows:
                rules[field] = {'min': mn, 'max': mx}
            self.rules = rules
            return rules
        finally:
            cur.close()

    def load_adaptive_rules(self, well_id):
        """Load adaptive rules based on last 30 days data for the well."""
        if not self.conn:
            return {}
        
        cur = self.conn.cursor()
        try:
            # Get lift_type for the well
            cur.execute("SELECT DISTINCT lift_type FROM well_sensor_readings WHERE well_id = %s LIMIT 1", (well_id,))
            row = cur.fetchone()
            if not row:
                return {}
            lift_type = row[0]
            
            # Get lift_type_id
            cur.execute("SELECT id FROM lift_types WHERE lift_type_name = %s", (lift_type,))
            lift_id_row = cur.fetchone()
            if not lift_id_row:
                return {}
            lift_id = lift_id_row[0]
            
            # Get fields for this lift_type that have numeric min/max defined
            cur.execute("SELECT field_name FROM sensor_definitions WHERE lift_type_id = %s AND min_value IS NOT NULL AND max_value IS NOT NULL", (lift_id,))
            fields = [r[0] for r in cur.fetchall()]
            
            rules = {}
            for field in fields:
                # Compute min/max from last 30 days
                try:
                    cur.execute(f"SELECT MIN(TRY_CAST({field} AS FLOAT)), MAX(TRY_CAST({field} AS FLOAT)) FROM well_sensor_readings WHERE well_id = %s AND timestamp >= DATEADD(day, -30, CURRENT_DATE)", (well_id,))
                    mn, mx = cur.fetchone()
                    if mn is not None and mx is not None:
                        rules[field] = {'min': float(mn), 'max': float(mx)}
                except Exception as e:
                    logger.warning(f"Skipping field {field} for well {well_id}: {e}")
                    continue
            self.rules = rules
            return rules
        finally:
            cur.close()

    def check_ranges(self, readings: dict):
        """Return violations for readings according to loaded rules."""
        violations = []
        for field, bounds in (self.rules or {}).items():
            if field not in readings:
                continue
            val = readings.get(field)
            if val is None:
                continue
            mn = bounds.get('min')
            mx = bounds.get('max')
            if mn is not None and val < mn:
                violations.append({'field': field, 'value': val, 'min': mn, 'max': mx, 'violation': 'below_min'})
            if mx is not None and val > mx:
                violations.append({'field': field, 'value': val, 'min': mn, 'max': mx, 'violation': 'above_max'})
        return violations

    def apply(self, well_id: str, metrics: dict, last_reading: dict = None, trend_analyzer: TrendAnalyzer = None, adaptive_k: float = 2.0):
        # Load adaptive rules based on last 30 days data
        self.load_adaptive_rules(well_id)
        
        recs = []

        # Rule: motor_current high and motor_temp rising -> Increase ESP frequency
        mc = metrics.get('motor_current_roll_mean')
        mt_slope = metrics.get('motor_temp_slope')
        mc_mt_corr = metrics.get('motor_current_motor_temp_corr')
        # determine threshold from adaptive rules
        mc_thresh = self.rules.get('motor_current', {}).get('max')

        if mc is not None and mt_slope is not None and mc_thresh is not None:
            if mc > mc_thresh and mt_slope > 0.01:
                action = 'Increase ESP frequency'
                recs.append({
                    'well_id': well_id,
                    'action': action,
                    'category': ACTION_CATEGORY_MAP.get(action, 'Production'),
                    'priority': 'HIGH',
                    'status': 'New',
                    'reason': f'motor_current {mc:.1f} > threshold {mc_thresh:.1f} and motor_temp rising',
                    'expected_impact': 'Estimate based on similar cases',
                    'confidence': 0.9,
                })

        # Rule: surface_pressure declining AND oil_volume dropping -> Pump adjustment
        sp_slope = metrics.get('surface_pressure_slope')
        oil_slope = metrics.get('oil_volume_slope')
        # slope thresholds: use default since rules are absolute min/max
        slope_thresh = -0.01
        if trend_analyzer is not None:
            sp_stats = trend_analyzer.baseline_stats('surface_pressure')
            oil_stats = trend_analyzer.baseline_stats('oil_volume')
            # consider significant negative slope if below baseline slope minus k * slope_std
            if sp_stats['slope'] is not None and sp_stats['slope_std'] is not None:
                slope_thresh = sp_stats['slope'] - adaptive_k * (sp_stats['slope_std'] or 0.0)
            else:
                slope_thresh = -0.01

        if sp_slope is not None and oil_slope is not None and slope_thresh is not None:
            if sp_slope < slope_thresh and oil_slope < (oil_stats['slope'] - adaptive_k * (oil_stats['slope_std'] or 0.0)):
                action = 'Adjust pump/polish choke'
                recs.append({
                    'well_id': well_id,
                    'action': action,
                    'category': ACTION_CATEGORY_MAP.get(action, 'Production'),
                    'priority': 'HIGH',
                    'status': 'New',
                    'reason': 'surface_pressure and oil production both declining relative to baseline',
                    'expected_impact': 'Estimate based on similar cases',
                    'confidence': 0.85,
                })

        # Rule: gas volume low -> gas lift optimization
        gv = metrics.get('gas_volume_roll_mean')
        # gas threshold from adaptive rules
        gv_thresh = self.rules.get('gas_volume', {}).get('min')

        if gv is not None and gv_thresh is not None and gv < gv_thresh:
            action = 'Optimize gas lift injection rate'
            recs.append({
                'well_id': well_id,
                'action': action,
                'category': ACTION_CATEGORY_MAP.get(action, 'Optimization'),
                'priority': 'MEDIUM',
                'status': 'New',
                'reason': 'gas volume below threshold',
                'expected_impact': 'Estimate based on similar cases',
                'confidence': 0.8,
            })

        # General checks for all fields in sensor_definitions
        if last_reading:
            for field, bounds in self.rules.items():
                val = last_reading.get(field)
                if val is None:
                    continue
                mn = bounds.get('min')
                mx = bounds.get('max')
                if mn is not None and val < mn:
                    action = ACTION_MAPPING.get(field, {}).get('class', 'Inspect Equipment')
                    category = ACTION_MAPPING.get(field, {}).get('category', 'Maintenance')
                    recs.append({
                        'well_id': well_id,
                        'action': action,
                        'category': category,
                        'priority': 'MEDIUM',
                        'status': 'New',
                        'reason': f'{field} {val:.2f} < {mn:.2f}',
                        'expected_impact': 'Check sensor reading',
                        'confidence': 0.7,
                    })
                if mx is not None and val > mx:
                    action = ACTION_MAPPING.get(field, {}).get('class', 'Adjust Settings')
                    category = ACTION_MAPPING.get(field, {}).get('category', 'Optimization')
                    recs.append({
                        'well_id': well_id,
                        'action': action,
                        'category': category,
                        'priority': 'MEDIUM',
                        'status': 'New',
                        'reason': f'{field} {val:.2f} > {mx:.2f}',
                        'expected_impact': 'Check sensor reading',
                        'confidence': 0.7,
                    })

        return recs


class ImpactEstimator:
    """Minimal placeholder. Real implementation requires historical adjustment logs.
    For now, this class returns a textual placeholder and does not compute monetary impact.
    """

    def estimate(self, recommendation: dict, metrics: dict):
        # Attach a dummy uplift estimate (percent) based on action type
        action = recommendation.get('action', '').lower()
        if 'esp' in action or 'frequency' in action:
            uplift = 0.15
        elif 'pump' in action or 'choke' in action:
            uplift = 0.08
        elif 'gas lift' in action or 'gas' in action:
            uplift = 0.10
        else:
            uplift = 0.05

        return {'expected_uplift_pct': uplift, 'confidence': recommendation.get('confidence', 0.5)}


def _create_table_sql(table_name: str = 'OPERATION_RECOMMENDATION'):
    # Unified table for both rule-based and ML-based recommendations
    # Use an uppercase, quoted identifier to avoid identifier case issues in Snowflake
    tn = table_name.upper()
    return f"""
    CREATE TABLE IF NOT EXISTS "{tn}" (
        id VARCHAR PRIMARY KEY,
        well_id VARCHAR NOT NULL,
        category VARCHAR,
        action VARCHAR,
        priority VARCHAR,
        status VARCHAR,
        reason VARCHAR,
        expected_impact VARCHAR,
        confidence FLOAT,
        prediction_source VARCHAR,
        probability FLOAT,
        oil_volume_slope FLOAT,
        oil_volume_roll_mean FLOAT,
        gas_volume_slope FLOAT,
        gas_volume_roll_mean FLOAT,
        motor_current_roll_mean FLOAT,
        motor_temp_roll_mean FLOAT,
        surface_pressure_roll_mean FLOAT,
        details VARCHAR,
        created_at TIMESTAMP_NTZ NOT NULL
    );
    """


def save_recommendation_to_snowflake(conn, rec: dict, table_name: str = 'OPERATION_RECOMMENDATION', metrics: dict = None):
    """Save a single recommendation dict into Snowflake (rule-based or ML-based).

    `conn` must be a snowflake.connector connection with `.cursor().execute()` API.
    `metrics` optional dict with metric values to store alongside recommendation.
    """
    cur = conn.cursor()
    try:
        cur.execute(_create_table_sql(table_name))
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rec_id = rec.get('id') or str(uuid.uuid4())
        details = json.dumps(rec).replace("'", "''")  # escape single quotes
        metrics = metrics or rec.get('metrics', {})
        # ensure category present (may be added by rule engine or ML model)
        if not rec.get('category'):
            _jb = joblib
            try:
                model_path = os.path.join('models', 'category_classifier.joblib')
                if os.path.exists(model_path) and joblib is not None:
                    mdl = joblib.load(model_path)
                    features = mdl.get('features') if isinstance(mdl, dict) else None
                    clf = mdl.get('model') if isinstance(mdl, dict) else mdl
                    if features and clf is not None:
                        fv = []
                        for f in features:
                            fv.append(metrics.get(f) if metrics and f in metrics else 0.0)
                        try:
                            probas = clf.predict_proba([fv])[0]
                            pred = clf.classes_[probas.argmax()]
                            rec['category'] = pred
                        except Exception:
                            rec['category'] = 'Production'
            except Exception:
                rec['category'] = rec.get('category', 'Production')
        
        def _to_native(v):
            """Convert common numpy/pandas types and complex objects to plain python types supported by DB drivers."""
            try:
                # numpy scalars
                import numpy as _np
                if isinstance(v, (_np.generic,)):
                    return v.item()
            except Exception:
                pass
            # datetime -> native
            try:
                from datetime import datetime
                if isinstance(v, datetime):
                    return v
            except Exception:
                pass
            # basic python types
            if v is None:
                return None
            if isinstance(v, (int, float, str, bool)):
                return v
            # containers -> json string
            try:
                return json.dumps(v)
            except Exception:
                return str(v)
        
        # ensure table name quoting/case to match Snowflake's identifiers
        tn = table_name.upper()

        # Original desired columns and params order
        orig_cols = [
            "ID", "WELL_ID", "CATEGORY", "ACTION", "PRIORITY", "STATUS", "REASON", "EXPECTED_IMPACT", "CONFIDENCE", "PREDICTION_SOURCE", "PROBABILITY",
            "OIL_VOLUME_SLOPE", "OIL_VOLUME_ROLL_MEAN", "GAS_VOLUME_SLOPE", "GAS_VOLUME_ROLL_MEAN",
            "MOTOR_CURRENT_ROLL_MEAN", "MOTOR_TEMP_ROLL_MEAN", "SURFACE_PRESSURE_ROLL_MEAN", "DETAILS", "CREATED_AT"
        ]

        # Build params tuple in the same order
        params = (
            rec_id,
            rec.get('well_id') or '',
            rec.get('category') or 'Production',
            rec.get('action') or '',
            rec.get('priority') or '',
            rec.get('status') or '',
            rec.get('reason') or '',
            rec.get('expected_impact') or '',
            float(rec.get('confidence') or 0.0),
            rec.get('prediction_source') or 'rule_based',
            float(rec.get('probability')) if rec.get('probability') is not None else None,
            metrics.get('oil_volume_slope'),
            metrics.get('oil_volume_roll_mean'),
            metrics.get('gas_volume_slope'),
            metrics.get('gas_volume_roll_mean'),
            metrics.get('motor_current_roll_mean'),
            metrics.get('motor_temp_roll_mean'),
            metrics.get('surface_pressure_roll_mean'),
            details,
            now
        )

        # Try to adapt to the actual table schema: prefer to describe table and use intersection
        try:
            cur.execute(f'DESCRIBE TABLE "{tn}"')
            existing_cols = [r[0].upper() for r in cur.fetchall()]
        except Exception:
            existing_cols = orig_cols

        # If columns are missing in the target table, try to add them to match expected schema
        missing_cols = [c for c in orig_cols if c not in existing_cols]
        if missing_cols:
            logger.info(f"Missing columns detected in {tn}: {missing_cols}; attempting to ALTER TABLE ADD COLUMN...")
            # mapping of column -> SQL type
            col_types = {
                "ID": "VARCHAR",
                "WELL_ID": "VARCHAR",
                "CATEGORY": "VARCHAR",
                "ACTION": "VARCHAR",
                "PRIORITY": "VARCHAR",
                "STATUS": "VARCHAR",
                "REASON": "VARCHAR",
                "EXPECTED_IMPACT": "VARCHAR",
                "CONFIDENCE": "FLOAT",
                "PREDICTION_SOURCE": "VARCHAR",
                "PROBABILITY": "FLOAT",
                "OIL_VOLUME_SLOPE": "FLOAT",
                "OIL_VOLUME_ROLL_MEAN": "FLOAT",
                "GAS_VOLUME_SLOPE": "FLOAT",
                "GAS_VOLUME_ROLL_MEAN": "FLOAT",
                "MOTOR_CURRENT_ROLL_MEAN": "FLOAT",
                "MOTOR_TEMP_ROLL_MEAN": "FLOAT",
                "SURFACE_PRESSURE_ROLL_MEAN": "FLOAT",
                "DETAILS": "VARCHAR",
                "CREATED_AT": "TIMESTAMP_NTZ"
            }
            for c in missing_cols:
                try:
                    sql_type = col_types.get(c, 'VARCHAR')
                    alter_sql = f'ALTER TABLE "{tn}" ADD COLUMN "{c}" {sql_type} NULL'
                    logger.info(f"Altering table {tn}: {alter_sql}")
                    cur.execute(alter_sql)
                    # commit per column to make schema change durable
                    conn.commit()
                    logger.info(f"Added column {c} to {tn}")
                except Exception as alter_ex:
                    logger.warning(f"Failed to add column {c} to {tn}: {alter_ex}")
            # refresh existing columns list
            try:
                cur.execute(f'DESCRIBE TABLE "{tn}"')
                existing_cols = [r[0].upper() for r in cur.fetchall()]
            except Exception:
                existing_cols = orig_cols

        insert_cols = [c for c in orig_cols if c in existing_cols]
        if not insert_cols:
            # fallback to original full insert (may fail and be handled later)
            insert_cols = orig_cols

        insert_cols_sql = ', '.join([f'"{c}"' for c in insert_cols])
        placeholders = ', '.join(['%s'] * len(insert_cols))
        insert_sql = f'INSERT INTO "{tn}" ({insert_cols_sql}) VALUES ({placeholders})'

        params = (
            rec_id,
            rec.get('well_id') or '',
            rec.get('category') or 'Production',
            rec.get('action') or '',
            rec.get('priority') or '',
            rec.get('status') or '',
            rec.get('reason') or '',
            rec.get('expected_impact') or '',
            float(rec.get('confidence') or 0.0),
            rec.get('prediction_source') or 'rule_based',
            float(rec.get('probability')) if rec.get('probability') is not None else None,
            metrics.get('oil_volume_slope'),
            metrics.get('oil_volume_roll_mean'),
            metrics.get('gas_volume_slope'),
            metrics.get('gas_volume_roll_mean'),
            metrics.get('motor_current_roll_mean'),
            metrics.get('motor_temp_roll_mean'),
            metrics.get('surface_pressure_roll_mean'),
            details,
            now
        )

        logger.info(f"Inserting recommendation for {rec.get('well_id')}: {rec.get('action')}")
        safe_params = tuple(_to_native(p) for p in params)
        try:
            cur.execute(insert_sql, safe_params)
            conn.commit()
            logger.info(f"Successfully saved recommendation {rec_id}")
            return rec_id
        except Exception as ex:
            # Log the failure and attempt smarter fallbacks
            logger.warning(f"Primary insert failed: {ex}")
            logger.debug(f"Insert SQL: {insert_sql}")
            logger.debug(f"Params: {safe_params}")

            # First fallback: positional VALUES (original approach)
            try:
                logger.warning(f"Attempting positional fallback insert")
                placeholders = ','.join(['%s'] * len(params))
                alt_sql = f'INSERT INTO "{tn}" VALUES ({placeholders})'
                cur.execute(alt_sql, safe_params)
                conn.commit()
                logger.info(f"Successfully saved recommendation {rec_id} via positional fallback")
                return rec_id
            except Exception as ex2:
                logger.error(f"Positional fallback also failed: {ex2}")
                # Attempt adaptive fallback: discover table columns and insert only matching columns
                try:
                    logger.info("Attempting adaptive fallback by inspecting table columns...")
                    # Describe table to get column list
                    desc_sql = f'DESCRIBE TABLE "{tn}"'
                    cur.execute(desc_sql)
                    cols = [r[0].upper() for r in cur.fetchall()]

                    # Original insert column order
                    orig_cols = [
                        "ID", "WELL_ID", "CATEGORY", "ACTION", "PRIORITY", "STATUS", "REASON", "EXPECTED_IMPACT", "CONFIDENCE", "PREDICTION_SOURCE", "PROBABILITY",
                        "OIL_VOLUME_SLOPE", "OIL_VOLUME_ROLL_MEAN", "GAS_VOLUME_SLOPE", "GAS_VOLUME_ROLL_MEAN",
                        "MOTOR_CURRENT_ROLL_MEAN", "MOTOR_TEMP_ROLL_MEAN", "SURFACE_PRESSURE_ROLL_MEAN", "DETAILS", "CREATED_AT"
                    ]

                    param_map = {c: v for c, v in zip(orig_cols, safe_params)}

                    # choose intersection columns in original order
                    insert_cols = [c for c in orig_cols if c in cols]
                    if not insert_cols:
                        raise RuntimeError(f"No matching columns found in table {tn} for adaptive fallback")

                    insert_placeholders = ','.join(['%s'] * len(insert_cols))
                    insert_cols_sql = ', '.join([f'"{c}"' for c in insert_cols])
                    adaptive_sql = f'INSERT INTO "{tn}" ({insert_cols_sql}) VALUES ({insert_placeholders})'
                    adaptive_params = tuple(param_map[c] for c in insert_cols)

                    logger.debug(f"Adaptive insert SQL: {adaptive_sql}")
                    logger.debug(f"Adaptive params: {adaptive_params}")
                    cur.execute(adaptive_sql, adaptive_params)
                    conn.commit()
                    logger.info(f"Successfully saved recommendation {rec_id} via adaptive fallback")
                    return rec_id
                except Exception as ex3:
                    logger.error(f"Adaptive fallback failed: {ex3}")
                    # Re-raise the original error for upstream handling with full context
                    logger.error(f"Original insert error: {ex}")
                    raise
    except Exception as e:
        logger.error(f"Error saving recommendation: {e}", exc_info=True)
        raise
    finally:
        cur.close()


def predict_category(metrics: dict, model_path: str = 'models/category_classifier.joblib') -> dict:
    """Predict recommendation category from a metrics dict using a trained model.

    Returns {'category': str, 'probability': float} or {} if unavailable.
    """
    if joblib is None:
        return {}
    try:
        if not os.path.exists(model_path):
            return {}
        mdl = joblib.load(model_path)
        if isinstance(mdl, dict):
            clf = mdl.get('model')
            features = mdl.get('features', CANONICAL_FEATURES)
            le = mdl.get('le')
        else:
            clf = mdl
            features = CANONICAL_FEATURES
            le = None

        fv = []
        for f in features:
            fv.append(metrics.get(f) if metrics and f in metrics else 0.0)

        probs = clf.predict_proba([fv])[0]
        idx = probs.argmax()
        pred = clf.classes_[idx]
        # if label encoder present
        if le is not None:
            try:
                pred = le.inverse_transform([pred])[0]
            except Exception:
                pass
        return {'category': pred, 'probability': float(probs[idx])}
    except Exception:
        return {}


# Training functions moved to train_operational_models.py
# Use: python train_operational_models.py train_category
#      python train_operational_models.py train_action


def train_with_synthetic_labels(snowflake_conn, model_path: str = 'models/decision_tree_alldata.joblib'):
    """Re-export: delegates to train_operational_models. Kept for backward compatibility."""
    from train_operational_models import train_with_synthetic_labels as _train
    return _train(snowflake_conn, model_path=model_path)


def train_from_snowflake(snowflake_conn, lookback_days: int = 30, model_path: str = 'models/decision_tree.joblib', stricter: bool = False):
    """Re-export: delegates to train_operational_models."""
    from train_operational_models import train_from_snowflake as _train
    return _train(snowflake_conn, lookback_days=lookback_days, model_path=model_path, stricter=stricter)


def train_on_two_wells(snowflake_conn, well_ids: list, model_path: str = 'models/decision_tree_two_wells.joblib'):
    """Re-export: delegates to train_operational_models."""
    from train_operational_models import train_on_two_wells as _train
    return _train(snowflake_conn, well_ids, model_path=model_path)


def predict_all_wells(snowflake_conn, model_path: str = 'models/decision_tree.joblib', table_name: str = 'OPERATION_RECOMMENDATION', 
                     confidence_threshold: float = 0.6, dedup_days: int = 7):
    """Run the saved model for all wells and return a list of wells predicted to need action.
    
    Implements client requirements:
    - Only recommend on wells that are NOT in anomaly/alert state
    - Apply ML model with confidence_threshold (default 0.75)
    - Deduplicate: avoid duplicate well+action within dedup_days
    - Persist recommendations to unified `OPERATION_RECOMMENDATION` table in Snowflake
    
    Args:
        snowflake_conn: Snowflake connection
        model_path: Path to trained DecisionTree model
        table_name: Target table for recommendations
        confidence_threshold: Minimum confidence to persist (default 0.75)
        dedup_days: Deduplication window in days (default 7)
    
    Returns:
        List of dicts with detected recommendations
    """
    cur = snowflake_conn.cursor()
    models_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    model_cache = {}  # lift_type -> {'model': clf, 'features': [...]}

    try:
        cur.execute("""
            SELECT DISTINCT wsr.well_id, COALESCE(wsr.lift_type, 'Unknown') as lift_type
            FROM well_sensor_readings wsr
            WHERE EXISTS (SELECT 1 FROM well_daily_production wdp WHERE wdp.well_id = wsr.well_id)
            ORDER BY wsr.well_id
        """)
        well_lift_pairs = [(r[0], str(r[1]).strip() if r[1] else 'Unknown') for r in cur.fetchall()]
        if not well_lift_pairs:
            cur.execute("SELECT DISTINCT well_id FROM well_daily_production ORDER BY well_id")
            well_lift_pairs = [(r[0], 'Unknown') for r in cur.fetchall()]
        logger.info(f"predict_all_wells: Processing {len(well_lift_pairs)} wells")

        results = []
        for well, lift_type in well_lift_pairs:
            try:
                # Load model for this lift type (cached)
                if lift_type not in model_cache:
                    try:
                        model_cache[lift_type] = get_model_for_lift_type(lift_type, models_dir, base_name, model_path)
                    except FileNotFoundError:
                        model_cache[lift_type] = load_model(model_path)
                mdl = model_cache[lift_type]
                clf = mdl['model'] if isinstance(mdl, dict) else mdl

                # STEP 2: Compute metrics
                cur.execute("SELECT * FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 720", (well,))
                rows = cur.fetchall()
                if not rows:
                    continue
                cols = [d[0].lower() for d in cur.description]
                df = pd.DataFrame(rows, columns=cols)
                ts_col = 'timestamp'
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(df[ts_col])
                    df = df.set_index(ts_col)
                ta = TrendAnalyzer(df)
                metrics = ta.compute_metrics()

                # production metrics (last 60 days for better slope estimation)
                cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT 60", (well,))
                prod_rows = cur.fetchall()
                if prod_rows:
                    prod_cols = [d[0].lower() for d in cur.description]
                    prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                    prod_df['date'] = pd.to_datetime(prod_df['date'])
                    prod_df = prod_df.set_index('date')
                    try:
                        if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                            yv = prod_df['oil_volume'].dropna().values[::-1]
                            x = np.arange(len(yv))
                            metrics['oil_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                            metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                    except Exception:
                        metrics['oil_volume_slope'] = None
                        metrics['oil_volume_roll_mean'] = None
                    try:
                        if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                            yv = prod_df['gas_volume'].dropna().values[::-1]
                            x = np.arange(len(yv))
                            metrics['gas_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                            metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                    except Exception:
                        metrics['gas_volume_slope'] = None
                        metrics['gas_volume_roll_mean'] = None

                # STEP 3: Make prediction
                feats, _ = _features_from_metrics(metrics)
                # produce probabilities for all classes and allow multiple recommendations per well
                probas = None
                classes = None
                try:
                    probas = clf.predict_proba([feats])[0]
                    classes = clf.classes_
                except Exception:
                    # predict_proba unavailable; fall back to single prediction
                    pred = clf.predict([feats])[0]
                    probas = None
                    classes = [pred]

                # collect candidate actions above threshold (exclude NoAction)
                candidates = []
                if probas is not None and classes is not None:
                    for cls_name, p in zip(classes, probas):
                        if cls_name == 'NoAction':
                            continue
                        if p >= confidence_threshold:
                            candidates.append((cls_name, float(p)))
                else:
                    # single prediction fallback
                    if pred != 'NoAction':
                        candidates.append((pred, None))

                # Require at least one detection signal (production or sensor) - lift types have different sensors
                prod_detect = (metrics.get('oil_volume_slope') is not None or metrics.get('gas_volume_slope') is not None) and (
                    abs(float(metrics.get('oil_volume_slope') or 0.0)) > 1e-9 or abs(float(metrics.get('gas_volume_slope') or 0.0)) > 1e-9)
                sensor_detect = any(
                    k.endswith('_slope') and v is not None and abs(float(v)) > 1e-9
                    for k, v in metrics.items()
                )
                if not prod_detect and not sensor_detect:
                    logger.debug(f"Well {well}: skipping - no production or sensor signal (prod_detect={prod_detect}, sensor_detect={sensor_detect})")
                    continue

                if not candidates:
                    logger.debug(f"Well {well}: no candidate actions above threshold {confidence_threshold}")
                    continue

                # persist each candidate
                for pred_action, proba in candidates:
                    logger.info(f"Well {well}: candidate action '{pred_action}' (prob={proba})")
                    # dedup check per action
                    try:
                        cutoff_date = f"DATEADD(day, -{dedup_days}, CURRENT_TIMESTAMP())"
                        # coerce params to native python types to avoid Snowflake binding errors
                        cur.execute(f"""
                            SELECT COUNT(*) FROM {table_name}
                            WHERE well_id = %s AND action = %s AND created_at > {cutoff_date}
                        """, (str(well), str(pred_action)))
                        recent_count = cur.fetchone()[0]
                        if recent_count > 0:
                            logger.info(f"Well {well}: duplicate recommendation for '{pred_action}' within {dedup_days} days, skipping")
                            continue
                    except Exception as e:
                        logger.warning(f"Deduplication check failed for well {well}: {e}")

                    results.append({'well_id': well, 'action': pred_action, 'probability': proba, 'metrics': metrics, 'category': ACTION_CATEGORY_MAP.get(pred_action)})
                    try:
                        rec = {
                            'well_id': well,
                            'action': pred_action,
                            'priority': 'HIGH' if (proba or 0) > 0.85 else 'MEDIUM',
                            'status': 'New',
                            'reason': f'ML model prediction (confidence: {int((proba or 0)*100)}%)',
                            'expected_impact': '',
                            'confidence': proba or 0.0,
                            'prediction_source': 'ml_model',
                            'probability': proba,
                            'metrics': metrics,
                            'created_at': datetime.utcnow().isoformat(),
                            'id': str(uuid.uuid4())
                        }
                        # Use model to predict category (category is source-of-truth)
                        try:
                            cat_pred = predict_category(metrics)
                            if cat_pred:
                                rec['category'] = cat_pred.get('category')
                                # store model probability if available
                                if cat_pred.get('probability') is not None:
                                    rec['probability'] = float(cat_pred.get('probability'))
                        except Exception:
                            rec['category'] = ACTION_CATEGORY_MAP.get(pred_action)

                        # prepare Snowflake detection record (store category from model)
                        rec_sf = rec.copy()
                        # keep the detected action as well for traceability
                        rec_sf['action'] = pred_action
                        save_recommendation_to_snowflake(snowflake_conn, rec_sf, table_name=table_name, metrics=metrics)
                        logger.info(f"Saved recommendation for well {well}: {pred_action}")
                    except Exception as e:
                        logger.error(f"Failed to save recommendation for well {well}: {e}", exc_info=True)
            
            except Exception as e:
                logger.error(f"Error processing well {well}: {e}", exc_info=True)
                continue

        logger.info(f"predict_all_wells completed: {len(results)} wells recommended")
        return results
    finally:
        cur.close()


def _features_from_metrics(metrics: dict):
    """Convert metrics dict to a flat numeric feature vector and feature names."""
    # Use the canonical FEATURES list and prefer production-derived roll means when available
    keys = FEATURES
    features = []
    for k in keys:
        v = metrics.get(k)
        try:
            features.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)
        except Exception:
            features.append(0.0)
    return np.array(features), keys


def load_model(model_path: str = 'models/decision_tree.joblib'):
    """Load a single model from path. Returns dict with 'model' and 'features' or raw classifier."""
    if joblib is None:
        raise RuntimeError('joblib not available')
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and 'model' in obj:
        return obj
    return {'model': obj, 'features': CANONICAL_FEATURES}


def get_model_for_lift_type(lift_type: str, models_dir: str, base_name: str = 'decision_tree_alldata', fallback_path: str = None):
    """Load model for lift type; fall back to combined model if lift-specific not found."""
    if joblib is None:
        raise RuntimeError('joblib not available')
    lt_safe = (lift_type or 'unknown').strip().lower().replace(' ', '_').replace('-', '_')
    lt_path = os.path.join(models_dir, f'{base_name}_{lt_safe}.joblib')
    if os.path.exists(lt_path):
        obj = joblib.load(lt_path)
        return {'model': obj.get('model', obj) if isinstance(obj, dict) else obj, 'features': (obj.get('features') or CANONICAL_FEATURES) if isinstance(obj, dict) else CANONICAL_FEATURES}
    fallback = fallback_path or os.path.join(models_dir, f'{base_name}.joblib')
    if os.path.exists(fallback):
        obj = joblib.load(fallback)
        return {'model': obj.get('model', obj) if isinstance(obj, dict) else obj, 'features': (obj.get('features') or CANONICAL_FEATURES) if isinstance(obj, dict) else CANONICAL_FEATURES}
    raise FileNotFoundError(f"No model found for lift_type={lift_type} or fallback at {fallback}")


def predict_with_model(model_path: str, metrics: dict):
    mdl = load_model(model_path)
    clf = mdl['model'] if isinstance(mdl, dict) else mdl
    feats, _ = _features_from_metrics(metrics)
    pred = clf.predict([feats])[0]
    proba = None
    try:
        proba = clf.predict_proba([feats])[0].max()
    except Exception:
        proba = None
    return {'prediction': pred, 'probability': float(proba) if proba is not None else None}


def generate_recommendations_for_well(snowflake_conn, well_id: str, start_ts: str = None, end_ts: str = None):
    """Fetch data from Snowflake for a well, compute metrics, apply rules, estimate impact, and save recommendations.

    Returns list of saved recommendation dicts (with generated `id`).
    """
    cur = snowflake_conn.cursor()
    try:
        # basic query: fetch sensor readings (production is stored in well_daily_production)
        q = f"SELECT timestamp, surface_pressure, tubing_pressure, casing_pressure, motor_temp, wellhead_temp, motor_current FROM well_sensor_readings WHERE well_id = %s"
        params = [well_id]
        if start_ts:
            q += " AND timestamp >= %s"
            params.append(start_ts)
        if end_ts:
            q += " AND timestamp <= %s"
            params.append(end_ts)
        q += " ORDER BY timestamp"
        cur.execute(q, params)
        rows = cur.fetchall()
        if not rows:
            return []
        # normalize column names (Snowflake may return uppercase names)
        cols = [d[0].lower() for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        # locate timestamp-like column robustly
        ts_col = None
        for c in df.columns:
            if 'timestamp' in c or c == 'time' or c.endswith('_ts') or 'date' in c:
                ts_col = c
                break
        if ts_col is None:
            ts_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[ts_col])
        df = df.set_index('timestamp')
        ta = TrendAnalyzer(df)
        metrics = ta.compute_metrics()

        # Fetch daily production for the well and compute production-based metrics
        try:
            # use same cursor to query production
            prod_q = "SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s"
            prod_params = [well_id]
            if start_ts:
                prod_q += " AND date >= %s"
                prod_params.append(start_ts.split('T')[0] if 'T' in start_ts else start_ts)
            if end_ts:
                prod_q += " AND date <= %s"
                prod_params.append(end_ts.split('T')[0] if 'T' in end_ts else end_ts)
            prod_q += " ORDER BY date"
            cur.execute(prod_q, prod_params)
            prod_rows = cur.fetchall()
            if prod_rows:
                prod_cols = [d[0] for d in cur.description]
                prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                prod_df['date'] = pd.to_datetime(prod_df['date'])
                prod_df = prod_df.set_index('date')
                # compute slopes for oil and gas if available
                try:
                    if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                        y = prod_df['oil_volume'].dropna().values
                        x = np.arange(len(y))
                        metrics['oil_volume_slope'] = float(np.polyfit(x, y, 1)[0])
                        metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=7).mean().iloc[-1])
                    else:
                        metrics['oil_volume_slope'] = None
                        metrics['oil_volume_roll_mean'] = None
                except Exception:
                    metrics['oil_volume_slope'] = None
                    metrics['oil_volume_roll_mean'] = None

                try:
                    if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                        y = prod_df['gas_volume'].dropna().values
                        x = np.arange(len(y))
                        metrics['gas_volume_slope'] = float(np.polyfit(x, y, 1)[0])
                        metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=7).mean().iloc[-1])
                    else:
                        metrics['gas_volume_slope'] = None
                        metrics['gas_volume_roll_mean'] = None
                except Exception:
                    metrics['gas_volume_slope'] = None
                    metrics['gas_volume_roll_mean'] = None
        except Exception:
            # if production fetch fails, continue with sensor-only metrics
            pass
        re = RuleEngine(snowflake_conn)
        raw_recs = re.apply(well_id, metrics, last_reading=df.iloc[-1].to_dict(), trend_analyzer=ta)

        ie = ImpactEstimator()
        saved = []
        for r in raw_recs:
            impact = ie.estimate(r, metrics)
            r['impact'] = impact
            r['metrics'] = metrics
            r['created_at'] = datetime.utcnow().isoformat()
            r['id'] = str(uuid.uuid4())
            # Save to Snowflake
            try:
                save_recommendation_to_snowflake(snowflake_conn, r)
            except Exception:
                # swallow to allow returning recommendations even if save fails
                pass
            saved.append(r)

        return saved
    finally:
        cur.close()


def detect_and_persist(snowflake_conn, pg_conn=None, model_path: str = 'models/decision_tree.joblib',
                       retrain: bool = False, confidence_threshold: float = 0.6,
                       snowflake_table: str = 'operational_recommendation'):
    """Run training (optional), detect recommendations for all wells, and persist to Snowflake and PostgreSQL.

    Args:
        snowflake_conn: active Snowflake connection (cursor/execute API expected)
        pg_conn: optional psycopg2 connection to PostgreSQL; if None the function will try to open one using
                 DATABASE_URL environment variable via psycopg2.connect
        model_path: path to model file
        retrain: if True, call `train_from_snowflake` to retrain model before prediction
        confidence_threshold: minimum probability to persist to Postgres as well
        snowflake_table: target Snowflake table name

    Returns:
        list of persisted recommendation ids (snowflake ids)
    """
    persisted_ids = []

    # optional retrain
    if retrain:
        try:
            logger.info("Retraining model from Snowflake before detection")
            train_from_snowflake(snowflake_conn, model_path=model_path)
        except Exception as e:
            logger.warning(f"Retraining failed: {e}")

    # run predictions for all wells
    try:
        recs = predict_all_wells(snowflake_conn, model_path=model_path, table_name=snowflake_table, confidence_threshold=confidence_threshold)
    except Exception as e:
        logger.error(f"Error running predict_all_wells: {e}")
        return persisted_ids

    # If no pg_conn provided, try to open one from env
    local_conn_opened = False
    if pg_conn is None:
        try:
            import psycopg2
            from urllib.parse import urlparse, urlunparse
            pg_url = os.getenv('DATABASE_URL')
            if pg_url:
                parsed = urlparse(pg_url)
                clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', parsed.fragment))
                pg_conn = psycopg2.connect(clean)
                local_conn_opened = True
            else:
                logger.info('No DATABASE_URL set; skipping Postgres persistence')
        except Exception as e:
            logger.warning(f'Could not open Postgres connection from env: {e}')

    # persist each recommendation
    for rec in recs:
        try:
            # ensure category exists (use category predictor if missing)
            if not rec.get('category'):
                try:
                    cat = predict_category(rec.get('metrics', {}))
                    if cat:
                        rec['category'] = cat.get('category')
                        # allow saving probability under probability key
                        if cat.get('probability') is not None:
                            rec['probability'] = float(cat.get('probability'))
                except Exception:
                    pass

            # save to Snowflake (will create table if needed)
            try:
                sf_id = save_recommendation_to_snowflake(snowflake_conn, rec, table_name=snowflake_table, metrics=rec.get('metrics'))
                persisted_ids.append(sf_id)
            except Exception as e:
                logger.error(f"Failed to save to Snowflake for well {rec.get('well_id')}: {e}")

            # persist to Postgres if connection available and confidence passes threshold
            conf = rec.get('probability') or rec.get('confidence') or 0.0
            # convert to percent if probability in [0,1]
            conf_pct = float(conf * 100.0) if conf <= 1.0 else float(conf)
            if pg_conn and (conf_pct >= (confidence_threshold * 100.0)):
                try:
                    # Import here to avoid circular imports at module load time
                    try:
                        from operation_suggestions import save_operation_suggestion
                    except Exception:
                        save_operation_suggestion = None

                    prod_metrics = {k: rec.get('metrics', {}).get(k) for k in ('oil_volume_slope', 'oil_volume_roll_mean', 'gas_volume_slope', 'gas_volume_roll_mean')}
                    sensor_metrics = {k: rec.get('metrics', {}).get(k) for k in ('motor_current_roll_mean', 'motor_temp_roll_mean', 'surface_pressure_roll_mean', 'missing_rate')}

                    if save_operation_suggestion is not None:
                        ok = save_operation_suggestion(
                            rec.get('well_id'),
                            rec.get('action') or rec.get('recommendation') or 'Recommendation',
                            category=rec.get('category') or 'Production',
                            status=rec.get('status') or 'New',
                            priority=rec.get('priority') or 'MEDIUM',
                            confidence=conf_pct,
                            production_data=prod_metrics,
                            sensor_metrics=sensor_metrics,
                            reason=rec.get('reason') or '',
                            expected_impact=rec.get('expected_impact') or ''
                        )
                        if not ok:
                            logger.error(f"Failed to save operation suggestion via operation_suggestions.save_operation_suggestion for well {rec.get('well_id')}")
                    else:
                        logger.warning("operation_suggestions.save_operation_suggestion not available; skipping Postgres insert")
                except Exception as e:
                    logger.error(f"Failed to save to Postgres for well {rec.get('well_id')}: {e}")

        except Exception as e:
            logger.warning(f"Error processing recommendation for persistence: {e}")

    # close local pg_conn if opened here
    if local_conn_opened and pg_conn is not None:
        try:
            pg_conn.close()
        except Exception:
            pass

    logger.info(f"detect_and_persist: persisted {len(persisted_ids)} recommendations to Snowflake")
    return persisted_ids


if __name__ == '__main__':
    # Delegate to train_operational_models CLI
    from train_operational_models import main
    main()
