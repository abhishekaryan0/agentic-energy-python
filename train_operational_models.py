#!/usr/bin/env python3
"""
Training module for operational recommendation models.

Trains:
- Action classifier (per lift-type + fallback): predicts recommended actions from well metrics
- Category classifier: predicts Production/Optimization/Maintenance from metrics

Run:
  python train_operational_models.py train_category
  python train_operational_models.py train_action
  python train_operational_models.py train_action --model-path models/decision_tree.joblib
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import shared components from operational_recommendations
from operational_recommendations import (
    TrendAnalyzer,
    RuleEngine,
    CANONICAL_FEATURES,
    ACTION_MAPPING,
    ACTION_CATEGORY_MAP,
    _features_from_metrics,
)

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    DecisionTreeClassifier = None
    RandomForestClassifier = None
    CalibratedClassifierCV = None
    train_test_split = None
    classification_report = None
    joblib = None
    SKLEARN_AVAILABLE = False


def train_category_model(snowflake_conn, model_path: str = 'models/category_classifier.joblib') -> dict:
    """Train a category classifier using synthetic labels from RuleEngine.

    Returns training summary dict.
    """
    if not SKLEARN_AVAILABLE or RandomForestClassifier is None or joblib is None:
        raise RuntimeError('Required sklearn/joblib not available')

    cur = snowflake_conn.cursor()
    X, y = [], []
    try:
        cur.execute("SELECT DISTINCT well_id FROM well_sensor_readings")
        wells = [r[0] for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching wells: {e}")
        raise

    re = RuleEngine(snowflake_conn)

    for well in wells:
        try:
            cur.execute("SELECT * FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 720", (well,))
            rows = cur.fetchall()
            if not rows:
                continue
            cols = [str(d[0]).lower() for d in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            if 'timestamp' not in df.columns:
                for c in df.columns:
                    if 'timestamp' in c or c == 'time' or c.endswith('_ts'):
                        df = df.rename(columns={c: 'timestamp'})
                        break
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            ta = TrendAnalyzer(df)
            metrics = ta.compute_metrics()

            # Production-derived metrics
            cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT 60", (well,))
            prod_rows = cur.fetchall()
            if prod_rows:
                prod_cols = [str(d[0]).lower() for d in cur.description]
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
                    pass
                try:
                    if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['gas_volume'].dropna().values[::-1]
                        x = np.arange(len(yv))
                        metrics['gas_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                        metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    pass

            recs = re.apply(well, metrics, trend_analyzer=ta)
            for rec in recs:
                cat = rec.get('category')
                if not cat:
                    continue
                feats, _ = _features_from_metrics(metrics)
                X.append(feats.tolist())
                y.append(cat)
        except Exception as e:
            logger.warning(f"Error processing well {well}: {e}")
            continue

    if not X:
        raise RuntimeError('No training samples generated')

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_enc)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': clf, 'features': CANONICAL_FEATURES, 'le': le, 'classes_': le.classes_.tolist()}, model_path)

    preds = clf.predict(X)
    report = classification_report(y_enc, preds, target_names=le.classes_.tolist(), zero_division=0)
    summary = {'n_samples': len(X), 'classes': le.classes_.tolist(), 'report': report, 'model_path': model_path}
    logger.info(f"Trained category classifier: {summary}")
    return summary


def train_with_synthetic_labels(snowflake_conn, model_path: str = 'models/decision_tree_alldata.joblib') -> dict:
    """Train action models (per lift-type + fallback) using synthetic labels from z-score deviations."""
    if not SKLEARN_AVAILABLE or DecisionTreeClassifier is None:
        raise RuntimeError('scikit-learn not available')

    cur = snowflake_conn.cursor()
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
        logger.info(f"Found {len(well_lift_pairs)} wells for training")

        all_metrics = []
        for well, lift_type in well_lift_pairs:
            try:
                cur.execute("SELECT timestamp, surface_pressure, tubing_pressure, casing_pressure, motor_temp, wellhead_temp, motor_current FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 720", (well,))
                rows = cur.fetchall()
                if not rows:
                    continue
                cols = [str(d[0]).lower() for d in cur.description]
                df = pd.DataFrame(rows, columns=cols)
                ts_col = next((c for c in df.columns if 'timestamp' in c or c == 'time' or c.endswith('_ts')), df.columns[0])
                df['timestamp'] = pd.to_datetime(df[ts_col])
                df = df.set_index('timestamp')
                ta = TrendAnalyzer(df)
                metrics = ta.compute_metrics()

                cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT 60", (well,))
                prod_rows = cur.fetchall()
                if prod_rows:
                    prod_cols = [str(d[0]).lower() for d in cur.description]
                    prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                    prod_df['date'] = pd.to_datetime(prod_df['date'])
                    prod_df = prod_df.set_index('date')
                    try:
                        if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                            yv = prod_df['oil_volume'].dropna().values[::-1]
                            metrics['oil_volume_slope'] = float(np.polyfit(np.arange(len(yv)), yv, 1)[0])
                            metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                    except Exception:
                        metrics['oil_volume_slope'] = None
                        metrics['oil_volume_roll_mean'] = None
                    try:
                        if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                            yv = prod_df['gas_volume'].dropna().values[::-1]
                            metrics['gas_volume_slope'] = float(np.polyfit(np.arange(len(yv)), yv, 1)[0])
                            metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                    except Exception:
                        metrics['gas_volume_slope'] = None
                        metrics['gas_volume_roll_mean'] = None

                all_metrics.append((well, lift_type, metrics))
            except Exception as e:
                logger.warning(f"Error processing well {well}: {e}")

        if not all_metrics:
            raise ValueError('No metrics collected from Snowflake')

        logger.info(f"Collected metrics for {len(all_metrics)} wells")

        metric_keys = list(all_metrics[0][2].keys())
        field_means = {k: np.mean([m.get(k) for _, _, m in all_metrics if m.get(k) is not None]) or 0.0 for k in metric_keys}
        field_stds = {k: (np.std([m.get(k) for _, _, m in all_metrics if m.get(k) is not None]) or 1.0) for k in metric_keys}

        samples_by_lift = {}
        for well, lift_type, metrics in all_metrics:
            feats, _ = _features_from_metrics(metrics)
            candidates = []
            for key, fmap in [('motor_current_roll_mean', 'motor_current'), ('motor_temp_slope', 'motor_temp'),
                              ('surface_pressure_slope', 'surface_pressure'), ('oil_volume_slope', 'oil_volume'),
                              ('gas_volume_slope', 'gas_volume')]:
                val = metrics.get(key)
                std, mean = field_stds.get(key, 1.0), field_means.get(key, 0.0)
                if val is not None and std and abs(std) > 1e-6:
                    z = (val - mean) / (std + 1e-6)
                    if abs(z) > 1.2:
                        amap = ACTION_MAPPING.get(fmap)
                        if amap and isinstance(amap, dict):
                            act = amap.get('class') or amap.get('category')
                            cat = amap.get('category') or ACTION_CATEGORY_MAP.get(act, 'Production')
                            candidates.append((act, cat, abs(z)))

            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                label = candidates[0][0]
            else:
                label = 'NoAction'

            lt_key = str(lift_type).strip() or 'Unknown'
            samples_by_lift.setdefault(lt_key, []).append((feats, label))

        X_all = np.vstack([s[0] for samples in samples_by_lift.values() for s in samples])
        y_all = np.array([s[1] for samples in samples_by_lift.values() for s in samples])
        label_dist = {k: int((y_all == k).sum()) for k in np.unique(y_all)}
        logger.info(f"Label distribution: {label_dist}")

        def _train_and_save(X, y, save_path, lift_label='combined'):
            if len(X) < 3:
                logger.warning(f"Insufficient samples ({len(X)}) for {lift_label}")
                return None
            stratify = y if min((y == c).sum() for c in np.unique(y)) >= 2 else None
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

            if RandomForestClassifier and CalibratedClassifierCV:
                rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=3, min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1)
                try:
                    cal = CalibratedClassifierCV(rf, method='sigmoid', cv=min(3, len(np.unique(y_tr))))
                    cal.fit(X_tr, y_tr)
                    clf = cal
                except Exception:
                    rf.fit(X_tr, y_tr)
                    clf = rf
            else:
                clf = DecisionTreeClassifier(max_depth=5, min_samples_split=3, min_samples_leaf=2, random_state=42, class_weight='balanced')
                clf.fit(X_tr, y_tr)

            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
            te_acc = accuracy_score(y_te, clf.predict(X_te))
            prec, rec, _, _ = precision_recall_fscore_support(y_te, clf.predict(X_te), average='weighted', zero_division=0)
            logger.info(f"  {lift_label}: acc={te_acc:.3f}, prec={prec:.3f}, rec={rec:.3f}")

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({'model': clf, 'features': CANONICAL_FEATURES}, save_path)
            return {'train_accuracy': tr_acc, 'test_accuracy': te_acc, 'precision': prec, 'recall': rec, 'trained_on': len(y)}

        results = {}
        models_dir = Path(model_path).parent
        base_name = Path(model_path).stem

        for lt_key, samples in samples_by_lift.items():
            if len(samples) < 5:
                logger.info(f"Skipping lift_type '{lt_key}': {len(samples)} samples (min 5)")
                continue
            X_lt = np.vstack([s[0] for s in samples])
            y_lt = np.array([s[1] for s in samples])
            lt_safe = lt_key.lower().replace(' ', '_').replace('-', '_')
            lt_path = str(models_dir / f"{base_name}_{lt_safe}.joblib")
            logger.info(f"Training model for lift_type: {lt_key} ({len(samples)} samples)")
            res = _train_and_save(X_lt, y_lt, lt_path, lt_key)
            if res:
                results[lt_key] = {**res, 'model_path': lt_path}

        if len(X_all) >= 3:
            logger.info(f"Training combined fallback model ({len(X_all)} samples)")
            res = _train_and_save(X_all, y_all, model_path, 'combined')
            if res:
                results['_fallback'] = {**res, 'model_path': model_path}

        fallback = results.get('_fallback') or (list(results.values())[0] if results else {})
        return {
            'model_path': model_path,
            'per_lift_type': results,
            'label_distribution': label_dist,
            'train_accuracy': fallback.get('train_accuracy', 0),
            'test_accuracy': fallback.get('test_accuracy', 0),
            'trained_on': len(y_all),
        }
    finally:
        cur.close()


def train_from_snowflake(snowflake_conn, lookback_days: int = 30, model_path: str = 'models/decision_tree.joblib', stricter: bool = False) -> dict:
    """Train from Snowflake using rule violations as labels."""
    if not SKLEARN_AVAILABLE or DecisionTreeClassifier is None:
        raise RuntimeError('scikit-learn not available')

    re = RuleEngine(snowflake_conn)
    try:
        re.load_rules_from_db()
    except Exception:
        pass

    if stricter and re.rules:
        for field in re.rules:
            r = re.rules[field]
            if r.get('min') is not None:
                r['min'] = r['min'] * 0.8
            if r.get('max') is not None:
                r['max'] = r['max'] * 1.2

    cur = snowflake_conn.cursor()
    X, y = [], []
    try:
        cur.execute("SELECT DISTINCT well_id FROM well_daily_production ORDER BY well_id")
        wells = [r[0] for r in cur.fetchall()]
        for well in wells:
            cur.execute("SELECT * FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 720", (well,))
            rows = cur.fetchall()
            if not rows:
                continue
            cols = [str(d[0]).lower() for d in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            df['timestamp'] = pd.to_datetime(df.get('timestamp', df.iloc[:, 0]))
            df = df.set_index('timestamp')
            ta = TrendAnalyzer(df)
            metrics = ta.compute_metrics()

            cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT %s", (well, lookback_days))
            prod_rows = cur.fetchall()
            if prod_rows:
                prod_cols = [str(d[0]).lower() for d in cur.description]
                prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                prod_df['date'] = pd.to_datetime(prod_df['date'])
                prod_df = prod_df.set_index('date')
                try:
                    if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['oil_volume'].dropna().values[::-1]
                        metrics['oil_volume_slope'] = float(np.polyfit(np.arange(len(yv)), yv, 1)[0])
                        metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    pass
                try:
                    if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['gas_volume'].dropna().values[::-1]
                        metrics['gas_volume_slope'] = float(np.polyfit(np.arange(len(yv)), yv, 1)[0])
                        metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    pass

            last = df.iloc[-1].to_dict() if len(df) else {}
            violations = re.check_ranges(last)
            mapping = {'motor_current': 'Increase ESP frequency', 'motor_temp': 'Increase ESP frequency',
                       'surface_pressure': 'Adjust pump/polish choke', 'oil_volume': 'Adjust pump/polish choke',
                       'gas_volume': 'Optimize gas lift injection rate'}
            label = 'NoAction'
            for v in violations:
                label = mapping.get(v['field'], label)
                break

            feats, _ = _features_from_metrics(metrics)
            X.append(feats)
            y.append(label)

        if not X:
            raise ValueError('No training data assembled')

        X = np.vstack(X)
        y = np.array(y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        clf.fit(X_tr, y_tr)
        report = classification_report(y_te, clf.predict(X_te), output_dict=True, zero_division=0)

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': clf, 'features': CANONICAL_FEATURES}, model_path)
        return {'model_path': model_path, 'report': report, 'trained_on': len(y)}
    finally:
        cur.close()


def train_on_two_wells(snowflake_conn, well_ids: list, model_path: str = 'models/decision_tree_two_wells.joblib') -> dict:
    """Train on specific wells (for testing)."""
    if not SKLEARN_AVAILABLE or not well_ids or len(well_ids) < 2:
        raise ValueError('Provide at least two well ids and sklearn')

    re = RuleEngine(snowflake_conn)
    try:
        re.load_rules_from_db()
    except Exception:
        pass

    cur = snowflake_conn.cursor()
    X, y = [], []
    mapping = {'motor_current': 'Increase ESP frequency', 'motor_temp': 'Increase ESP frequency',
               'surface_pressure': 'Adjust pump/polish choke', 'oil_volume': 'Adjust pump/polish choke',
               'gas_volume': 'Optimize gas lift injection rate'}
    try:
        for well in well_ids:
            cur.execute("SELECT * FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 720", (well,))
            rows = cur.fetchall()
            if not rows:
                continue
            cols = [str(d[0]).lower() for d in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            df['timestamp'] = pd.to_datetime(df.get('timestamp', df.iloc[:, 0]))
            df = df.set_index('timestamp')
            ta = TrendAnalyzer(df)
            metrics = ta.compute_metrics()

            cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT 60", (well,))
            prod_rows = cur.fetchall()
            if prod_rows:
                prod_cols = [str(d[0]).lower() for d in cur.description]
                prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                prod_df['date'] = pd.to_datetime(prod_df['date'])
                prod_df = prod_df.set_index('date')
                try:
                    if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['oil_volume'].dropna().values[::-1]
                        metrics['oil_volume_slope'] = float(np.polyfit(np.arange(len(yv)), yv, 1)[0])
                        metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    pass
                try:
                    if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['gas_volume'].dropna().values[::-1]
                        metrics['gas_volume_slope'] = float(np.polyfit(np.arange(len(yv)), yv, 1)[0])
                        metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    pass

            last = df.iloc[-1].to_dict() if len(df) else {}
            violations = re.check_ranges(last)
            label = 'NoAction'
            for v in violations:
                label = mapping.get(v['field'], label)
                break

            feats, _ = _features_from_metrics(metrics)
            X.append(feats)
            y.append(label)

        if not X:
            raise RuntimeError('No training samples from provided wells')

        X = np.vstack(X)
        y = np.array(y)
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        clf.fit(X, y)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': clf, 'features': CANONICAL_FEATURES}, model_path)
        return {'model_path': model_path, 'trained_on': len(y), 'label_distribution': {k: int((y == k).sum()) for k in np.unique(y)}}
    finally:
        cur.close()


def _get_snowflake_conn(args=None):
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / '.env')
    try:
        import snowflake.connector
        kw = {
            'user': getattr(args, 'user', None) or os.getenv('SNOWFLAKE_USER'),
            'password': getattr(args, 'password', None) or os.getenv('SNOWFLAKE_PASSWORD'),
            'account': getattr(args, 'account', None) or os.getenv('SNOWFLAKE_ACCOUNT'),
            'warehouse': getattr(args, 'warehouse', None) or os.getenv('SNOWFLAKE_WAREHOUSE'),
            'database': getattr(args, 'database', None) or os.getenv('SNOWFLAKE_DATABASE'),
            'schema': getattr(args, 'schema', None) or os.getenv('SNOWFLAKE_SCHEMA'),
            'role': os.getenv('SNOWFLAKE_ROLE'),
        }
        kw = {k: v for k, v in kw.items() if v}
        return snowflake.connector.connect(**kw)
    except Exception as e:
        logger.error(f"Snowflake connection failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Train operational recommendation models')
    sub = parser.add_subparsers(dest='cmd', help='Command')
    p_cat = sub.add_parser('train_category', help='Train category classifier')
    p_cat.add_argument('--model-path', default='models/category_classifier.joblib')

    p_act = sub.add_parser('train_action', help='Train action models (per lift-type + fallback)')
    p_act.add_argument('--model-path', default='models/decision_tree_alldata.joblib')

    args = parser.parse_args()

    if args.cmd == 'train_category':
        conn = _get_snowflake_conn(args)
        try:
            train_category_model(conn, model_path=args.model_path)
        finally:
            conn.close()
        return

    if args.cmd == 'train_action':
        conn = _get_snowflake_conn(args)
        try:
            train_with_synthetic_labels(conn, model_path=args.model_path)
        finally:
            conn.close()
        return

    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
