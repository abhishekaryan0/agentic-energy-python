"""
Rule Engine Module
==================

Provides the RuleEngine and ImpactEstimator classes for rule-based
recommendation generation from sensor metrics.

Usage:
    >>> from operations.recommendations.rules import RuleEngine, ACTION_MAPPING
    >>> engine = RuleEngine(snowflake_conn)
    >>> recommendations = engine.apply(well_id, metrics)
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from operations.recommendations.analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)


# =============================================================================
# ACTION MAPPING CONFIGURATION
# =============================================================================

# Data-driven action mapping: sensor_field -> action configuration
# Each entry defines what action to recommend when this sensor is out of range
ACTION_MAPPING: Dict[str, Dict[str, Any]] = {
    # === ROD PUMP SENSORS ===
    'strokes_per_minute': {
        'class': 'Adjust pumping speed',
        'category': 'Production',
        'high_action': 'Reduce pumping speed to prevent equipment wear',
        'low_action': 'Increase pumping speed to improve production',
        'templates': ['Adjust strokes per minute from {current} to {target} SPM']
    },
    'torque': {
        'class': 'Check rod string/gearbox',
        'category': 'Maintenance',
        'high_action': 'Inspect rod string for potential parting or stuck conditions',
        'low_action': 'Check for gas interference or pump-off conditions',
        'templates': ['Inspect gearbox and rod string - torque at {current} in-lbs']
    },
    'polish_rod_load': {
        'class': 'Adjust counterbalance',
        'category': 'Optimization',
        'high_action': 'Rebalance pumping unit - excessive peak load',
        'low_action': 'Check for gas lock or fluid pound conditions',
        'templates': ['Adjust counterbalance - polish rod load at {current} lbs']
    },
    'pump_fillage': {
        'class': 'Optimize pump operation',
        'category': 'Production',
        'high_action': 'Maintain current pumping parameters',
        'low_action': 'Reduce pumping speed or add POC timer - low fillage detected',
        'templates': ['Install pump-off controller - fillage at {current}%']
    },
    'structural_load': {
        'class': 'Reduce structural stress',
        'category': 'Maintenance',
        'high_action': 'Reduce pumping speed to prevent structural damage',
        'low_action': 'Check for gas interference',
        'templates': ['Reduce structural load from {current}% - risk of equipment damage']
    },
    'inferred_production': {
        'class': 'Optimize production rate',
        'category': 'Production',
        'high_action': 'Maintain current settings',
        'low_action': 'Investigate production decline - adjust lift parameters',
        'templates': ['Inferred production at {current} bbls - evaluate optimization']
    },
    'pump_intake_pressure': {
        'class': 'Optimize submergence',
        'category': 'Production',
        'high_action': 'Consider increasing pump rate - high intake pressure',
        'low_action': 'Risk of gas interference - consider deeper pump setting',
        'templates': ['Adjust pump depth or rate - intake pressure at {current} psi']
    },
    
    # === ESP SENSORS ===
    'motor_temp': {
        'class': 'Inspect ESP motor',
        'category': 'Maintenance',
        'high_action': 'Critical: Motor overheating - reduce frequency or shut down',
        'low_action': 'Monitor for potential flow issues',
        'templates': ['Reduce ESP frequency immediately - motor temp at {current}°F']
    },
    'motor_current': {
        'class': 'Adjust ESP frequency',
        'category': 'Production',
        'high_action': 'Reduce ESP frequency - motor overloaded',
        'low_action': 'Increase ESP frequency - motor underloaded',
        'templates': ['Adjust ESP frequency to optimize motor loading - current at {current}A']
    },
    'discharge_pressure': {
        'class': 'Check tubing/wellhead',
        'category': 'Optimization',
        'high_action': 'Check for tubing restriction or closed valve',
        'low_action': 'Check for pump wear or gas interference',
        'templates': ['Inspect tubing - discharge pressure at {current} psi']
    },
    'motor_voltage': {
        'class': 'Check electrical system',
        'category': 'Maintenance',
        'high_action': 'High voltage - check transformer taps',
        'low_action': 'Low voltage - check cable and connections',
        'templates': ['Motor voltage at {current}V - verify electrical system']
    },
    'vibration_x': {
        'class': 'Inspect ESP assembly',
        'category': 'Maintenance',
        'high_action': 'Schedule ESP pullout - excessive vibration detected',
        'low_action': None,
        'templates': ['Critical vibration X at {current} in/s - schedule workover']
    },
    'vibration_y': {
        'class': 'Inspect ESP assembly',
        'category': 'Maintenance',
        'high_action': 'Schedule ESP pullout - excessive vibration detected',
        'low_action': None,
        'templates': ['Critical vibration Y at {current} in/s - schedule workover']
    },
    'motor_load': {
        'class': 'Optimize ESP loading',
        'category': 'Production',
        'high_action': 'Reduce frequency - motor overloaded at {current}%',
        'low_action': 'Increase frequency - motor underutilized',
        'templates': ['Adjust frequency to target 70-85% motor load']
    },
    'drive_frequency': {
        'class': 'Optimize drive settings',
        'category': 'Optimization',
        'high_action': 'Review if high frequency is necessary',
        'low_action': 'Consider increasing frequency if production allows',
        'templates': ['Drive frequency at {current}Hz - evaluate optimization']
    },
    'vsd_temp': {
        'class': 'Check VSD cooling',
        'category': 'Maintenance',
        'high_action': 'VSD overheating - check cooling system',
        'low_action': None,
        'templates': ['VSD temperature at {current}°F - inspect cooling fans']
    },
    'intake_fluid_temp': {
        'class': 'Monitor cooling flow',
        'category': 'Maintenance',
        'high_action': 'High intake temp may indicate low flow - check motor cooling',
        'low_action': None,
        'templates': ['Intake temperature {current}°F - verify adequate flow past motor']
    },
    'downhole_flow_rate': {
        'class': 'Optimize flow rate',
        'category': 'Production',
        'high_action': 'Verify measurement accuracy at high rates',
        'low_action': 'Low flow - check for pump wear or gas interference',
        'templates': ['Downhole flow rate at {current} bpd - evaluate pump performance']
    },
    'total_harmonic_distortion': {
        'class': 'Check power quality',
        'category': 'Maintenance',
        'high_action': 'High THD may damage motor - check power supply',
        'low_action': None,
        'templates': ['THD at {current}% - verify power quality']
    },
    
    # === GAS LIFT SENSORS ===
    'injection_rate': {
        'class': 'Optimize gas injection',
        'category': 'Optimization',
        'high_action': 'Reduce injection rate - possible over-injection',
        'low_action': 'Increase injection rate to improve lift efficiency',
        'templates': ['Adjust injection rate from {current} to {target} mcf']
    },
    'injection_pressure': {
        'class': 'Check injection system',
        'category': 'Maintenance',
        'high_action': 'Check for valve or choke restriction',
        'low_action': 'Check compressor output or line leaks',
        'templates': ['Injection pressure at {current} psi - inspect valves']
    },
    'injection_temperature': {
        'class': 'Monitor injection conditions',
        'category': 'Optimization',
        'high_action': 'High injection temp - check compressor aftercooler',
        'low_action': None,
        'templates': ['Injection temperature at {current}°F']
    },
    'bottomhole_pressure': {
        'class': 'Optimize lift depth',
        'category': 'Production',
        'high_action': 'Consider deeper lift point or increased rate',
        'low_action': 'Risk of gas lock - reduce injection rate',
        'templates': ['BHP at {current} psi - evaluate lift optimization']
    },
    'bottomhole_temp': {
        'class': 'Monitor reservoir conditions',
        'category': 'Optimization',
        'high_action': 'Monitor for potential issues',
        'low_action': None,
        'templates': ['Bottomhole temperature at {current}°F']
    },
    'plunger_velocity': {
        'class': 'Adjust plunger timing',
        'category': 'Optimization',
        'high_action': 'Reduce velocity to prevent plunger damage',
        'low_action': 'Increase buildup pressure for better velocity',
        'templates': ['Plunger velocity {current} ft/min - adjust cycle timing']
    },
    'missed_arrivals': {
        'class': 'Investigate plunger issues',
        'category': 'Maintenance',
        'high_action': 'High missed arrivals - check plunger condition',
        'low_action': None,
        'templates': ['{current} missed arrivals - inspect plunger and bumper spring']
    },
    'arrival_count': {
        'class': 'Monitor plunger performance',
        'category': 'Optimization',
        'high_action': 'Good arrival rate - maintain settings',
        'low_action': 'Low arrivals - check plunger and timing',
        'templates': ['Arrival count at {current} - evaluate cycle efficiency']
    },
    'shutin_time': {
        'class': 'Optimize cycle timing',
        'category': 'Optimization',
        'high_action': 'Long shut-in may indicate well loading issues',
        'low_action': 'Short shut-in may cause incomplete unloading',
        'templates': ['Shut-in time {current} min - optimize for well conditions']
    },
    'flow_time': {
        'class': 'Optimize flow period',
        'category': 'Production',
        'high_action': 'Long flow time - verify well can sustain production',
        'low_action': 'Short flow time - may need longer flow period',
        'templates': ['Flow time at {current} min - evaluate cycle optimization']
    },
    
    # === COMMON SENSORS ===
    'surface_pressure': {
        'class': 'Check flowline/separator',
        'category': 'Optimization',
        'high_action': 'Check for flowline restriction or separator issues',
        'low_action': 'Check for tubing leak or packer failure',
        'templates': ['Surface pressure at {current} psi - inspect flowline']
    },
    'tubing_pressure': {
        'class': 'Inspect tubing integrity',
        'category': 'Maintenance',
        'high_action': 'High tubing pressure - check for restriction',
        'low_action': 'Low tubing pressure - check for leak or communication',
        'templates': ['Tubing pressure at {current} psi - evaluate integrity']
    },
    'casing_pressure': {
        'class': 'Check casing integrity',
        'category': 'Maintenance',
        'high_action': 'High casing pressure - check for tubing leak or gas migration',
        'low_action': 'Verify casing valve positions',
        'templates': ['Casing pressure at {current} psi - check for tubing leak']
    },
    'wellhead_temp': {
        'class': 'Monitor well conditions',
        'category': 'Optimization',
        'high_action': 'High wellhead temp - verify if within equipment limits',
        'low_action': 'Low wellhead temp may indicate reduced flow',
        'templates': ['Wellhead temperature at {current}°F']
    },
    
    # === PRODUCTION FIELDS ===
    'oil_volume': {
        'class': 'Optimize production rate',
        'category': 'Production',
        'high_action': 'Maintain current settings - good production',
        'low_action': 'Investigate production decline - adjust lift parameters',
        'templates': ['Oil production at {current} bbl/day - {trend} trend detected']
    },
    'gas_volume': {
        'class': 'Optimize gas handling',
        'category': 'Optimization',
        'high_action': 'Check for gas breakout or casing flow',
        'low_action': 'Low gas may indicate depletion or liquid loading',
        'templates': ['Gas production at {current} mcf - evaluate GOR trend']
    },
    'water_volume': {
        'class': 'Monitor water production',
        'category': 'Optimization',
        'high_action': 'High water cut - evaluate water shutoff options',
        'low_action': None,
        'templates': ['Water production at {current} bbl/day']
    },
    'boe': {
        'class': 'Monitor total production',
        'category': 'Production',
        'high_action': 'Good overall production',
        'low_action': 'Total production declining - evaluate lift optimization',
        'templates': ['BOE at {current} - evaluate overall performance']
    }
}

# Build ACTION_CATEGORY_MAP dynamically from ACTION_MAPPING
ACTION_CATEGORY_MAP: Dict[str, str] = {
    v['class']: v['category'] 
    for k, v in ACTION_MAPPING.items() 
    if 'class' in v
}

# Add legacy mappings for backward compatibility
ACTION_CATEGORY_MAP.update({
    'Increase ESP frequency': 'Production',
    'Adjust pump/polish choke': 'Production',
    'Optimize gas lift injection rate': 'Optimization',
    'Inspect ESP equipment': 'Maintenance',
    'NoAction': 'None'
})


def load_sensor_config_from_db(conn) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load sensor definitions from database to get actual min/max ranges.
    
    Args:
        conn: Database connection with cursor API
        
    Returns:
        Dict: {lift_type: {field_name: {'min': x, 'max': y, 'unit': z}}}
    """
    config: Dict[str, Dict[str, Dict[str, Any]]] = {}
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT lt.lift_type_name, sd.field_name, sd.min_value, sd.max_value, sd.unit
            FROM sensor_definitions sd
            JOIN lift_types lt ON sd.lift_type_id = lt.id
            WHERE sd.min_value IS NOT NULL AND sd.max_value IS NOT NULL
        """)
        for row in cur.fetchall():
            lt_name, field, mn, mx, unit = row
            if lt_name not in config:
                config[lt_name] = {}
            config[lt_name][field.lower()] = {
                'min': float(mn) if mn is not None else None,
                'max': float(mx) if mx is not None else None,
                'unit': unit
            }
        return config
    except Exception as e:
        logger.warning(f"Could not load sensor config from DB: {e}")
        return {}
    finally:
        cur.close()


class RuleEngine:
    """
    Apply deterministic rules to metrics and generate recommendations.
    
    This class checks sensor values against their defined thresholds
    and generates appropriate recommendations using ACTION_MAPPING.
    
    Attributes:
        conn: Database connection for loading rules
        rules: Dict of field -> {min, max} bounds
    
    Example:
        >>> engine = RuleEngine(snowflake_conn)
        >>> engine.load_rules_from_db(lift_type='ESP')
        >>> violations = engine.check_ranges(readings)
    """

    def __init__(self, conn=None):
        """
        Initialize with database connection for DB-based rules.
        
        Args:
            conn: Optional database connection (Snowflake)
        """
        self.conn = conn
        self.rules: Dict[str, Dict[str, Any]] = {}

    def load_rules_from_db(self, lift_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load field ranges from sensor_definitions table in DB.
        
        Args:
            lift_type: If specified, load rules for that type only
            
        Returns:
            Dict of field -> {min, max} bounds
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
                cur.execute(
                    "SELECT field_name, min_value, max_value FROM sensor_definitions WHERE lift_type_id = %s", 
                    (lift_id,)
                )
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

    def load_adaptive_rules(self, well_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Load adaptive rules based on last 30 days data for the well.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Dict of field -> {min, max} bounds computed from historical data
        """
        if not self.conn:
            return {}
        
        cur = self.conn.cursor()
        try:
            # Get lift_type for the well
            cur.execute(
                "SELECT DISTINCT lift_type FROM well_sensor_readings WHERE well_id = %s LIMIT 1", 
                (well_id,)
            )
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
            cur.execute(
                "SELECT field_name FROM sensor_definitions "
                "WHERE lift_type_id = %s AND min_value IS NOT NULL AND max_value IS NOT NULL", 
                (lift_id,)
            )
            fields = [r[0] for r in cur.fetchall()]
            
            rules = {}
            for field in fields:
                # Compute min/max from last 30 days
                try:
                    cur.execute(
                        f"SELECT MIN(TRY_CAST({field} AS FLOAT)), MAX(TRY_CAST({field} AS FLOAT)) "
                        f"FROM well_sensor_readings "
                        f"WHERE well_id = %s AND timestamp >= DATEADD(day, -30, CURRENT_DATE)", 
                        (well_id,)
                    )
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

    def check_ranges(self, readings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Return violations for readings according to loaded rules.
        
        Args:
            readings: Dict of field_name -> value
            
        Returns:
            List of violation dicts with field, value, min, max, and violation type
        """
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
                violations.append({
                    'field': field, 
                    'value': val, 
                    'min': mn, 
                    'max': mx, 
                    'violation': 'below_min'
                })
            if mx is not None and val > mx:
                violations.append({
                    'field': field, 
                    'value': val, 
                    'min': mn, 
                    'max': mx, 
                    'violation': 'above_max'
                })
        return violations

    def apply(
        self, 
        well_id: str, 
        metrics: Dict[str, Any], 
        last_reading: Optional[Dict[str, Any]] = None, 
        trend_analyzer: Optional['TrendAnalyzer'] = None, 
        adaptive_k: float = 2.0, 
        lift_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply data-driven rules to generate recommendations.
        
        This method checks ALL sensors from sensor_definitions against their thresholds
        and generates appropriate recommendations using ACTION_MAPPING.
        
        Args:
            well_id: Well identifier
            metrics: Dict of computed metrics (slopes, roll_means, etc.)
            last_reading: Dict of latest sensor values
            trend_analyzer: TrendAnalyzer instance for baseline stats
            adaptive_k: Z-score threshold for trend-based detection
            lift_type: Lift type for context-aware recommendations
        
        Returns:
            List of recommendation dicts
        """
        # Load rules from database based on well's lift type
        self.load_adaptive_rules(well_id)
        
        recs: List[Dict[str, Any]] = []
        seen_actions: set = set()  # Avoid duplicate recommendations
        
        # Get lift type from trend_analyzer if not provided
        if not lift_type and trend_analyzer:
            lift_type = trend_analyzer.lift_type
        
        # Part 1: Check sensors against thresholds
        if last_reading:
            recs.extend(self._check_sensor_thresholds(
                well_id, last_reading, seen_actions, lift_type
            ))
        
        # Part 2: Check trends against baseline statistics
        if trend_analyzer is not None:
            recs.extend(self._check_trend_deviations(
                well_id, metrics, trend_analyzer, seen_actions, adaptive_k, lift_type
            ))
        
        # Part 3: Production-specific checks
        recs.extend(self._check_production_trends(
            well_id, metrics, seen_actions, lift_type
        ))
        
        return recs

    def _check_sensor_thresholds(
        self, 
        well_id: str, 
        last_reading: Dict[str, Any], 
        seen_actions: set, 
        lift_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Check sensors against min/max thresholds."""
        recs = []
        
        for field, bounds in self.rules.items():
            field_lower = field.lower()
            val = last_reading.get(field_lower) or last_reading.get(field)
            if val is None:
                continue
                
            mn = bounds.get('min')
            mx = bounds.get('max')
            action_config = ACTION_MAPPING.get(field_lower, {})
            
            # Check for below minimum
            if mn is not None and val < mn:
                rec = self._create_threshold_recommendation(
                    well_id, field, field_lower, val, mn, mx,
                    action_config, 'below_min', seen_actions, lift_type
                )
                if rec:
                    recs.append(rec)
            
            # Check for above maximum
            if mx is not None and val > mx:
                rec = self._create_threshold_recommendation(
                    well_id, field, field_lower, val, mn, mx,
                    action_config, 'above_max', seen_actions, lift_type
                )
                if rec:
                    recs.append(rec)
        
        return recs

    def _create_threshold_recommendation(
        self,
        well_id: str,
        field: str,
        field_lower: str,
        val: float,
        mn: Optional[float],
        mx: Optional[float],
        action_config: Dict[str, Any],
        violation_type: str,
        seen_actions: set,
        lift_type: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Create a recommendation for a threshold violation."""
        if violation_type == 'below_min':
            action_class = action_config.get('class', f'Investigate {field}')
            action_text = action_config.get('low_action', f'{field} below minimum threshold')
            category = action_config.get('category', 'Maintenance')
            threshold = mn
            deviation_pct = ((mn - val) / mn * 100) if mn != 0 else 0
        else:  # above_max
            action_class = action_config.get('class', f'Investigate {field}')
            action_text = action_config.get('high_action', f'{field} above maximum threshold')
            category = action_config.get('category', 'Optimization')
            threshold = mx
            deviation_pct = ((val - mx) / mx * 100) if mx != 0 else 0
        
        if action_class in seen_actions:
            return None
        
        # Data-driven priority: LOW (<10%), MEDIUM (10-25%), HIGH (>25%)
        if deviation_pct > 25:
            priority = 'HIGH'
        elif deviation_pct > 10:
            priority = 'MEDIUM'
        else:
            priority = 'LOW'
        
        detection_metrics = {
            'sensor_field': field_lower,
            'current_value': float(val),
            'optimal_min': float(mn) if mn else None,
            'optimal_max': float(mx) if mx else None,
            'deviation_pct': round(deviation_pct, 2),
            'deviation_type': violation_type,
            'unit': action_config.get('unit', '')
        }
        
        templates = action_config.get('templates', [''])
        expected_impact = templates[0].format(current=val, target=threshold) if templates else ''
        
        rec = {
            'well_id': well_id,
            'action': action_class,
            'action_detail': action_text,
            'category': category,
            'priority': priority,
            'status': 'New',
            'detection_metrics': detection_metrics,
            'reason': f'{field} at {val:.2f} is {"below min" if violation_type == "below_min" else "above max"} threshold {threshold:.2f} ({deviation_pct:.1f}% deviation)',
            'expected_impact': expected_impact,
            'confidence': min(0.9, 0.6 + deviation_pct / 100),
            'sensor_field': field_lower,
            'lift_type': lift_type
        }
        
        seen_actions.add(action_class)
        return rec

    def _check_trend_deviations(
        self,
        well_id: str,
        metrics: Dict[str, Any],
        trend_analyzer: 'TrendAnalyzer',
        seen_actions: set,
        adaptive_k: float,
        lift_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Check trends (slopes) against baseline statistics."""
        recs = []
        
        for field in self.rules.keys():
            field_lower = field.lower()
            slope_key = f'{field_lower}_slope'
            slope = metrics.get(slope_key)
            
            if slope is None:
                continue
            
            # Get baseline statistics for this field
            baseline = trend_analyzer.baseline_stats(field_lower)
            if baseline['slope'] is None or baseline['slope_std'] is None:
                continue
            
            baseline_slope = baseline['slope']
            slope_std = baseline['slope_std'] or 0.01
            
            # Calculate z-score of current slope vs baseline
            z_score = (slope - baseline_slope) / (slope_std + 1e-6)
            
            if abs(z_score) > adaptive_k:
                action_config = ACTION_MAPPING.get(field_lower, {})
                
                if z_score > adaptive_k:
                    action_class = action_config.get('class', f'Monitor {field} trend')
                    action_text = action_config.get('high_action', f'{field} showing significant upward trend')
                else:
                    action_class = action_config.get('class', f'Monitor {field} trend')
                    action_text = action_config.get('low_action', f'{field} showing significant downward trend')
                
                if action_text and action_class not in seen_actions:
                    category = action_config.get('category', 'Optimization')
                    
                    if abs(z_score) > 3:
                        priority = 'HIGH'
                    elif abs(z_score) > 2:
                        priority = 'MEDIUM'
                    else:
                        priority = 'LOW'
                    
                    detection_metrics = {
                        'sensor_field': field_lower,
                        'current_slope': round(float(slope), 4),
                        'baseline_slope': round(float(baseline_slope), 4),
                        'z_score': round(float(z_score), 2),
                        'deviation_type': 'trend_increasing' if z_score > 0 else 'trend_decreasing',
                        'trend_direction': 'up' if z_score > 0 else 'down'
                    }
                    
                    recs.append({
                        'well_id': well_id,
                        'action': action_class,
                        'action_detail': action_text,
                        'category': category,
                        'priority': priority,
                        'status': 'New',
                        'detection_metrics': detection_metrics,
                        'reason': f'{field} trend z-score={z_score:.2f} (slope={slope:.4f} vs baseline={baseline_slope:.4f})',
                        'expected_impact': f'Trend deviation detected - investigate {field}',
                        'confidence': min(0.95, 0.5 + abs(z_score) / 10),
                        'sensor_field': field_lower,
                        'lift_type': lift_type
                    })
                    seen_actions.add(action_class)
        
        return recs

    def _check_production_trends(
        self,
        well_id: str,
        metrics: Dict[str, Any],
        seen_actions: set,
        lift_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Check production-specific trends."""
        recs = []
        
        oil_slope = metrics.get('oil_volume_slope')
        water_slope = metrics.get('water_volume_slope')
        
        # Declining oil production
        if oil_slope is not None and oil_slope < -0.5:
            action = 'Optimize production rate'
            if action not in seen_actions:
                if oil_slope < -3:
                    priority = 'HIGH'
                elif oil_slope < -1.5:
                    priority = 'MEDIUM'
                else:
                    priority = 'LOW'
                
                oil_roll_mean = metrics.get('oil_volume_roll_mean', 0)
                detection_metrics = {
                    'sensor_field': 'oil_volume',
                    'current_slope': round(float(oil_slope), 2),
                    'current_avg': round(float(oil_roll_mean), 2) if oil_roll_mean else None,
                    'decline_rate_bbl_day': round(abs(float(oil_slope)), 2),
                    'deviation_type': 'production_decline',
                    'unit': 'bbl/day'
                }
                
                recs.append({
                    'well_id': well_id,
                    'action': action,
                    'action_detail': 'Oil production declining - evaluate lift optimization',
                    'category': 'Production',
                    'priority': priority,
                    'status': 'New',
                    'detection_metrics': detection_metrics,
                    'reason': f'Oil production slope = {oil_slope:.2f} bbl/day/day',
                    'expected_impact': 'Investigate cause of production decline',
                    'confidence': 0.85,
                    'sensor_field': 'oil_volume',
                    'lift_type': lift_type
                })
                seen_actions.add(action)
        
        # Rising water cut
        if oil_slope is not None and water_slope is not None:
            if water_slope > 0.5 and oil_slope < 0:
                action = 'Monitor water production'
                if action not in seen_actions:
                    if water_slope > 2 and oil_slope < -1:
                        priority = 'HIGH'
                    elif water_slope > 1:
                        priority = 'MEDIUM'
                    else:
                        priority = 'LOW'
                    
                    water_roll_mean = metrics.get('water_volume_roll_mean', 0)
                    detection_metrics = {
                        'sensor_field': 'water_volume',
                        'water_slope': round(float(water_slope), 2),
                        'oil_slope': round(float(oil_slope), 2),
                        'current_water_avg': round(float(water_roll_mean), 2) if water_roll_mean else None,
                        'deviation_type': 'water_cut_increasing',
                        'unit': 'bbl/day'
                    }
                    
                    recs.append({
                        'well_id': well_id,
                        'action': action,
                        'action_detail': 'Water cut increasing - evaluate water management',
                        'category': 'Optimization',
                        'priority': priority,
                        'status': 'New',
                        'detection_metrics': detection_metrics,
                        'reason': f'Water trend = {water_slope:.2f}, Oil trend = {oil_slope:.2f}',
                        'expected_impact': 'Evaluate water shutoff or management options',
                        'confidence': 0.75,
                        'sensor_field': 'water_volume',
                        'lift_type': lift_type
                    })
                    seen_actions.add(action)
        
        return recs


class ImpactEstimator:
    """
    Estimate potential impact/uplift for recommendations.
    
    This is a placeholder implementation. Real implementation would require
    historical adjustment logs and outcome tracking.
    
    Example:
        >>> estimator = ImpactEstimator()
        >>> impact = estimator.estimate(recommendation, metrics)
        >>> print(f"Expected uplift: {impact['expected_uplift_pct']*100}%")
    """

    def estimate(self, recommendation: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate impact for a recommendation.
        
        Args:
            recommendation: Recommendation dict
            metrics: Current metrics dict
            
        Returns:
            Dict with expected_uplift_pct and confidence
        """
        action = recommendation.get('action', '').lower()
        
        if 'esp' in action or 'frequency' in action:
            uplift = 0.15
        elif 'pump' in action or 'choke' in action:
            uplift = 0.08
        elif 'gas lift' in action or 'gas' in action:
            uplift = 0.10
        else:
            uplift = 0.05

        return {
            'expected_uplift_pct': uplift, 
            'confidence': recommendation.get('confidence', 0.5)
        }
