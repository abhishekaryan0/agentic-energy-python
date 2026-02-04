"""
Configuration module for at-risk-assets.

Provides:
- Database setup utilities (Snowflake + PostgreSQL)
- Sensor configuration loader (lift types, sensor definitions, anomaly rules)
"""

from config.sensor_config import (
    get_db_connection,
    get_lift_types,
    get_sensors_for_lift_type,
    get_form_sensors,
    get_sensor_ranges,
    get_form_fields,
    get_features_for_lift_type,
    get_all_sensor_columns,
    get_anomaly_rules,
    get_all_sensor_info,
    clear_cache,
)

__all__ = [
    'get_db_connection',
    'get_lift_types',
    'get_sensors_for_lift_type',
    'get_form_sensors',
    'get_sensor_ranges',
    'get_form_fields',
    'get_features_for_lift_type',
    'get_all_sensor_columns',
    'get_anomaly_rules',
    'get_all_sensor_info',
    'clear_cache',
]
