import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from anomaly_detection.s3_history import (
    normalize_anomaly_suggestion_record,
    normalize_violation_record,
    write_anomaly_suggestions_history_to_s3,
)

logger = logging.getLogger(__name__)

try:
    from anomaly_detection.sensor_config import get_sensor_ranges
    from anomaly_detection.tenant_sensors import tenant_profile_label
    _SENSOR_CONFIG_AVAILABLE = True
except ImportError:
    _SENSOR_CONFIG_AVAILABLE = False
    get_sensor_ranges = None
    tenant_profile_label = lambda organization_id=None: "tenant"  # type: ignore


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int_, int)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)


def format_field_name(field: str) -> str:
    if not field:
        return "Unknown Parameter"
    return " ".join(word.capitalize() for word in field.replace("_", " ").split())


def format_number_with_commas(num) -> Optional[str]:
    if num is None:
        return None
    try:
        num_float = float(num)
        if num_float == int(num_float):
            return f"{int(num_float):,}"
        return f"{num_float:,.2f}"
    except (ValueError, TypeError):
        return str(num)


def _format_value_with_unit(value: Any, unit: str = "") -> str:
    formatted = format_number_with_commas(value)
    if formatted is None:
        return "N/A"
    unit = (unit or "").strip()
    return f"{formatted}{f' {unit}' if unit else ''}"


def _format_band(min_val: Any, max_val: Any, unit: str = "") -> str:
    lo = _format_value_with_unit(min_val, unit)
    hi = _format_value_with_unit(max_val, unit)
    return f"{lo}–{hi}"


def _issue_label(value: Any, min_val: Any, max_val: Any) -> str:
    """Short O&G-friendly direction label (no values or expected range in titles)."""
    try:
        val = float(value)
        lo = float(min_val)
        hi = float(max_val)
        if val > hi:
            pct_over = (val - hi) / max(abs(hi - lo), 1e-8) * 100.0
            return "Above Normal" if pct_over < 50 else "High Spike"
        if val < lo:
            pct_under = (lo - val) / max(abs(hi - lo), 1e-8) * 100.0
            return "Below Normal" if pct_under < 50 else "Sharp Drop"
    except (TypeError, ValueError):
        pass
    return "Outside Normal Limits"


def pseudo_violations_from_readings(
    readings: Dict[str, Any],
    lift_type: Optional[str],
) -> List[Dict[str, Any]]:
    """Build violation-like records from readings vs expected tenant sensor bands."""
    if not readings or not lift_type or not get_sensor_ranges:
        return []

    try:
        bands = get_sensor_ranges(lift_type)
    except Exception:
        return []

    pseudo: List[Dict[str, Any]] = []
    for field, band in bands.items():
        value = readings.get(field)
        if value is None:
            continue
        try:
            val = float(value)
            lo = float(band.get("min"))
            hi = float(band.get("max"))
        except (TypeError, ValueError):
            continue
        if lo <= val <= hi:
            continue
        width = max(abs(hi - lo), 1e-8)
        if val < lo:
            deviation_pct = abs(lo - val) / width * 100.0
        else:
            deviation_pct = abs(val - hi) / width * 100.0
        unit = band.get("unit") or ""
        pseudo.append(
            {
                "field": field,
                "value": round(val, 2),
                "min": round(lo, 2),
                "max": round(hi, 2),
                "unit": unit,
                "deviation_pct": round(deviation_pct, 2),
                "violation": (
                    f"{format_field_name(field)} {_format_value_with_unit(val, unit)} "
                    f"outside expected band {_format_band(lo, hi, unit)}"
                ),
            }
        )
    pseudo.sort(key=lambda x: float(x.get("deviation_pct", 0) or 0), reverse=True)
    return pseudo


def effective_violations(
    anomaly_details: Dict[str, Any],
    readings: Dict[str, Any],
    lift_type: Optional[str],
) -> List[Dict[str, Any]]:
    """Rule violations from detection, or band-derived pseudo-violations for ML-only hits."""
    violations = list(anomaly_details.get("violations") or [])
    if violations:
        return violations
    return pseudo_violations_from_readings(readings, lift_type)


def generate_alert_title_from_violations(violations: List[Dict[str, Any]]) -> str:
    """Short data-driven title: sensor name + direction (e.g. 'Injection Rate Sharp Drop')."""
    if not violations:
        return "Sensor Anomaly"

    sorted_violations = sorted(
        violations, key=lambda x: float(x.get("deviation_pct", 0) or 0), reverse=True
    )
    top = sorted_violations[0]
    field_name = format_field_name(str(top.get("field") or "parameter"))
    issue = _issue_label(top.get("value"), top.get("min"), top.get("max"))
    return f"{field_name} {issue}"[:80]


def generate_alert_title_from_anomaly(
    anomaly_details: Dict[str, Any],
    readings: Dict[str, Any],
    lift_type: Optional[str] = None,
) -> str:
    violations = effective_violations(anomaly_details, readings, lift_type)
    if violations:
        return generate_alert_title_from_violations(violations)
    try:
        score = float(anomaly_details.get("anomaly_score", 0) or 0)
    except (TypeError, ValueError):
        score = 0.0
    if score > 0:
        return "Unusual Sensor Pattern"
    return "Sensor Anomaly"


def build_data_driven_description(
    anomaly_details: Dict[str, Any],
    readings: Dict[str, Any],
    lift_type: Optional[str] = None,
) -> str:
    violations = effective_violations(anomaly_details, readings, lift_type)
    score = float(anomaly_details.get("anomaly_score", 0) or 0)
    rule_score = float(anomaly_details.get("rule_score", 0) or 0)
    ml_score = float(anomaly_details.get("ml_score", 0) or 0)
    lstm_score = float(anomaly_details.get("lstm_score", 0) or 0)

    if not violations:
        return (
            f"Ensemble anomaly score {score:.2f} "
            f"(rules {rule_score:.2f}, ML {ml_score:.2f}, LSTM {lstm_score:.2f}) "
            f"indicates an unusual multivariate sensor pattern for this well."
        )

    parts: List[str] = []
    for v in violations:
        field = format_field_name(str(v.get("field") or "parameter"))
        unit = str(v.get("unit") or "")
        val_str = _format_value_with_unit(v.get("value"), unit)
        band_str = _format_band(v.get("min"), v.get("max"), unit)
        dev = float(v.get("deviation_pct", 0) or 0)
        parts.append(
            f"{field} read {val_str}, outside normal operating range {band_str} "
            f"({dev:.1f}% beyond limit)."
        )

    summary = (
        f"Anomaly score {score:.2f} across {len(violations)} sensor"
        f"{'s' if len(violations) != 1 else ''}. "
    )
    return summary + " ".join(parts)


def build_data_driven_suggested_actions(violations: List[Dict[str, Any]]) -> List[str]:
    if not violations:
        return [
            "Review recent sensor trends and confirm data quality for this well.",
            "Compare current readings to the well's recent operating baseline.",
            "Escalate to production engineering if the pattern persists beyond one reading cycle.",
        ]

    top = max(violations, key=lambda x: float(x.get("deviation_pct", 0) or 0))
    field = format_field_name(str(top.get("field") or "parameter"))
    unit = str(top.get("unit") or "")
    val_str = _format_value_with_unit(top.get("value"), unit)
    band_str = _format_band(top.get("min"), top.get("max"), unit)

    actions = [
        f"Verify {field} reading {val_str} against normal operating range {band_str} and field instrumentation.",
        f"Inspect related equipment and recent operational changes affecting {field}.",
    ]
    if len(violations) > 1:
        others = ", ".join(format_field_name(str(v.get("field"))) for v in violations[1:4])
        actions.append(f"Review correlated sensors also outside normal limits: {others}.")
    else:
        actions.append("Monitor the next 2–3 reading cycles before clearing the alert.")
    return actions[:3]


def apply_data_driven_fields(
    suggestion: Dict[str, Any],
    anomaly_details: Dict[str, Any],
    readings: Dict[str, Any],
    lift_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Always set title, severity, confidence, and baseline narrative from anomaly data."""
    violations = effective_violations(anomaly_details, readings, lift_type)

    suggestion["alert_title"] = generate_alert_title_from_anomaly(
        anomaly_details, readings, lift_type
    )
    suggestion["severity"] = calculate_severity(
        anomaly_details, anomaly_details.get("anomaly_score", 0), readings
    )
    suggestion["confidence"] = f"{calculate_data_driven_confidence(anomaly_details)}%"
    suggestion["status"] = suggestion.get("status") or "ACTIVE"

    suggestion["description"] = build_data_driven_description(anomaly_details, readings, lift_type)
    suggestion["suggested_actions"] = build_data_driven_suggested_actions(violations)

    if not (suggestion.get("explanation") or "").strip():
        suggestion["explanation"] = suggestion["description"]

    return suggestion


def format_violations_for_display(violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not violations:
        return violations
    formatted: List[Dict[str, Any]] = []
    for v in violations:
        formatted_v = dict(v)
        for key in ["value", "min", "max"]:
            if key in formatted_v and formatted_v[key] is not None:
                formatted_v[key] = format_number_with_commas(formatted_v[key])
        formatted.append(formatted_v)
    return formatted


def normalize_violations_for_storage(violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Numeric violations with UI aliases (observed, expected_min, expected_max)."""
    if not violations:
        return violations
    return [normalize_violation_record(v) for v in violations if isinstance(v, dict)]


def format_anomaly_details_for_storage(anomaly_details: Dict[str, Any]) -> Dict[str, Any]:
    if not anomaly_details:
        return anomaly_details
    formatted = dict(anomaly_details)
    violations = formatted.get("violations") or []
    if isinstance(violations, list) and violations:
        formatted["violations"] = normalize_violations_for_storage(violations)
    return formatted


def calculate_severity(anomaly_details: Dict[str, Any], anomaly_score: float, readings: Dict[str, Any]) -> str:
    violations = anomaly_details.get("violations", []) or []
    num_violations = len(violations)
    deviations: List[float] = []
    for v in violations:
        try:
            value = v.get("value")
            min_val = v.get("min")
            max_val = v.get("max")
            if value is None or min_val is None or max_val is None:
                continue
            fv = float(value)
            fmin = float(min_val)
            fmax = float(max_val)
            rng = abs(fmax - fmin) if (fmax - fmin) != 0 else 1.0
            deviation = abs(fmin - fv) / rng * 100.0 if fv < fmin else abs(fv - fmax) / rng * 100.0
            deviations.append(deviation)
        except Exception:
            continue
    avg_deviation = float(sum(deviations) / len(deviations)) if deviations else 0.0
    a_score = float(anomaly_score or 0.0)

    if (num_violations >= 3 and (a_score >= 0.85 or avg_deviation > 60)) or a_score >= 0.98:
        return "CRITICAL"
    if num_violations == 2:
        if a_score >= 0.60 or avg_deviation > 40:
            return "HIGH"
        if a_score >= 0.35 or avg_deviation > 20:
            return "MEDIUM"
        return "LOW"
    if num_violations == 1:
        if a_score >= 0.50 or avg_deviation > 25:
            return "HIGH"
        if a_score >= 0.35 or avg_deviation > 12:
            return "MEDIUM"
        return "LOW"
    if a_score >= 0.75:
        return "HIGH"
    if a_score >= 0.35:
        return "MEDIUM"
    return "LOW"


def calculate_data_driven_confidence(anomaly_details: Dict[str, Any]) -> int:
    import math

    rule_score = float(anomaly_details.get("rule_score", 0.0) or 0.0)
    ml_score = float(anomaly_details.get("ml_score", 0.0) or 0.0)
    combined_score = float(anomaly_details.get("anomaly_score", 0.0) or 0.0)
    violations = anomaly_details.get("violations", []) or []

    if not violations:
        return max(1, min(99, int(round(combined_score * 100))))

    total_deviation_weight = 0.0
    deviation_sum = 0.0
    for v in violations:
        dev_pct = float(v.get("deviation_pct", 0) or 0)
        weight = 1.0 + (dev_pct / 50.0)
        deviation_sum += dev_pct * weight
        total_deviation_weight += weight

    avg_weighted_deviation = deviation_sum / total_deviation_weight if total_deviation_weight > 0 else 0
    deviation_factor = math.log1p(avg_weighted_deviation) / math.log1p(100)
    deviation_factor = min(1.0, deviation_factor)

    deviation_component = deviation_factor * 0.40
    rule_component = rule_score * 0.25
    ml_component = ml_score * 0.20
    violation_count_factor = min(1.0, math.sqrt(len(violations)) / 2.0)
    violation_component = violation_count_factor * 0.15

    base_confidence = deviation_component + rule_component + ml_component + violation_component

    variance_seed = sum(float(v.get("deviation_pct", 0) or 0) * (i + 1) for i, v in enumerate(violations))
    micro_variance = (variance_seed % 7) / 100.0

    final_confidence = base_confidence + micro_variance
    conf_percent = int(round(final_confidence * 100))
    return max(1, min(99, conf_percent))


def _openai_client():
    import openai

    app_url = os.getenv("APP_URL", "http://localhost:5000")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return openai.OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE"),
        default_headers={"HTTP-Referer": app_url, "X-Title": "Well Anomaly Detection"},
    )


def _coerce_asset_history_to_strings(historical_context: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible wrapper; canonical normalization happens at persist time."""
    from anomaly_detection.s3_history import _normalize_historical_context

    return _normalize_historical_context(historical_context)


def generate_suggestion(
    anomaly_details: Dict[str, Any],
    well_id: str,
    readings: Dict[str, Any],
    timestamp_str: str,
    historical_anomalies: Optional[List[Dict[str, Any]]] = None,
    lift_type: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        formatted_history = []
        if historical_anomalies:
            for anom in historical_anomalies:
                raw_data = anom.get("raw_anomaly_data")
                if isinstance(raw_data, str):
                    try:
                        raw_data = json.loads(raw_data)
                    except Exception:
                        raw_data = {}

                past_violations_list = []
                if raw_data and isinstance(raw_data.get("violations"), list):
                    for v in raw_data["violations"]:
                        raw_msg = v.get("violation", "")
                        clean_msg = re.sub(r"\s*\(.*?\)", "", raw_msg).strip()
                        past_violations_list.append({"field": v.get("field"), "violation": clean_msg})

                formatted_history.append(
                    {
                        "well": anom.get("well_id"),
                        "date": str(anom.get("timestamp")),
                        "issue": anom.get("alert_title"),
                        "severity": anom.get("severity"),
                        "violations": past_violations_list,
                        "readings": (raw_data or {}).get("full_readings", {}),
                    }
                )

        history_context_str = json.dumps(formatted_history, indent=2, cls=CustomJSONEncoder, default=str)
        history_guidance = ""
        if not formatted_history:
            history_guidance = (
                "\nNOTE: No similar historical incidents found matching the current anomaly within 20% tolerance.\n"
                'IMPORTANT: You MUST still provide a "key_learnings" field stating: "No similar historical incidents found."\n'
            )

        calculated_severity = calculate_severity(anomaly_details, anomaly_details.get("anomaly_score", 0), readings)
        violations = effective_violations(anomaly_details, readings, lift_type)
        formatted_violations = format_violations_for_display(violations)
        data_title = generate_alert_title_from_anomaly(anomaly_details, readings, lift_type)
        data_description = build_data_driven_description(anomaly_details, readings, lift_type)
        data_actions = build_data_driven_suggested_actions(violations)

        sensor_ranges_str = "{}"
        if lift_type and _SENSOR_CONFIG_AVAILABLE and get_sensor_ranges:
            try:
                sensor_ranges = get_sensor_ranges(lift_type)
                sensor_ranges_str = json.dumps(sensor_ranges, indent=2, cls=CustomJSONEncoder, default=str)
            except Exception:
                sensor_ranges_str = "{}"

        tenant_label = tenant_profile_label(organization_id)
        prompt = f"""
You are an expert Production Engineer AI assistant. Analyze the new anomaly.

---
New Anomaly Details:
Well ID: {well_id}
Timestamp: {timestamp_str}
Lift Type: {lift_type or 'N/A'}
Anomaly Score: {float(anomaly_details.get("anomaly_score", 0) or 0):.2f}
Rule Score: {float(anomaly_details.get("rule_score", 0) or 0):.2f}
ML Score: {float(anomaly_details.get("ml_score", 0) or 0):.2f}
LSTM Score: {float(anomaly_details.get("lstm_score", 0) or 0):.2f}
Pre-Classified Severity: {calculated_severity}

Server-Computed Alert Title (authoritative — do not invent a different title):
{data_title}

Server-Computed Description (authoritative baseline — use these exact values/units):
{data_description}

Server-Computed Suggested Actions:
{json.dumps(data_actions, indent=2, cls=CustomJSONEncoder)}

Current Sensor Readings:
{json.dumps(readings, indent=2, cls=CustomJSONEncoder, default=str)}

Violations / Out-of-Band Sensors:
{json.dumps(formatted_violations, indent=2, cls=CustomJSONEncoder, default=str)}

Expected Sensor Bands ({tenant_label} reference mean ± deviation%) for lift type ({lift_type or 'N/A'}):
{sensor_ranges_str}
---
Real Database History for this well (for context):
{history_context_str}

{history_guidance}
---

IMPORTANT:
- alert_title, severity, confidence, description, and suggested_actions are computed from sensor data on the server. Do NOT invent alternate titles or generic labels like "Anomaly Detected".
- Use only values, units, and bands present in Violations / Out-of-Band Sensors and Current Sensor Readings.
- Do not convert units. Use comma-delimited numbers in all text you write.

Provide JSON with ONLY these keys (enrichment fields):

1. "explanation": 2–4 sentences expanding on the server description with likely operational causes and what to verify first. Reference specific sensor names, values, and bands from the data above.

2. "historical_context": Object containing:
   - "asset_history": Object with keys "commissioned" (string), "operating_hours" (string), "last_inspection" (string). Since this data is not provided, you MUST use the string "Data Not Available" for each value.
   - "similar_incidents": A list of objects based ONLY on the 'Real Database History' provided above. If the history is empty, this MUST be an empty list []. For each incident in the history, create an object with the following keys:
     - "well": Use the "well" value from the history.
     - "date": Use the "date" value from the history, use format "MM/DD/YYYY HH:MM".
     - "issue": Use the "issue" value from history when present; otherwise summarize using field, value, and unit from violations.
     - "severity": Use the "severity" value from the history.
     - "violations": Use the "violations" array from history (containing "field" and "violation", violation must be in simple value example 300 A). Do NOT repeat the parameter name or add words like "value" or "Historical" in value.
   - "key_learnings": String field (NOT inside similar_incidents). ALWAYS REQUIRED.
     * If similar_incidents is empty, write: "No similar historical incidents found."
     * If similar_incidents exist, analyze patterns in frequency, severity, or affected parameters (e.g., "Three pressure-related incidents occurred within 24 hours, suggesting potential equipment degradation").

3. "risk_analysis": Object containing:
   - "severity_breakdown": Object with keys "safety_risk", "production_impact", "financial_exposure" — each grounded in the violated sensors and scores above.
   - "mitigation_strategies": Object with keys "immediate", "short_term", "long_term".
   - "decision_support": One-sentence recommendation referencing the top violated sensor and value.

Ensure your entire output is a single, valid JSON object.
"""

        client = _openai_client()
        model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for oil and gas anomaly detection. Output valid JSON only, matching the required schema exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        suggestion_data = json.loads(content.strip())
        apply_data_driven_fields(suggestion_data, anomaly_details, readings, lift_type)

        if "historical_context" not in suggestion_data:
            suggestion_data["historical_context"] = {}
        hc = suggestion_data["historical_context"]
        if "key_learnings" not in hc or not hc["key_learnings"]:
            if not hc.get("similar_incidents"):
                hc["key_learnings"] = (
                    "No similar historical incidents found within 20% tolerance. "
                    "This appears to be a novel anomaly pattern for this well, requiring careful investigation."
                )
            else:
                hc["key_learnings"] = "Historical patterns identified - see similar incidents for details."
        if "similar_incidents" not in hc:
            hc["similar_incidents"] = []
        if "asset_history" not in hc:
            hc["asset_history"] = {
                "commissioned": "Data Not Available",
                "operating_hours": "Data Not Available",
                "last_inspection": "Data Not Available",
            }

        return suggestion_data
    except Exception as e:
        logger.error("Error generating suggestion: %s", e)
        violations = effective_violations(anomaly_details, readings, lift_type)
        fallback = {
            "severity": "UNKNOWN",
            "status": "ACTIVE",
            "confidence": "0%",
            "explanation": f"LLM enrichment unavailable: {str(e)}",
            "historical_context": {
                "asset_history": {
                    "commissioned": "Data Not Available",
                    "operating_hours": "Data Not Available",
                    "last_inspection": "Data Not Available",
                },
                "similar_incidents": [],
                "key_learnings": "LLM enrichment unavailable; see data-driven description.",
            },
            "risk_analysis": {
                "severity_breakdown": {
                    "safety_risk": "Assess based on violated sensors listed in the alert.",
                    "production_impact": "Monitor production if multiple sensors are out of band.",
                    "financial_exposure": "Data Not Available",
                },
                "mitigation_strategies": {
                    "immediate": build_data_driven_suggested_actions(violations)[0]
                    if violations
                    else "Verify sensor readings and field conditions.",
                    "short_term": "Review recent trends for the affected well.",
                    "long_term": "Schedule engineering review if anomalies repeat.",
                },
                "decision_support": generate_alert_title_from_anomaly(
                    anomaly_details, readings, lift_type
                ),
            },
        }
        return apply_data_driven_fields(fallback, anomaly_details, readings, lift_type)


def persist_suggestion_to_s3(
    *,
    anomaly_details: Dict[str, Any],
    suggestion_result: Dict[str, Any],
    well_id: str,
    timestamp_str: str,
    lift_type: str,
    well_status: Optional[str],
    organization_id: Optional[str],
    readings: Optional[Dict[str, Any]] = None,
) -> None:
    suggestion_result = suggestion_result or {}
    suggestion_result["asset_id"] = suggestion_result.get("asset_id") or f"ALT-{random.randint(100000, 999999)}"

    try:
        apply_data_driven_fields(
            suggestion_result,
            anomaly_details,
            readings or {},
            lift_type,
        )
    except Exception:
        pass

    suggestion_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    formatted_anomaly_details = format_anomaly_details_for_storage(anomaly_details)

    record = normalize_anomaly_suggestion_record(
        {
            "id": suggestion_id,
            "well_id": well_id,
            "timestamp": timestamp_str,
            "alert_title": suggestion_result.get("alert_title"),
            "severity": suggestion_result.get("severity"),
            "status": suggestion_result.get("status"),
            "confidence": suggestion_result.get("confidence"),
            "description": suggestion_result.get("description"),
            "suggested_actions": suggestion_result.get("suggested_actions"),
            "explanation": suggestion_result.get("explanation"),
            "created_at": created_at,
            "historical_context": suggestion_result.get("historical_context", {}),
            "risk_analysis": suggestion_result.get("risk_analysis", {}),
            "asset_id": suggestion_result.get("asset_id"),
            "raw_anomaly_data": formatted_anomaly_details,
            "lift_type": lift_type,
            "category": anomaly_details.get("category") or lift_type,
            "well_status": well_status,
            "organization_id": organization_id,
        }
    )
    write_anomaly_suggestions_history_to_s3([record])

