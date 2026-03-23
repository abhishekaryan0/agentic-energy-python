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

from anomaly_detection.s3_history import write_anomaly_suggestions_history_to_s3

logger = logging.getLogger(__name__)

try:
    from config.sensor_config import get_sensor_ranges
    _SENSOR_CONFIG_AVAILABLE = True
except ImportError:
    _SENSOR_CONFIG_AVAILABLE = False
    get_sensor_ranges = None


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


def format_anomaly_details_for_storage(anomaly_details: Dict[str, Any]) -> Dict[str, Any]:
    if not anomaly_details:
        return anomaly_details
    formatted = dict(anomaly_details)
    violations = formatted.get("violations") or []
    if isinstance(violations, list) and violations:
        formatted["violations"] = format_violations_for_display(violations)
    return formatted


def generate_alert_title_from_violations(violations: List[Dict[str, Any]]) -> str:
    if not violations:
        return "Sensor Parameter Anomaly"
    sorted_violations = sorted(
        violations, key=lambda x: float(x.get("deviation_pct", 0) or 0), reverse=True
    )
    top_violation = sorted_violations[0]
    field = top_violation.get("field", "parameter")
    value = top_violation.get("value")
    min_val = top_violation.get("min")
    max_val = top_violation.get("max")
    field_name = format_field_name(field)

    issue_type = "Anomaly"
    try:
        if value is not None and min_val is not None and max_val is not None:
            val = float(value)
            mn = float(min_val)
            mx = float(max_val)
            if val > mx:
                issue_type = "High" if val < mx * 1.5 else "Spike"
            elif val < mn:
                issue_type = "Low" if val > mn * 0.5 else "Drop"
    except (ValueError, TypeError):
        pass
    return f"{field_name} {issue_type}"


def sanitize_alert_title(title: str, violations: List[Dict[str, Any]]) -> str:
    forbidden_terms = ["esp", "rod pump", "gas lift", "rod_pump", "gas_lift"]
    if not title:
        return generate_alert_title_from_violations(violations)
    title_lower = title.lower()
    if any(term in title_lower for term in forbidden_terms):
        return generate_alert_title_from_violations(violations)
    generic_titles = ["anomaly detected", "alert detected", "sensor anomaly detected"]
    if title_lower.strip() in generic_titles:
        return generate_alert_title_from_violations(violations)
    return title.strip()


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
    """
    Ensure historical_context.asset_history.* are always strings.

    This prevents schema drift in S3 parquet across runs (e.g. operating_hours
    being sometimes int and sometimes str).
    """
    asset_history = historical_context.get("asset_history") or {}
    if not isinstance(asset_history, dict):
        asset_history = {}

    def _to_str(v: Any, default: str) -> str:
        if v is None:
            return default
        try:
            if pd.isna(v):
                return default
        except Exception:
            pass
        return str(v)

    historical_context = dict(historical_context)
    historical_context["asset_history"] = {
        "commissioned": _to_str(asset_history.get("commissioned"), "Data Not Available"),
        "operating_hours": _to_str(asset_history.get("operating_hours"), "Data Not Available"),
        "last_inspection": _to_str(asset_history.get("last_inspection"), "Data Not Available"),
    }
    return historical_context


def generate_suggestion(
    anomaly_details: Dict[str, Any],
    well_id: str,
    readings: Dict[str, Any],
    timestamp_str: str,
    historical_anomalies: Optional[List[Dict[str, Any]]] = None,
    lift_type: Optional[str] = None,
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
        formatted_violations = format_violations_for_display(anomaly_details.get("violations", []))

        sensor_ranges_str = "{}"
        if lift_type and _SENSOR_CONFIG_AVAILABLE and get_sensor_ranges:
            try:
                sensor_ranges = get_sensor_ranges(lift_type)
                sensor_ranges_str = json.dumps(sensor_ranges, indent=2, cls=CustomJSONEncoder, default=str)
            except Exception:
                sensor_ranges_str = "{}"

        prompt = f"""
You are an expert Production Engineer AI assistant. Analyze the new anomaly.

---
New Anomaly Details:
Well ID: {well_id}
Timestamp: {timestamp_str}
Current Sensor Readings:
{json.dumps(readings, indent=2, cls=CustomJSONEncoder, default=str)}
Violations Detected:
{json.dumps(formatted_violations, indent=2, cls=CustomJSONEncoder, default=str)}

Authoritative Sensor Ranges for lift type ({lift_type or 'N/A'}):
{sensor_ranges_str}
---
Real Database History for this well (for context):
{history_context_str}

{history_guidance}
---

IMPORTANT: Use the units provided in each violation entry exactly as given. Do NOT convert units (e.g., do not change °F to °C) or invent ranges. Use the 'unit', 'min', and 'max' fields from the Violations Detected objects when describing values and expected ranges.
Additionally, do not use raw parameter keys as names everywhere. Convert them into clear, human-readable labels (for example, rod_pump → Rod Pump).

Based on the information above, provide a structured JSON response.

SEVERITY CLASSIFICATION GUIDE (Authoritative - follow code logic):
- CRITICAL: 3+ violations with strong signal (>=0.85) or avg deviation >60%, or anomaly_score >= 0.98
- HIGH: 2 violations with moderate signal (>=0.60) or avg deviation >40%, or anomaly_score >= 0.75
- MEDIUM: 1 violation with modest signal (>=0.35) or avg deviation >12%
- LOW: otherwise (small deviations or weak ML signal)

CONFIDENCE CALCULATION:
Calculate confidence as a SINGLE percentage value directly derived from the anomaly score.
DO NOT use bucketed ranges. Use this exact rule:
- Compute `confidence_percent = round(anomaly_score * 100)`.
- Clamp `confidence_percent` into the inclusive range 1 to 99.
- Return `confidence` as a string with a percent sign, e.g. "78%".
The returned confidence MUST match this calculation and be a single number string.

PRE-CLASSIFIED SEVERITY: {calculated_severity}
ANOMALY SCORE: {float(anomaly_details.get("anomaly_score", 0) or 0):.2f}
Strictly assign severity level that matches or stays close to this pre-classification.

Important:
1. If 'Real Database History' above is not an empty list, populate 'similar_incidents' using that data.

Required JSON Structure:
1. "alert_title": STRICT RULES - Read carefully:
   A) NEVER include "ESP", "Rod Pump", "Gas Lift", or any lift type name in the title.
   B) ONLY use the violated parameter name from 'Violations Detected' section.
   C) Format: "[Parameter Name] [Issue Type]" - nothing else.

   CORRECT Examples (use these exact patterns):
   - "Torque Overload" (for torque violation)
   - "Strokes Per Minute Anomaly" (for strokes_per_minute violation)
   - "Polish Rod Load Spike" (for polish_rod_load violation)
   - "Pump Fillage Low" (for pump_fillage violation)
   - "Tubing Pressure High" (for tubing_pressure violation)
   - "Motor Current Spike" (for motor_current violation)
   - "Motor Temperature High" (for motor_temp violation)
   - "Discharge Pressure Drop" (for discharge_pressure violation)
   - "Injection Rate Anomaly" (for injection_rate violation)
   - "Bottomhole Pressure Low" (for bottomhole_pressure violation)

   WRONG Examples (NEVER do this):
   - "ESP Motor Current Spike" ← WRONG (contains ESP)
   - "Rod Pump Torque Overload" ← WRONG (contains Rod Pump)
   - "Gas Lift Injection Anomaly" ← WRONG (contains Gas Lift)
   - "ESP Anomaly Detected" ← WRONG (generic and contains ESP)

   Convert parameter keys to human-readable names (e.g., motor_current → Motor Current, strokes_per_minute → Strokes Per Minute).
   Use the MOST CRITICAL violation (highest deviation) from 'Violations Detected' to determine the title.

2. "severity": Assign based on severity matrix above (CRITICAL, HIGH, MEDIUM, or LOW).
3. "status": "ACTIVE".
4. "confidence": IMPORTANT - Single percentage string (e.g., "78%") calculated from anomaly score. MUST be a single number. DO NOT return a range.
5. "description": Write a concise, technical paragraph. Within this paragraph, you MUST clearly describe each parameter from 'Violations Detected' that is out of range.
   For each parameter, state the nature of the anomaly by including its observed value and the expected range, both with their units.
   Only mention violations detected, not all parameters.
   For example: "The motor temperature reached 265 °F, exceeding the expected operational range of 100–250 °F." Use the 'unit' field from each violation object.
6. "suggested_actions": A list of three clear, concise and actionable steps.
7. "explanation": Detailed explanation of the new anomaly.

CRITICAL FORMATTING REQUIREMENT FOR ALL TEXT FIELDS (description, suggested_actions, explanation):
Use comma-delimited numbers (e.g., "1,234,567.89" NOT "1234567.89"). This applies to all numeric values.
Use the exact formatted values provided in the 'Violations Detected' section above - they already have proper comma delimiters.

8. "historical_context": Object containing:
   - "asset_history": Object with keys "commissioned" (date), "operating_hours" (int), "last_inspection" (date). Since this data is not provided, you MUST use the string "Data Not Available" for each value.
   - "similar_incidents": A list of objects based ONLY on the 'Real Database History' provided above. If the history is empty, this MUST be an empty list []. For each incident in the history, create an object with the following keys:
     - "well": Use the "well" value from the history.
     - "date": Use the "date" value from the history, use format "MM/DD/YYYY HH:MM".
     - "issue": Summarize the technical issue briefly (e.g., "Motor Current High" or "Pressure Violation"). Do NOT copy long "Out of range" error strings.
     - "severity": Use the "severity" value from the history.
     - "violations": Use the "violations" array from history (containing "field" and "violation", violation must be in simple value example 300 A). Do NOT repeat the parameter name or add words like "value" or "Historical" in value.
   - "key_learnings": String field (NOT inside similar_incidents). ALWAYS REQUIRED.
     * If similar_incidents is empty, write: "No similar historical incidents found."
     * If similar_incidents exist, analyze patterns in frequency, severity, or affected parameters (e.g., "Three pressure-related incidents occurred within 24 hours, suggesting potential equipment degradation").

9. "risk_analysis": Object containing:
   - "severity_breakdown": Object with keys "safety_risk", "production_impact", "financial_exposure".
   - "mitigation_strategies": Object with keys "immediate", "short_term", "long_term".
   - "decision_support": A final string recommendation.

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
        )

        content = response.choices[0].message.content or ""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        suggestion_data = json.loads(content.strip())
        if "alert_title" in suggestion_data:
            suggestion_data["alert_title"] = sanitize_alert_title(
                suggestion_data.get("alert_title", ""), anomaly_details.get("violations", [])
            )

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
        violations = anomaly_details.get("violations", [])
        fallback_title = generate_alert_title_from_violations(violations)
        return {
            "alert_title": fallback_title,
            "severity": "UNKNOWN",
            "status": "ACTIVE",
            "confidence": "0%",
            "description": "Could not generate detailed analysis.",
            "suggested_actions": ["Investigate manually"],
            "explanation": f"AI Generation failed: {str(e)}",
            "historical_context": {
                "asset_history": {"commissioned": "N/A", "operating_hours": "N/A", "last_inspection": "N/A"},
                "similar_incidents": [],
                "key_learnings": "AI generation failed.",
            },
            "risk_analysis": {},
        }


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
        suggestion_result["severity"] = calculate_severity(
            anomaly_details, anomaly_details.get("anomaly_score", 0), readings or {}
        )
        suggestion_result["confidence"] = f"{calculate_data_driven_confidence(anomaly_details)}%"
    except Exception:
        pass

    suggestion_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    formatted_anomaly_details = format_anomaly_details_for_storage(anomaly_details)

    # Normalize schema types before writing parquet.
    # Specifically: historical_context.asset_history.operating_hours must be string.
    suggestion_result["historical_context"] = _coerce_asset_history_to_strings(
        suggestion_result.get("historical_context", {}) or {}
    )

    record = {
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
        "reason": None,
        "raw_anomaly_data": formatted_anomaly_details,
        "lift_type": lift_type,
        "category": anomaly_details.get("category"),
        "well_status": well_status,
        "organization_id": organization_id,
    }
    write_anomaly_suggestions_history_to_s3([record])

