"""
Operational recommendations and suggestions.
Storage: S3 Parquet only. Numeric fields follow ML/EIA models first; OpenRouter enriches text only.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
import openai

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI Configuration
api_key = os.getenv("OPENAI_API_KEY")
APP_URL = os.getenv("APP_URL", "http://localhost:5000")

if not api_key:
    logger.error("OPENAI_API_KEY is missing in .env")

client = openai.OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_API_BASE"),
    default_headers={
        "HTTP-Referer": APP_URL,
        "X-Title": "Well Anomaly Detection",
    }
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-3.5-turbo")


def generate_operation_suggestion_with_openai(well_id: str, action: str,
                                              production_data: dict = None,
                                              sensor_metrics: dict = None) -> dict:
    """
    Generate detailed operation suggestion using OpenAI.

    Args:
        well_id: Well identifier
        action: Recommended action (e.g., "Increase ESP frequency")
        production_data: Dict with current production metrics
        sensor_metrics: Dict with sensor readings

    Returns:
        Dict with detailed suggestion data
    """
    try:
        context = f"Well: {well_id}\nRecommended Action: {action}\n"

        if production_data:
            context += "\nProduction Data:\n"
            for key, value in production_data.items():
                context += f"  - {key}: {value}\n"

        if sensor_metrics:
            context += "\nSensor Metrics:\n"
            for key, value in sensor_metrics.items():
                context += f"  - {key}: {value}\n"

        prompt = f"""You are an expert petroleum engineer and reservoir analyst. Generate a detailed operational recommendation for an oil well.

{context}

Provide a comprehensive analysis as a single JSON object matching the schema below.

CRITICAL: alternative_models must be CONTENT ONLY—no section headers (no "CONSERVATIVE APPROACH:", "AGGRESSIVE OPTIMIZATION:", "HYBRID MODEL:"). No "RECOMMENDED:" prefix on recommended_model. Start each approach with bullets (•) only.

Use the style guide below only as a reference—tailor all content, numbers, and specifics to the given well and action.

conservative_approach: 3-4 bullets—incremental/safe change, risk, expected uplift % and timeframe.
aggressive_optimization: 3-4 bullets—immediate/bold change, higher return + monitoring, expected uplift % and timeframe.
hybrid_model: 3-4 bullets—phased + real-time/AI, expected uplift % and timeframe.
recommended_model: One sentence picking the best option and why—specific to this well, not generic.
time_reduced_hours: Estimated hours until measurable improvement (e.g., 24, 48, 72). Must be a real numeric estimate, never 0.

CITATIONS: Use ONLY AUTHENTIC, NON-FABRICATED data.
- referenced_data: Array of 2-4 strings using only generic, authentic sources.
- supporting_analysis: One paragraph of technical analysis specific to this well.
- data_quality: One line with confidence and basis.

FINANCIAL (placeholders): Downstream replaces USD fields using EIA Brent spot and ML models. Treat implementation_cost_usd as total ONE-TIME implementation cost (USD lump sum), not a daily rate.

Schema (JSON):
{{
    "production_impact": {{ "increase_bbl_day": <number>, "increase_percent": <number>, "confidence_percent": <number> }},
    "financial_impact": {{ "daily_expense_benefit_usd": <number>, "implementation_cost_usd": <number>, "net_daily_benefit_usd": <number>, "asset_value_increase_usd": <number> }},
    "operational_details": {{ "time_reduced_hours": <realistic number e.g. 24>, "current_performance": "<description>", "historical_trend": "<description — recent production / trend context>", "detailed_analysis": "<text>" }},
    "alternative_models": {{
        "conservative_approach": "<bullets only>",
        "aggressive_optimization": "<bullets only>",
        "hybrid_model": "<bullets only>",
        "recommended_model": "<one sentence only>"
    }},
    "citations": {{
        "referenced_data": ["<string>", "<string>"],
        "supporting_analysis": "<paragraph specific to this well>",
        "data_quality": "<e.g. High confidence based on historical data>"
    }},
    "category": "Production|Optimization|Maintenance"
}}

Return valid JSON only. Preserve newlines and bullet characters (•). Output content only—no headers.
"""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a petroleum engineer. Output valid JSON only matching the required schema."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        suggestion_data = json.loads(content.strip())
        return suggestion_data

    except Exception as e:
        logger.error(f"Error generating suggestion: {e}")
        return {
            "production_impact": {"increase_bbl_day": 0, "increase_percent": 0, "confidence_percent": 0},
            "financial_impact": {"daily_expense_benefit_usd": 0, "implementation_cost_usd": 0, "net_daily_benefit_usd": 0, "asset_value_increase_usd": 0},
            "operational_details": {"time_reduced_hours": 0, "current_performance": "Unknown", "historical_trend": "Unknown", "detailed_analysis": f"AI Generation failed: {str(e)}"},
            "alternative_models": {"conservative_approach": "N/A", "aggressive_optimization": "N/A", "hybrid_model": "N/A", "recommended_model": "N/A"},
            "citations": {"referenced_data": [], "supporting_analysis": "N/A", "data_quality": "Low"}
        }


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        fv = float(val)
        if fv != fv:  # NaN
            return None
        return fv
    except Exception:
        return None


def _fmt_evidence_num(val: Any, digits: int = 2) -> Optional[str]:
    fv = _safe_float(val)
    if fv is None:
        return None
    if abs(fv) >= 1000:
        return f"{fv:,.0f}"
    if abs(fv) >= 10 or digits <= 0:
        return f"{fv:,.1f}" if digits == 1 else f"{fv:,.0f}"
    return f"{fv:.{digits}f}"


def _humanize_metric_key(key: str) -> str:
    name = str(key or "").replace("_", " ").strip().lower()
    replacements = (
        ("oil volume", "oil"),
        ("gas volume", "gas"),
        ("water volume", "water"),
        ("roll mean", "average"),
        ("freq hz", "frequency"),
        ("drive frequency", "drive frequency"),
        ("surface pressure", "surface pressure"),
        ("tubing pressure", "tubing pressure"),
        ("casing pressure", "casing pressure"),
        ("discharge pressure", "discharge pressure"),
        ("strokes", "strokes per minute"),
    )
    for src, dst in replacements:
        if src in name:
            name = name.replace(src, dst)
    return name


def _extract_metric_snapshot(metrics: dict) -> dict:
    """Pull the strongest readable facts from raw metrics for the LLM."""
    if not isinstance(metrics, dict):
        return {}

    def first(*keys: str) -> Optional[float]:
        for k in keys:
            if k in metrics:
                fv = _safe_float(metrics.get(k))
                if fv is not None:
                    return fv
        return None

    oil_avg = first("oil_volume_roll_mean", "oil_volume_avg", "oil_volume_mean", "oil_volume")
    oil_latest = first("oil_volume_latest", "oil_volume_last", "oil_volume")
    oil_trend = first("oil_volume_trend", "oil_volume_slope")
    gas_avg = first("gas_volume_roll_mean", "gas_volume_avg", "gas_volume_mean", "gas_volume")
    gas_trend = first("gas_volume_trend", "gas_volume_slope")
    water_avg = first("water_volume_roll_mean", "water_volume_avg", "water_volume_mean", "water_volume")
    water_trend = first("water_volume_trend", "water_volume_slope")

    production = {}
    if oil_avg is not None:
        production["oil_avg_bbl_day"] = round(oil_avg, 2)
    if oil_latest is not None and (oil_avg is None or abs(oil_latest - oil_avg) > 1e-6):
        production["oil_latest_bbl_day"] = round(oil_latest, 2)
    if oil_trend is not None:
        production["oil_trend_bbl_day_per_day"] = round(oil_trend, 3)
    if gas_avg is not None:
        production["gas_avg"] = round(gas_avg, 2)
    if gas_trend is not None:
        production["gas_trend"] = round(gas_trend, 3)
    if water_avg is not None:
        production["water_avg_bbl_day"] = round(water_avg, 2)
    if water_trend is not None:
        production["water_trend_bbl_day_per_day"] = round(water_trend, 3)

    if oil_avg is not None and water_avg is not None and (oil_avg + water_avg) > 0:
        production["water_cut_pct_approx"] = round(100.0 * water_avg / (oil_avg + water_avg), 1)

    sensor_candidates = []
    skip_prefixes = ("oil_volume", "gas_volume", "water_volume")
    for k, v in metrics.items():
        if any(str(k).startswith(p) for p in skip_prefixes):
            continue
        fv = _safe_float(v)
        if fv is None:
            continue
        score = abs(fv)
        if any(s in str(k) for s in ("trend", "slope", "delta", "change", "pct", "percent")):
            score = abs(fv) * 10.0
        sensor_candidates.append((score, str(k), fv))
    sensor_candidates.sort(key=lambda x: x[0], reverse=True)

    top_sensors = []
    for _, k, fv in sensor_candidates[:8]:
        top_sensors.append(
            {
                "metric": k,
                "label": _humanize_metric_key(k),
                "value": round(fv, 3),
                "display": _fmt_evidence_num(fv, 2),
            }
        )

    return {
        "production": production,
        "top_sensors": top_sensors,
    }


def _build_analysis_context(recommendation: dict) -> dict:
    """Build bounded JSON context for OpenRouter analysis field generation."""
    well_id = recommendation.get("well_id")
    action = recommendation.get("action")
    metrics = recommendation.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    if "production" in metrics or "sensors" in metrics:
        merged = {}
        if isinstance(metrics.get("production"), dict):
            merged.update(metrics["production"])
        if isinstance(metrics.get("sensors"), dict):
            merged.update(metrics["sensors"])
        metrics = merged

    numeric_summary = {}
    for key in (
        "production_increase_bbl_day",
        "production_increase_percent",
        "daily_expense_benefit_usd",
        "implementation_cost_usd",
        "net_daily_benefit_usd",
        "asset_value_increase_usd",
        "time_reduced_hours",
    ):
        fv = _safe_float(recommendation.get(key))
        if fv is not None and abs(fv) > 1e-9:
            numeric_summary[key] = round(fv, 2)

    key_metrics = {}
    important_prefixes = (
        "oil_volume", "gas_volume", "water_volume", "motor_", "surface_pressure",
        "tubing_pressure", "casing_pressure", "discharge_pressure", "freq_hz",
        "drive_frequency", "pump_", "injection_", "vibration_", "torque", "strokes_",
        "intake_", "amp", "current", "load", "choke", "gl_", "esp_",
    )
    important_suffixes = (
        "_avg", "_mean", "_roll_mean", "_trend", "_slope", "_latest", "_last",
        "_pct", "_percent", "_std", "_min", "_max", "_delta", "_change",
    )
    for k, v in metrics.items():
        if any(str(k).startswith(p) for p in important_prefixes) or any(
            str(k).endswith(s) for s in important_suffixes
        ):
            fv = _safe_float(v)
            key_metrics[k] = round(fv, 3) if fv is not None else v
    if len(key_metrics) < 12:
        for k, v in metrics.items():
            if k in key_metrics:
                continue
            fv = _safe_float(v)
            key_metrics[k] = round(fv, 3) if fv is not None else v
            if len(key_metrics) >= 50:
                break

    conf_pct = _safe_float(recommendation.get("confidence_percent"))
    if conf_pct is None:
        conf = _safe_float(recommendation.get("confidence"))
        if conf is not None:
            conf_pct = conf * 100.0 if conf <= 1.0 else conf

    return {
        "well_id": well_id,
        "action": action,
        "priority": recommendation.get("priority"),
        "category": recommendation.get("category"),
        "reason": recommendation.get("reason"),
        "expected_impact": recommendation.get("expected_impact"),
        "confidence_percent": round(conf_pct, 1) if conf_pct is not None else None,
        "detailed_analysis": recommendation.get("detailed_analysis"),
        "current_performance": recommendation.get("current_performance"),
        "historical_trend": recommendation.get("historical_trend"),
        "numeric_summary": numeric_summary,
        "evidence_snapshot": _extract_metric_snapshot(metrics),
        "metrics": key_metrics,
    }


ANALYSIS_FIELD_KEYS: tuple[str, ...] = (
    "analysis_description",
    "sensor_evidence_description",
    "operational_interpretation_description",
    "recommended_next_step_description",
)

# Extra OpenRouter fields applied onto the recommendation record.
ANALYSIS_ENRICHMENT_KEYS: tuple[str, ...] = ANALYSIS_FIELD_KEYS + (
    "action",
    "detailed_analysis",
    "category",
)


def generate_analysis_fields_with_openai(recommendation: dict) -> dict[str, str]:
    """
    Generate data-driven analysis text plus a smart action and detailed_analysis.
    Never returns static template text; all content is OpenRouter-generated from data.
    """
    if not isinstance(recommendation, dict):
        raise ValueError("recommendation must be a dict")

    ml_hint = recommendation.get("action") or recommendation.get("ml_action_hint") or ""
    context = _build_analysis_context(recommendation)
    context["ml_action_hint"] = ml_hint
    context["existing_category"] = recommendation.get("category")
    context["existing_priority"] = recommendation.get("priority")

    prompt = f"""You are writing operator-facing content for ONE oilfield recommendation.
Be smart, specific, and data-driven. Do NOT use static/template wording.

Return JSON only with these keys:
- action
- category
- detailed_analysis
- analysis_description
- sensor_evidence_description
- operational_interpretation_description
- recommended_next_step_description

QUALITY BAR:
- Sound like a sharp production engineer for THIS well and THIS data.
- Different wells/actions must read differently.
- Prefer evidence_snapshot + numeric_summary + reason.
- Use ONLY real numbers from the JSON. Never invent sensors, ranges, or percentages.
- If a value is missing, skip it.
- All narrative fields must agree with the chosen action and the same numbers.

WRITE THESE FIELDS:

1) action
   - Create a smart, specific recommended action for this well from the data.
   - ml_action_hint is only a weak hint from the ML model — improve or replace it when data supports a better action.
   - Prefer concrete operational phrasing with numbers when available, e.g.:
     "Increase ESP frequency from 55 Hz to 58 Hz"
     "Raise gas injection from 2.5 to 3.2 MMSCF/day"
     "Reduce SPM from 12 to 10 to protect rod load"
     "Monitor water cut and review lift settings"
   - Keep action short: about 6-14 words. No generic filler like "Investigate well metrics".
   - Must fit the evidence (do not recommend ESP changes if no ESP/frequency evidence exists).

2) category
   - One of: Production, Optimization, Maintenance
   - Choose based on the action you wrote.

3) detailed_analysis
   - ONE short sentence only (about 50-120 characters, similar length to current UI blurbs).
   - State the core issue + recommended response using real numbers when present.
   - Examples of length/style (do not copy wording):
     "Oil falling while water rises; tighten water monitoring and lift review."
     "Motor load high at current Hz; step frequency down and recheck amps."
   - Unique per recommendation. Never reuse canned template text.

4) analysis_description
   - One flowing paragraph (3-5 sentences, ~70-140 words). No bullets/headings.
   - Naturally reference the action you chose.
   - Explain strongest evidence with clean numbers.
   - If numeric_summary has benefit/cost/uplift, weave in one practical impact clause.
   - Close with one concrete field check for this action.
   - Avoid repeating the same canned openers across wells.

5) sensor_evidence_description
   - One compact sentence of the key numbers used in analysis_description.
   - Empty string only when no numeric evidence exists.

6) operational_interpretation_description
   - One sentence: what this pattern means for operations on this well.
   - Must match analysis_description (no new diagnosis).

7) recommended_next_step_description
   - One actionable sentence matching the closing idea in analysis_description.
   - Specific to the action you chose.

NUMBER STYLE:
- Use commas for thousands.
- Prefer 1 decimal for rates/trends when useful; whole numbers for large USD values.
- Plain text only.

Recommendation data (JSON):
{json.dumps(context, default=str)}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior petroleum production engineer and excellent technical writer. "
                    "Return valid JSON only. Invent a smart specific action from the data "
                    "(ml_action_hint is optional guidance only). Write unique detailed_analysis "
                    "and narrative fields. Never invent numeric values. Keep all fields consistent."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.45,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("OpenRouter returned empty analysis fields")

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenRouter analysis response was not a JSON object")

    out: dict[str, str] = {}
    for key in ANALYSIS_ENRICHMENT_KEYS:
        value = parsed.get(key)
        if value is None:
            out[key] = ""
        else:
            out[key] = str(value).strip()

    if not out.get("analysis_description"):
        raise RuntimeError("OpenRouter returned empty analysis_description")
    if not out.get("action"):
        # Keep prior hint only if model omitted action entirely.
        out["action"] = str(ml_hint or "").strip()
    if not out.get("detailed_analysis"):
        raise RuntimeError("OpenRouter returned empty detailed_analysis")

    # Keep detailed_analysis near current UI length.
    da = out["detailed_analysis"]
    if len(da) > 160:
        first = da.split(". ")[0].strip()
        out["detailed_analysis"] = first if first.endswith(".") else (first + ".")
        if len(out["detailed_analysis"]) > 160:
            out["detailed_analysis"] = out["detailed_analysis"][:157].rstrip() + "..."

    cat = (out.get("category") or "").strip().lower()
    if cat == "production":
        out["category"] = "Production"
    elif cat == "optimization":
        out["category"] = "Optimization"
    elif cat == "maintenance":
        out["category"] = "Maintenance"
    else:
        out.pop("category", None)

    return out


def generate_analysis_description_with_openai(recommendation: dict) -> str:
    """Backward-compatible helper returning only analysis_description."""
    return generate_analysis_fields_with_openai(recommendation)["analysis_description"]


def infer_action_category_priority(well_id: str, production_data: dict = None, sensor_metrics: dict = None, hint: str = None) -> dict:
    """Use OpenAI to infer a concise `action`, `category` and `priority` for a well given data."""
    try:
        context = f"Well: {well_id}\n"
        if production_data:
            context += "Production:\n"
            for k, v in (production_data or {}).items():
                context += f" - {k}: {v}\n"
        if sensor_metrics:
            context += "Sensors:\n"
            for k, v in (sensor_metrics or {}).items():
                context += f" - {k}: {v}\n"
        if hint:
            context += f"\nModel hint: {hint}\n"

        prompt = f"""You are a petroleum engineer. Given the well context, choose a single concise recommended action (short phrase), a high-level category from [Production, Optimization, Maintenance], and a priority from [HIGH, MEDIUM, LOW]. Respond as JSON with keys: action, category, priority.

    Preferred action style:
    - "Increase ESP frequency from 55Hz to 58Hz"
    - "Increase gas injection rate from 2.5 MMSCF to 3.2 MMSCF"
    - "Replace impeller and perform alignment check within 7 days"
    - "Reduce separator pressure from 125 PSI to 110 PSI"
    - "Reduce pumping speed from 12 SPM to 10 SPM immediately"

    Context:
    {context}

    Example output:
    {{"action": "Increase ESP frequency from 55Hz to 58Hz", "category": "Production", "priority": "HIGH"}}
    """

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert petroleum engineer. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        return {
            'action': parsed.get('action'),
            'category': parsed.get('category'),
            'priority': parsed.get('priority')
        }
    except Exception as e:
        logger.warning(f"OpenAI inference failed, falling back to hint/mapping: {e}")
        fallback_action = (hint or 'Investigate well metrics')
        act_low = (fallback_action or '').lower()
        if any(k in act_low for k in ('increase', 'esp', 'pump', 'choke', 'polish')):
            fallback_category = 'Production'
        elif any(k in act_low for k in ('optimi', 'reduce', 'chemical', 'cost', 'efficien')):
            fallback_category = 'Optimization'
        elif any(k in act_low for k in ('inspect', 'maintenance', 'fail', 'fault', 'cable', 'replace')):
            fallback_category = 'Maintenance'
        else:
            fallback_category = 'Production'
        return {'action': fallback_action, 'category': fallback_category, 'priority': 'MEDIUM'}


def refine_suggestion_with_openai(well_id: str, suggested_action: str, production_data: dict = None,
                                   sensor_metrics: dict = None, model_confidence: float = None,
                                   category_hint: str = None) -> dict:
    """
    Ask OpenAI to refine or confirm the action label, pick a category, and suggest a priority.
    Returns: {'action': str, 'category': str, 'priority': 'HIGH'|'MEDIUM'|'LOW'}
    """
    try:
        context = f"Well: {well_id}\nSuggested action: {suggested_action}\nModel confidence: {model_confidence}\n"
        if production_data:
            context += "\nProduction:\n"
            for k, v in (production_data or {}).items():
                context += f" - {k}: {v}\n"
        if sensor_metrics:
            context += "\nSensors:\n"
            for k, v in (sensor_metrics or {}).items():
                context += f" - {k}: {v}\n"

        prompt = f"""
    You are a petroleum engineer advisor. Given the suggested action and recent production/sensor metrics, decide whether to keep or adjust the action label, assign one of the categories [Production, Optimization, Maintenance], and choose a priority [HIGH, MEDIUM, LOW].

    Return a JSON object with keys: action, category, priority. Keep values concise. Use operational phrasing like:
    - "Increase ESP frequency from 55Hz to 58Hz"
    - "Increase gas injection rate from 2.5 MMSCF to 3.2 MMSCF"
    - "Replace impeller and perform alignment check within 7 days"

    Context:
    {context}

    Respond with JSON only.
    """

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise technical advisor. Respond with JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        obj = json.loads(response.choices[0].message.content)
        action = obj.get('action') or suggested_action
        category = obj.get('category') or ''
        priority = obj.get('priority') or ('HIGH' if (model_confidence or 0) > 0.85 else 'MEDIUM')
        valid_cats = {'Production', 'Optimization', 'Maintenance'}
        if category not in valid_cats:
            category = category_hint if category_hint in valid_cats else 'Production'
        return {'action': action, 'category': category, 'priority': priority}
    except Exception as e:
        logger.warning(f"Refine OpenAI call failed: {e}")
        fallback_cat = category_hint if category_hint in {'Production', 'Optimization', 'Maintenance'} else 'Production'
        return {'action': suggested_action, 'category': fallback_cat, 'priority': ('HIGH' if (model_confidence or 0) > 0.85 else 'MEDIUM')}


def _strip_alternative_headers(text: str) -> str:
    """Remove section titles and RECOMMENDED: prefix so we store only content."""
    if not text or not isinstance(text, str):
        return (text or "").strip()
    lines = text.strip().split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if re.match(r"^[A-Z0-9][A-Z0-9\s\-\.]+:\s*$", stripped) and len(stripped) < 60:
            continue
        if stripped.upper().startswith("RECOMMENDED:"):
            stripped = stripped[12:].strip()
        out.append(stripped)
    return "\n".join(out).strip()


def save_operation_suggestion(well_id: str, action: str, category: str = "Production", status: str = "New",
                              priority: str = "HIGH", confidence: float = 0.0,
                              production_data: dict = None, sensor_metrics: dict = None,
                              reason: str = "", expected_impact: str = "") -> bool:
    """
    Save an operation suggestion to S3 Parquet.

    Production and USD fields are taken from the same metric + EIA + financial model path as
    ``recommend_from_metrics`` (deterministic). OpenRouter is used only for narrative fields
    (analysis, alternatives, citations). Schema types match ``write_operations_recommendations_to_s3``.
    """
    try:
        def _t(s, n):
            try:
                if s is None:
                    return None
                s2 = str(s)
                return s2 if len(s2) <= n else s2[:n]
            except Exception:
                return s

        def _nf(v, default=0.0) -> float:
            try:
                if v is None:
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        from operation_recommendations.recommendations.persistence import (
            compute_financial_impact_fields,
            derive_production_impact_from_metrics,
            merge_metrics_for_financial_impact,
        )

        merged_m = merge_metrics_for_financial_impact(production_data, sensor_metrics)
        prod_imp = derive_production_impact_from_metrics(merged_m)
        pi_bbl = prod_imp.get("production_increase_bbl_day")
        fin_model = compute_financial_impact_fields(merged_m, pi_bbl)
        fin_model.pop("_financial_meta", None)

        conf_float = _nf(confidence, 0.0)
        row = {
            "id": str(uuid4()),
            "well_id": well_id,
            "category": _t(category, 50),
            "action": _t(action, 255),
            "status": status,
            "priority": priority,
            "confidence_percent": conf_float,
            "confidence": conf_float / 100.0 if conf_float > 1 else conf_float,
            "probability": None,
            "reason": _t(reason, 2000),
            "expected_impact": _t(expected_impact, 500),
            "production_increase_bbl_day": _nf(pi_bbl, 0.0),
            "production_increase_percent": _nf(prod_imp.get("production_increase_percent"), 0.0),
            "daily_expense_benefit_usd": _nf(fin_model.get("daily_expense_benefit_usd"), 0.0),
            "implementation_cost_usd": _nf(fin_model.get("implementation_cost_usd"), 0.0),
            "net_daily_benefit_usd": _nf(fin_model.get("net_daily_benefit_usd"), 0.0),
            "asset_value_increase_usd": _nf(fin_model.get("asset_value_increase_usd"), 0.0),
            "time_reduced_hours": None,
            "detailed_analysis": "",
            "current_performance": "",
            "historical_trend": "",
            "conservative_approach": "",
            "aggressive_optimization": "",
            "hybrid_model": "",
            "recommended_model": "",
            "citations": json.dumps([]),
            "referenced_data": json.dumps([]),
            "supporting_analysis": "",
            "data_quality": "",
            "status_reason": None,
            "prediction_source": "ml_model_openrouter_text",
            "mute_till": None,
            "open_up": None,
            "metrics": {"production": production_data or {}, "sensors": sensor_metrics or {}},
            "user_id": None,
            "user_name": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # OpenRouter: refine wording (optional); does not override numeric fields above.
        try:
            refined = refine_suggestion_with_openai(
                well_id,
                action,
                production_data,
                sensor_metrics,
                model_confidence=(confidence / 100.0 if confidence and confidence > 1 else confidence),
                category_hint=category,
            )
            if isinstance(refined, dict):
                action = refined.get("action") or action
                priority = refined.get("priority") or priority
                ref_cat = refined.get("category")
                if ref_cat in {"Production", "Optimization", "Maintenance"}:
                    category = ref_cat
                row["action"] = _t(action, 255)
                row["category"] = _t(category, 50)
                row["priority"] = priority
        except Exception:
            logger.warning("Refine suggestion step failed; continuing with provided action/category")

        logger.info("Generating narrative (OpenRouter) for %s (action=%s)...", well_id, row["action"])
        suggestion = generate_operation_suggestion_with_openai(
            well_id, row["action"], production_data, sensor_metrics
        )

        op_details = suggestion.get("operational_details", {}) or {}
        alt_models = suggestion.get("alternative_models", {}) or {}
        citations = suggestion.get("citations", {}) or {}

        alt_models = dict(alt_models) if alt_models else {}
        for key in ("conservative_approach", "aggressive_optimization", "hybrid_model"):
            if key in alt_models and alt_models[key]:
                alt_models[key] = _strip_alternative_headers(alt_models[key])
        if alt_models.get("recommended_model"):
            alt_models["recommended_model"] = _strip_alternative_headers(alt_models["recommended_model"])

        # Prefer deterministic numeric fields (set earlier in `row`). Only use OpenRouter if missing/0.
        time_reduced = 0.0
        try:
            existing = float(row.get("time_reduced_hours") or 0.0)
        except Exception:
            existing = 0.0
        try:
            llm_val = float(op_details.get("time_reduced_hours") or 0.0)
        except Exception:
            llm_val = 0.0
        if existing > 0.0:
            time_reduced = existing
        elif llm_val > 0.0:
            time_reduced = llm_val
        else:
            time_reduced = 0.0

        row["detailed_analysis"] = _t(op_details.get("detailed_analysis", ""), 2000)
        row["current_performance"] = _t(op_details.get("current_performance", ""), 255)
        row["historical_trend"] = _t(
            op_details.get("historical_trend") or op_details.get("optimal_performance", ""), 255
        )
        row["conservative_approach"] = _t(alt_models.get("conservative_approach", ""), 2000)
        row["aggressive_optimization"] = _t(alt_models.get("aggressive_optimization", ""), 2000)
        row["hybrid_model"] = _t(alt_models.get("hybrid_model", ""), 2000)
        row["recommended_model"] = _t(alt_models.get("recommended_model", ""), 500)
        row["citations"] = json.dumps(citations.get("referenced_data", []))
        row["referenced_data"] = json.dumps(citations.get("referenced_data", []))
        row["supporting_analysis"] = _t(citations.get("supporting_analysis", ""), 2000)
        row["data_quality"] = _t(citations.get("data_quality", ""), 100)
        row["time_reduced_hours"] = float(round(time_reduced, 2)) if time_reduced > 0 else 0.0

        from operation_recommendations.recommendations.persistence import (
            apply_analysis_description,
            apply_metric_descriptions,
            write_operations_recommendations_to_s3,
        )
        apply_metric_descriptions(row)
        apply_analysis_description(row)
        written_key = write_operations_recommendations_to_s3([row])
        if written_key:
            logger.info(f"✓ Saved operation suggestion for {well_id}: {action} -> S3")
            return True
        else:
            logger.error(f"✗ Failed to write suggestion for {well_id} to S3")
            return False

    except Exception as e:
        logger.error(f"Error saving operation suggestion: {e}", exc_info=True)
        return False


def get_operation_suggestions(well_id: str = None, status: str = None,
                              priority: str = None, category: str = None, limit: int = 100) -> list:
    """Suggestions are stored in S3 Parquet. Live querying is not supported here."""
    return []


def update_operation_suggestion_status(suggestion_id: str, status: str) -> bool:
    """Suggestions are stored in S3 Parquet. Status updates are not supported here."""
    return False


def get_operation_suggestion_detail(suggestion_id: str) -> dict:
    """Suggestions are stored in S3 Parquet. Detail lookup is not supported here."""
    return {}


def stream_run_operational_pipeline(confidence: str = "0.2"):
    """Spawn the operational pipeline script and stream logs as Server-Sent Events (SSE).

    Yields SSE `data: ...\\n\\n` chunks containing JSON objects with either `line`,
    or `status`/`returncode`/`error` fields.
    """
    process = None
    try:
        venv_py = os.path.join(os.getcwd(), '.venv', 'bin', 'python')
        if not os.path.exists(venv_py):
            venv_py = 'python'

        _script_dir = Path(__file__).resolve().parent
        script_path = _script_dir / 'run_pipeline.py'
        cmd = [venv_py, str(script_path), '--confidence', str(confidence)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        processing_re = re.compile(r"Processing:\s*(?P<well_id>[^-]+)\s*-\s*(?P<action>.+?)\s*\(confidence:\s*(?P<conf>[\d%.]+)\)")
        saved_s3_re = re.compile(r"✓ Saved operation suggestion.+S3")
        failed_save_re = re.compile(r"✗ Failed|Failed to save")
        predictions_re = re.compile(r"✓ Predictions completed:\s*(?P<count>\d+) recommendations detected")
        train_acc_re = re.compile(r"Train Accuracy: (.+)")
        summary_key_re = re.compile(r"^(Recommendations Processed:|Saved to S3:|Failed:)\s*(?P<num>\d+)")

        last_ctx = {'well_id': None, 'action': None}

        if process.stdout is not None:
            for raw_line in iter(process.stdout.readline, ''):
                if raw_line is None:
                    break
                line = raw_line.rstrip('\n').strip()
                if not line:
                    continue

                if 'ERROR' in line or 'Traceback' in line or 'Exception' in line:
                    payload = json.dumps({'type': 'error', 'message': line})
                    yield f"data: {payload}\n\n"
                    continue

                m = processing_re.search(line)
                if m:
                    well = m.group('well_id').strip()
                    action = m.group('action').strip()
                    conf = m.group('conf').strip()
                    last_ctx['well_id'] = well
                    last_ctx['action'] = action
                    payload = json.dumps({'type': 'processing', 'well_id': well, 'action': action, 'confidence': conf})
                    yield f"data: {payload}\n\n"
                    continue

                if saved_s3_re.search(line):
                    payload = json.dumps({'type': 'saved', 'well_id': last_ctx.get('well_id'), 'action': last_ctx.get('action'), 'message': 'saved_s3'})
                    yield f"data: {payload}\n\n"
                    continue

                if failed_save_re.search(line):
                    payload = json.dumps({'type': 'failed_save', 'well_id': last_ctx.get('well_id'), 'action': last_ctx.get('action'), 'message': line})
                    yield f"data: {payload}\n\n"
                    continue

                m2 = predictions_re.search(line)
                if m2:
                    payload = json.dumps({'type': 'predictions', 'count': int(m2.group('count'))})
                    yield f"data: {payload}\n\n"
                    continue

                if 'PIPELINE SUMMARY' in line or 'OPERATIONAL RECOMMENDATIONS PIPELINE' in line or 'Processing Recommendations' in line:
                    payload = json.dumps({'type': 'stage', 'message': line})
                    yield f"data: {payload}\n\n"
                    continue

                m3 = train_acc_re.search(line)
                if m3:
                    payload = json.dumps({'type': 'train_summary', 'message': m3.group(1).strip()})
                    yield f"data: {payload}\n\n"
                    continue

                m5 = summary_key_re.search(line)
                if m5:
                    key = line.split(':')[0].strip()
                    num = int(m5.group('num'))
                    payload = json.dumps({'type': 'summary', 'key': key, 'value': num})
                    yield f"data: {payload}\n\n"
                    continue

                if line.startswith('✓') or line.startswith('✗') or line.startswith('Start Time') or line.startswith('Confidence Threshold') or line.startswith('Train') or line.startswith('No high-confidence'):
                    payload = json.dumps({'type': 'info', 'message': line})
                    yield f"data: {payload}\n\n"
                    continue

        rc = process.wait() if process is not None else 0
        yield f"data: {json.dumps({'type': 'finished', 'returncode': rc})}\n\n"

    except GeneratorExit:
        try:
            if process:
                process.terminate()
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            if process:
                process.kill()
        except Exception:
            pass
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
