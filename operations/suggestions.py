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

        from operations.recommendations.persistence import (
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

        time_reduced = None
        try:
            val = float(op_details.get("time_reduced_hours") or 0)
            time_reduced = val if val > 0 else None
        except Exception:
            time_reduced = None

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
        row["time_reduced_hours"] = time_reduced

        from operations.recommendations.persistence import write_operations_recommendations_to_s3
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
