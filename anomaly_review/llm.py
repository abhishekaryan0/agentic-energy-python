"""
LLM content generation for anomaly reports.

Generates detailed, professional reports using OpenAI API for detected anomalies.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional

from anomaly_review.data import get_openai_client

logger = logging.getLogger('AnomalyReviewLLM')


def generate_llm_content(
    anomaly_data: str,
    lift_type: str,
    chart_metric: str,
    historical_context: str,
    detected_at_str: str,
    diagnosis: Dict = None,
    impact_metrics: Dict = None
) -> Dict:
    """
    Generate detailed anomaly report using LLM.
    
    Args:
        anomaly_data: Raw metrics and indicators
        lift_type: Well lift type (Rod Pump, ESP, Gas Lift)
        chart_metric: Type of chart being generated
        historical_context: Previous occurrences
        detected_at_str: Detection timestamp
        diagnosis: Root cause assessment
        impact_metrics: Financial and production impact
        
    Returns:
        Dict with title, description, why_is_this_an_anomaly,
        suspected_root_cause, economic_impact, risk_analysis, historical_context
    """
    # Clean NaN strings
    anomaly_data = str(anomaly_data).replace('nan', 'Unknown')
    if diagnosis:
        diagnosis = {k: str(v).replace('nan', 'Unknown') for k, v in diagnosis.items()}

    try:
        client = get_openai_client()
        if not client:
            raise Exception("No OpenAI Client")
        
        # Format detection timestamp
        try:
            if 'T' in str(detected_at_str):
                dt_obj = datetime.fromisoformat(str(detected_at_str).replace('Z', '+00:00'))
            else:
                dt_obj = datetime.strptime(str(detected_at_str), '%Y-%m-%d %H:%M:%S.%f')
            formatted_date = dt_obj.strftime('%m/%d/%Y %H:%M:%S')
        except:
            formatted_date = str(detected_at_str)

        # Handle confidence levels in prompt
        confidence_instruction = ""
        diagnosis_context = ""
        
        if diagnosis:
            if diagnosis.get('confidence') == 'Low':
                diagnosis_context = f"DIAGNOSIS (LOW CONFIDENCE): {diagnosis['type']} (Indicators: {diagnosis['indicators']})"
                confidence_instruction = """
                NOTE: The root cause confidence is LOW. 
                - In 'suspected_root_cause', state "Possible causes include [Diagnosis] depending on validation."
                - DO NOT use the word "Unconfirmed" in the title. Treat the anomaly as valid but needing review.
                - Emphasize the need for further field validation in the description.
                """
            else:
                diagnosis_context = f"CONFIRMED DIAGNOSIS: {diagnosis['type']} (Indicators: {diagnosis['indicators']})"
                confidence_instruction = """
                - Use ONLY declarative statements ("The root cause is...").
                - NO hedging ("might", "could").
                - Be confident and direct in all fields.
                """
        
        # Prepare impact context
        impact_context = ""
        if impact_metrics:
            fin = impact_metrics.get('financial', {})
            prod = impact_metrics.get('production', {})
            impact_context = f"""
            CALCULATED IMPACT METRICS (Use EXACTLY these values, DO NOT re-calculate or guess):
            - Financial Loss: {fin.get('label', 'N/A')} (Value: {fin.get('value', 0)})
            - Production Loss: {prod.get('label', 'N/A')} (Value: {prod.get('value', 0)})
            """

        prompt = f"""
ROLE: Senior Production Engineer with 20+ years in oil & gas operations, specializing in rod pump systems and anomaly detection.
TASK: Generate a detailed, professional report on a detected well anomaly. The report must be factual, concise, and actionable for field operators and engineers. Use technical terminology appropriately but explain implications clearly.

{confidence_instruction}

ANOMALY DATA (raw metrics and indicators - use these exactly in your report):
{anomaly_data}

{impact_context}

DIAGNOSIS (root cause assessment - integrate this into the report):
{diagnosis_context}

HISTORICAL CONTEXT (past similar events - reference in historical_context field):
{historical_context}

LIFT TYPE: {lift_type} (tailor explanations to this, e.g., rod pump specifics like SPM, amps, tubing pressure correlations)

CHART METRIC: {chart_metric} (mention relevant visualization in risk_analysis if applicable)

DETECTED AT: {formatted_date}

NUMERICAL CONFORMITY STANDARDS (STRICTLY FOLLOW):
1. COMMA DELIMITED: Always use commas for thousands (e.g., 1,500 or 300,000).
2. NO DECIMALS: Round all metrics and currency to the nearest whole number (e.g., 46 Amps, $3,500).
3. EXCEPTIONS (Use exactly 1 decimal place):
   - Strokes Per Minute (e.g., 4.2 SPM)
   - Values less than 1 (e.g., 0.5 bbl)

IMPORTANT GUIDELINES FOR ALL FIELDS:
- ALWAYS explicitly name the primary sensor(s) involved (e.g., "Tubing Pressure", "Motor Current", "Strokes Per Minute") in the description, why_is_this_an_anomaly, and suspected_root_cause.
- SENSOR NAMING: Never use snake_case (e.g. oil_volume) in the output text. Always convert to Title Case (e.g. Oil Volume).
- If volatility, deviation, drop, or other percentages are in ANOMALY DATA, PROMINENTLY INCLUDE THE EXACT NUMBERS.
- Prioritize key metrics: volatility % over small mean deviations for instability; efficiency drop % for degradation; decline % for production changes.
- Use industry-specific language: reference rod pump dynamics, gas interference, pump off, parted rods, etc., where relevant.
- Ensure the report is actionable.
- Financial impacts: You MUST use the 'CALCULATED_REVENUE_LOSS' provided in the data. Do NOT calculate it yourself.

Generate a JSON response with these EXACT keys and STRICT formats:

1. "title": 
   - If High Confidence Diagnosis: Use diagnosis type (e.g., "Tubing Leak Detected").
   - If Low Confidence: Use the observed symptom (e.g., "Efficiency Drop Detected").
   - Max 6 words. No $ symbols. Make it attention-grabbing and specific.

2. "description": 
   - Write EXACTLY 3-4 sentences.
   - FIRST SENTENCE: State current value vs baseline with specific numbers and sensor name.
   - SECOND SENTENCE: Explain the physical/mechanical implication, naming sensors.
   - THIRD SENTENCE: State the financial impact.
   - FOURTH SENTENCE (optional): Add operational context or recommendation.

3. "why_is_this_an_anomaly": 
   - ONE clear sentence stating the deviation math, including sensor and key percentages.

4. "suspected_root_cause": 
   - Return a SINGLE, DETAILED sentence (30-60 words).
   - Include sensor names and numbers.
   - DO NOT provide a list.

5. "economic_impact": 
   - ONE sentence in this format: "This [problem type] is resulting in an estimated daily revenue loss of $[X], based on current rates."

6. "risk_analysis": 
   - Two paragraphs: Immediate Risk and Long-term Consequence.

7. "historical_context": 
   - Return a JSON object with recurrence_status and history_timeline.

Return ONLY valid JSON. Ensure all fields are populated based on data provided.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        logger.error(f"LLM content generation failed: {e}")
        return {
            "title": "Anomaly Detected",
            "description": "An operational deviation was detected. Review sensor data manually. Impact unknown.",
            "why_is_this_an_anomaly": "Metrics deviate from baseline.",
            "suspected_root_cause": "Unknown - data analysis required.",
            "economic_impact": "Impact calculation unavailable.",
            "risk_analysis": "Review required for immediate and long-term risks.",
            "historical_context": {"recurrence_status": "Unknown", "history_timeline": []}
        }
