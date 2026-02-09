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
    impact_metrics: Dict = None,
    anomaly_code: str = None
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
                diagnosis_context = f"POTENTIAL ISSUE (LOW CONFIDENCE): {diagnosis['type']} (Indicators: {diagnosis['indicators']})"
                confidence_instruction = """
                NOTE: The diagnosis is PRELIMINARY. 
                - In 'suspected_root_cause', provide a detailed technical hypothesis: "Preliminary analysis suggests [Mechanism], evidenced by [Data Points], though validation is required."
                - DO NOT use the word "Unconfirmed" in the title. Treat the anomaly as a valid lead.
                """
            else:
                diagnosis_context = f"DETECTED PATTERN: {diagnosis['type']} (Indicators: {diagnosis['indicators']})"
                confidence_instruction = """
                - Use professional, objective language ("Data indicates...", "Pattern suggests...").
                - Avoid alarmist terms like "Catastrophic" unless explicitly in the data.
                - Focus on the DEVIATION from normal dynamics.
                """
        
        # Construct Impact Context
        impact_context = ""
        if impact_metrics:
            fin = impact_metrics.get('financial', {})
            prod = impact_metrics.get('production', {})
            impact_context = f"""
            CALCULATED IMPACT METRICS (Use EXACTLY these values, DO NOT re-calculate or guess):
            - Financial Loss: {fin.get('label', 'N/A')} (Value: {fin.get('value', 0)})
            - Production Loss: {prod.get('label', 'N/A')} (Value: {prod.get('value', 0)})
            """

        # VALIDATION MODE PROMPT (For ML Anomalies)
        # We trigger validation if the code is explicitly ML_PATTERN_ANOMALY or if diagnosis implies uncertainty
        is_validation = (anomaly_code == 'ML_PATTERN_ANOMALY') or ('Unclassified' in str(diagnosis))
        
        validation_instruction = ""
        if is_validation:
            validation_instruction = """
            CRITICAL TASK: VALIDATE IF THIS IS A REAL ANOMALY.
            The "Autoencoder" detected a statistical deviation. As a Senior Engineer, you must decide:
            - Is this meaningful physics (e.g., pressure spike, fillage drop)? 
            - OR is it just sensor noise / tiny fluctuation?
            
            Return a JSON field "is_valid_anomaly": true/false.
            If "is_valid_anomaly": false, you may leave other fields empty. 
            """

        prompt = f"""
ROLE: Senior Production Engineer / Optimization Specialist with deep expertise in rod pumping and anomaly diagnostics.
TASK: Generate a COMPREHENSIVE technical investigation report for a well performance deviation. The output must be rich in engineering detail, explaining the *mechanics* of the issue, not just stating the metrics.
TONE: Investigative, highly analytical, professional.

{validation_instruction}

{confidence_instruction}

ANOMALY DATA (raw metrics and indicators - use these exactly in your report):
{anomaly_data}

{impact_context}

DIAGNOSIS (System Logic Output - use as a starting hypothesis):
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
- **CRITICAL: USE EXACT VALUES FROM ANOMALY DATA**: You MUST use the EXACT numerical values provided in the ANOMALY DATA section.
- PRIORITIZE DEVIATION: Focus on how much the value has shifted from baseline/model vs just the current value.
- ACTIONS > ALARMS: Frame the output as "What should the engineer look at?" rather than "What is broken?".

Generate a JSON response with these EXACT keys and STRICT formats:

1. "is_valid_anomaly": (boolean) - See critical verification task above.
2. "title": 
   - Max 6 words. No $ symbols.
   - Example: "Energy Efficiency Drift Detected" instead of "Pump Failure".
   - Example: "Pressure Volatility Increase" instead of "Flowline Blockage".

3. "description": 
   - A detailed, 3-5 sentence paragraph.
   - Combine exact data points with mechanical application. 
   - Explain HOW the deviation is occurring (e.g., "The 18% increase in energy intensity suggests the pump is overcoming increased friction or restrictions...").
   - Explicitly mention the financial/production impact using 'CALCULATED_REVENUE_LOSS'.

4. "why_is_this_an_anomaly": 
   - ONE clear sentence stating the deviation math.
   - "The current [Metric] of [Value] represents a [X]% deviation from the expected model/baseline."

5. "suspected_root_cause": 
   - **IMPORTANT**: Provide a DETAILED technical assessment (2-3 sentences, 40-70 words).
   - Explain the likely physical mechanism. 
   - Example: "Data suggests gas interference is reducing pump fillage, as evidenced by the 25% volatility in tubing pressure. This creates a gas lock condition that prevents the standing valve from opening, leading to the observed drop in fluid load."
   - Connect the sensors: "High Amps (Val) paired with Low Flow (Val) indicates..."

6. "economic_impact": 
   ONE sentence: "This deviation represents an opportunity cost of estimated $[X]/day due to [inefficiency/production gap]."
   - CRITICAL: You MUST extract $[X] from 'CALCULATED_REVENUE_LOSS'.

7. "risk_analysis": 
   - You MUST follow this EXACT two-paragraph structure, no objects just string:
   
    PARAGRAPH 1 (Operational Context):
    "The detected deviation in [Sensor/Metric] suggests a shift in [well dynamics/efficiency]. [Insert 1 sentence on the magnitude of the shift]. While immediate failure may not be imminent, this pattern indicates [wear/inefficiency/instability] that requires attention to prevent degradation. [Add specific mechanical Detail based on Lift Type]."
   
    PARAGRAPH 2 (Long-term Optimization):
    "Detected at: {formatted_date}
    Key Factors: [Insert content from suspected_root_cause above]
   
    [Insert 2-3 sentences on the value of early intervention. Focus on extending run-life, reducing power consumption, or recovering deferred production. Reference the impact value as potential savings/recovery]. Proactive tuning now can prevent [specific larger failure] later."

9. "recommended_actions":
   - PROVIDE 3 SPECIFIC, ACTIONABLE STEPS (Bulleted string or short paragraph).
   - Examples: "1. Reduce SPM by 1.5 to match inflow. 2. Verify fluid level with echometer. 3. Check chemical injection rate."
   - Focus on FIXING the root cause identified above.

8. "historical_context": 
   - Return a JSON object with:
     {{
       "recurrence_status": "[New Pattern|Recurring Deviation|Chronic Efficiency Issue]" (analyze from HISTORICAL CONTEXT),
       "history_timeline": [
         {{"date": "MM/DD/YYYY", "severity": "High|Moderate|Low", "title": "Previous event title"}},
         ...
       ] (limit to 5, format dates properly)
     }}

Return ONLY valid JSON. Ensure all fields are populated based on data provided.
"""
        
        # Call OpenAI with explicit timeout
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=15  # 15-second timeout
            )
            content = json.loads(response.choices[0].message.content)
            
        except Exception as e:
            # Handle Timeout or API Error by returning structured fallback
            logger.warning(f"LLM Call failed or timed out: {e}")
            return {
                "title": "Analysis Pending",
                "description": "Automated technical analysis is currently unavailable due to high system load. Please review the raw sensor data below.",
                "why_is_this_an_anomaly": "Metrics deviate from established baselines.",
                "suspected_root_cause": "Pending automated investigation.",
                "economic_impact": "Impact calculation pending.",
                "risk_analysis": "Please conduct manual review of the attached charts.",
                "historical_context": {"recurrence_status": "Unknown", "history_timeline": []},
                "is_valid_anomaly": True
            }

        # Default is_valid to True if not present (legacy support)
        if "is_valid_anomaly" not in content:
            content["is_valid_anomaly"] = True
            
        return content
        
    except Exception as e:
        logger.error(f"LLM content generation critical failure: {e}")
        return {
            "title": "Analysis Pending",
            "description": "Automated technical analysis is currently unavailable due to high system load. Please review the raw sensor data below.",
            "why_is_this_an_anomaly": "Metrics deviate from established baselines.",
            "suspected_root_cause": "Pending automated investigation.",
            "economic_impact": "Impact calculation pending.",
            "risk_analysis": "Please conduct manual review of the attached charts.",
            "historical_context": {"recurrence_status": "Unknown", "history_timeline": []},
            "is_valid_anomaly": True
        }
