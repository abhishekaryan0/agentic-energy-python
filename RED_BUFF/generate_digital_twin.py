import json
import numpy as np
import random
from datetime import datetime, timedelta

# ─── CONFIG ──────────────────────────────────────────────────────────────────
END_DATE        = datetime(2026, 5, 7)
COMPANY         = "Red Bluff Resources"
LEASES          = ["Chaparral", "Mesquite", "Caliche", "Llano"]
WELLS_PER_LEASE = 15

LIFT_POOL = (
    ["Pumping Unit"] * 27 +
    ["ESP"]          * 21 +
    ["Gas Lift"]     * 12
)
random.shuffle(LIFT_POOL)

def random_spud():
    bucket = random.random()
    if bucket < 0.30:
        return datetime(2012, 1, 1) + timedelta(days=random.randint(0, 730))
    elif bucket < 0.65:
        return datetime(2015, 1, 1) + timedelta(days=random.randint(0, 1095))
    elif bucket < 0.85:
        return datetime(2018, 1, 1) + timedelta(days=random.randint(0, 1095))
    else:
        return datetime(2021, 1, 1) + timedelta(days=random.randint(0, 730))

# ─── FAILURE LIBRARY ─────────────────────────────────────────────────────────
FAILURES = {
    # Window lengths reflect realistic field observation timelines:
    # - Acute mechanical events (parted rods, cavitation): short 14-28 day windows
    # - Slow-burn degradation (worn pump, valve failure): long 60-120 day windows
    # - The random walk engine means early signal is buried in noise regardless of window length
    "Pumping Unit": [
        {"cause": "Parted Rod String", "window_min": 14,  "window_max": 28,  "trigger_sensor": "motor_current_amps"},
        {"cause": "Worn Pump / Tag",   "window_min": 60,  "window_max": 120, "trigger_sensor": "pump_fillage_pct"},
    ],
    "ESP": [
        {"cause": "Motor Overtemp Shutdown", "window_min": 45, "window_max": 75, "trigger_sensor": "motor_temp_f"},
        {"cause": "Motor/Cable Failure",     "window_min": 20, "window_max": 40, "trigger_sensor": "motor_current_amps"},
        {"cause": "Pump Cavitation",         "window_min": 10, "window_max": 21, "trigger_sensor": "pump_intake_pressure_psi"},
    ],
    "Gas Lift": [
        {"cause": "Gas Lift Valve Failure", "window_min": 60, "window_max": 90, "trigger_sensor": "casing_injection_pressure_psi"},
        {"cause": "Mandrel Washout",        "window_min": 45, "window_max": 75, "trigger_sensor": "injection_rate_mscfd"},
    ],
}

# ─── SENSOR NORMAL RANGES ────────────────────────────────────────────────────
SENSOR_NORMALS = {
    # ── PUMPING UNIT ─────────────────────────────────────────────────────────
    # Ranges sourced from data_types_and_ranges.xlsx (Well Level sheet)
    "Pumping Unit": {
        "allocated_oil":        (1.0,    2500.0),   # xlsx: 1-2,500 bbl typical
        "test_oil":             (0.0,    0.0),       # handled separately
        "gross_stroke_len":     (20.0,   600.0),     # xlsx: Downhole Gross stroke length 20-600 inches
        "net_stroke_len":       (20.0,   600.0),     # xlsx: Downhole Net stroke length 20-600 inches
        "strokes_per_min":      (0.5,    12.0),      # xlsx: 0.5-12.0 SPM
        "pump_fillage_pct":     (0.0,    100.0),     # xlsx: 0-100%
        "casing_pressure_psi":  (50.0,   5000.0),    # xlsx: 0-5,000 usually 50-5,000 psi
        "tubing_pressure_psi":  (50.0,   150.0),     # xlsx: 0-5,000 usually around 50-150 psi
        "motor_current_amps":   (10.0,   200.0),     # xlsx: ESP ref 0-200 amps
        "freq_hz":              (57.0,   62.0),      # xlsx: Drive Frequency Hz — normal 60Hz
        "pump_intake_pressure": (0.0,    5000.0),    # xlsx: Pump Intake Pressure 0-5,000 psi
        "motor_temp_f":         (100.0,  160.0),     # healthy PU surface motor: 100-160F (xlsx range is full sensor capability)
    },
    # ── ESP ──────────────────────────────────────────────────────────────────
    # Ranges sourced from data_types_and_ranges.xlsx (Well Level sheet, ESP with VSD section)
    "ESP": {
        "allocated_oil":              (1.0,    2500.0),   # xlsx: 1-2,500 bbl typical
        "test_oil":                   (0.0,    0.0),      # handled separately
        "avg_tubing_pressure":        (200.0,  500.0),    # xlsx: Tubing Pressure 0-2,000 typically 200-500 psi
        "avg_casing_pressure":        (20.0,   200.0),    # xlsx: Casing Pressure 0-2,000 typically 20-200 psi
        "motor_temp_f":               (150.0,  250.0),    # healthy ESP motor: 150-250F (xlsx range is full sensor capability)
        "motor_current_amps":         (10.0,   200.0),    # xlsx: Motor Current 0-200 amps
        "freq_hz":                    (0.0,    100.0),    # xlsx: Drive Frequency 0-100 Hz
        "pump_intake_pressure_psi":   (0.0,    5000.0),   # xlsx: Pump Intake Pressure (PIP) 0-5,000 psi
        "pump_discharge_pressure_psi":(0.0,    5000.0),   # xlsx: Discharge Pressure 0-5,000 psi
        "vibration_hz":               (0.0,    1.25),     # xlsx: Vibration 0.00-1.25 in/s
    },
    # ── GAS LIFT ─────────────────────────────────────────────────────────────
    # Gas Lift not explicitly in xlsx — ranges based on industry standards
    # consistent with wellhead and facility data ranges in xlsx
    "Gas Lift": {
        "allocated_oil":                 (1.0,    2500.0),   # consistent with xlsx production range
        "test_oil":                      (0.0,    0.0),      # handled separately
        "avg_casing_pressure":           (50.0,   5000.0),   # xlsx: casing pressure 50-5,000 psi
        "casing_injection_pressure_psi": (800.0,  1200.0),   # industry standard GL injection pressure
        "tubing_pressure_psi":           (50.0,   150.0),    # xlsx: tubing pressure 50-150 psi typical
        "injection_rate_mscfd":          (0.0,    50000.0),  # xlsx: Gas flow 0-20,000 mscf/d; GL injection up to 50,000
        "wellhead_temp_f":               (10.0,   400.0),    # xlsx: facility temperature 10-400 F
        "choke_position_pct":            (0.0,    100.0),    # 0-100%
        "motor_temp_f":                  (80.0,   130.0),    # compressor motor — no xlsx ref, industry standard
    },
}

SENSOR_FAILURE_DIRECTION = {
    "motor_current_amps":              "up",
    "pump_fillage_pct":                "down",
    "strokes_per_min":                 "down",
    "gross_stroke_len":                "down",   # stroke shortens as pump wears
    "net_stroke_len":                  "down",   # net stroke collapses faster
    "motor_temp_f":                    "up",
    "avg_tubing_pressure":             "down",   # tubing pressure drops as ESP fails
    "avg_casing_pressure":             "up",     # casing pressure rises as gas breaks out
    "freq_hz":                         "down",   # VFD/motor frequency drops as motor struggles
    "pump_intake_pressure":            "down",   # intake pressure drops as pump wears
    "motor_temp_f":                    "up",     # motor temp rises on both PU and ESP
    "pump_intake_pressure_psi":        "down",
    "pump_discharge_pressure_psi":     "down",
    "vibration_hz":                    "up",
    "casing_injection_pressure_psi":   "down",
    "injection_rate_mscfd":            "down",
    "wellhead_temp_f":                 "down",
    "choke_position_pct":              "down",
    "avg_casing_pressure":             "up",     # casing pressure rises as valve degrades
    "motor_temp_f":                    "up",     # compressor motor temp rises as system struggles
    "casing_pressure_psi":             "up",
    "tubing_pressure_psi":             "down",
}

SENSOR_FAILURE_TARGET = {
    # Updated to align with xlsx-sourced ranges
    "motor_current_amps":              180.0,   # spikes toward 200A before failure
    "pump_fillage_pct":                15.0,    # collapses well below 40%
    "strokes_per_min":                 1.0,     # near stall before failure
    "gross_stroke_len":                10.0,    # stroke shortens severely
    "net_stroke_len":                  5.0,     # net collapses toward zero
    "motor_temp_f":                    320.0,   # ESP overtemp shutdown threshold ~300-350F
    "avg_tubing_pressure":             50.0,    # tubing pressure collapses
    "avg_casing_pressure":             4000.0,  # casing pressure spikes — xlsx max 5,000
    "freq_hz":                         45.0,    # frequency drops from 60Hz
    "pump_intake_pressure":            50.0,    # PU intake collapses
    "pump_intake_pressure_psi":        200.0,   # ESP intake collapses
    "pump_discharge_pressure_psi":     500.0,   # discharge collapses
    "vibration_hz":                    1.1,     # xlsx max 1.25 in/s — near limit
    "casing_injection_pressure_psi":   300.0,   # GL injection pressure collapses
    "injection_rate_mscfd":            500.0,   # injection rate drops sharply
    "wellhead_temp_f":                 15.0,    # wellhead temp drops as injection fails
    "choke_position_pct":              5.0,     # choke nearly fully closed
    "casing_pressure_psi":             4500.0,  # casing pressure spikes
    "tubing_pressure_psi":             20.0,    # tubing pressure collapses
}

def normal_reading(sensor, normals):
    """Healthy sensor — random walk within normal operating range."""
    lo, hi = normals[sensor]
    mid    = (lo + hi) / 2
    noise  = (hi - lo) * 0.04
    val    = float(np.random.normal(mid, noise))
    return round(float(np.clip(val, lo, hi)), 2)

def init_sensor_state(sensor, normals):
    """Starting value for a sensor at onset of deterioration — healthy range."""
    lo, hi = normals[sensor]
    mid    = (lo + hi) / 2
    noise  = (hi - lo) * 0.04
    return float(np.clip(np.random.normal(mid, noise), lo, hi))

def step_deteriorating_sensor(current_val, sensor, normals, day, total_window,
                               is_trigger, onset_day):
    """
    Gradual random-walk deterioration mimicking real well sensor behaviour.

    Design targets:
    - Early window (0-50%):  drift barely distinguishable from normal noise
    - Middle window (50-80%): trend becomes slightly more persistent
    - Late window (80-95%):  clearly drifting but still noisy
    - Final days (95-100%): unmistakable — sensor well outside normal range

    Trigger sensor drifts ~2x secondary sensors.
    Partial recoveries occur throughout — more common early, rare at the end.
    """
    lo, hi        = normals[sensor]
    rng           = hi - lo
    direction     = SENSOR_FAILURE_DIRECTION.get(sensor, "up")

    days_active   = max(0, day - onset_day)
    if days_active <= 0:
        # Not yet in deterioration — pure healthy noise
        return round(float(np.clip(np.random.normal(current_val, rng * 0.025), lo, hi)), 2)

    active_window = max(total_window - onset_day, 1)
    progress      = min(days_active / active_window, 1.0)

    # ── Drift scale: very small early, meaningful only in final stretch ──────
    # Trigger: peaks at ~2% of range per day at progress=1.0
    # Secondary: peaks at ~0.8% of range per day at progress=1.0
    # Acceleration only kicks in meaningfully after 85% progress
    base_drift    = (0.008 if is_trigger else 0.003) * rng
    if progress > 0.85:
        accel     = 1.0 + (progress - 0.85) * 12.0   # gentle until very end
    else:
        accel     = 1.0 + progress * 0.5              # almost flat early on
    drift_scale   = base_drift * accel

    bias = drift_scale * progress if direction == "up" else -drift_scale * progress

    # ── Partial recoveries — common early, rare late ─────────────────────────
    recovery_prob = max(0.0, 0.45 - progress * 0.45)  # 45% early → 0% at failure
    if random.random() < recovery_prob:
        bias *= -random.uniform(0.4, 1.0)  # full or partial reversal

    # ── Noise stays relatively high throughout so signal stays buried early ──
    # Noise is larger than drift early — only at the end does drift dominate
    noise   = rng * (0.025 + progress * (0.02 if is_trigger else 0.01))
    new_val = current_val + bias + float(np.random.normal(0, noise))

    # Physical floor — no negative sensor values
    new_val = max(new_val, 0.0)
    new_val = min(new_val, hi * 2.0)

    return round(new_val, 2)

def step_gross_net_stroke(gross_cur, net_cur, gross_base, net_base,
                           day, total_window, onset_day):
    """
    Gradual gross/net stroke deterioration.
    Net always <= gross. Both drift slowly — net faster than gross.
    """
    days_active = max(0, day - onset_day)

    if days_active <= 0:
        gross = max(1.0, float(np.random.normal(gross_base, gross_base * 0.01)))
        net   = max(1.0, float(min(np.random.normal(net_base, net_base * 0.012),
                                    gross * 0.98)))
        return round(gross, 2), round(net, 2)

    active_window = max(total_window - onset_day, 1)
    progress      = min(days_active / active_window, 1.0)

    # Very small daily drift — strokes shorten slowly over weeks
    g_drift = gross_base * 0.002 * (1.0 + progress * 2.5)
    n_drift = net_base   * 0.004 * (1.0 + progress * 3.0)  # net drops faster

    g_bias  = -g_drift * progress
    n_bias  = -n_drift * progress

    # Recoveries
    if random.random() < max(0.0, 0.40 - progress * 0.40):
        g_bias *= -random.uniform(0.3, 0.8)
        n_bias *= -random.uniform(0.3, 0.8)

    gross = max(1.0, gross_cur + g_bias + float(np.random.normal(0, gross_base * 0.008)))
    net   = max(1.0, net_cur   + n_bias + float(np.random.normal(0, net_base   * 0.010)))
    net   = min(net, gross * 0.98)

    return round(gross, 2), round(net, 2)

def simulate_well(asset_id, lift, spud_date, base_rate, gross_base, net_base):
    dates   = []
    current = spud_date
    while current <= END_DATE:
        dates.append(current)
        current += timedelta(days=1)

    days         = len(dates)
    t            = np.arange(days)

    # ── Hyperbolic decline curve (Arps) ──────────────────────────────────────
    # q(t) = q_i / (1 + b * Di * t)^(1/b)
    # b = hyperbolic exponent — varies by lift type and well character
    # Di = initial decline rate per day
    # Minimum floor prevents reaching zero — realistic economic limit
    # Pumping Unit (vertical conventional): b=0.5-0.8, gentler decline, long tail
    # ESP (horizontal high-rate):           b=0.8-1.2, steeper early, long flat tail
    # Gas Lift (horizontal):                b=0.7-1.0, moderate decline
    variance_map = {"Gas Lift": 0.050, "Pumping Unit": 0.035, "ESP": 0.020}
    variance     = variance_map[lift]

    b_map  = {
        "Pumping Unit": random.uniform(0.5,  0.8),   # conventional vertical
        "ESP":          random.uniform(0.8,  1.2),   # high-rate horizontal
        "Gas Lift":     random.uniform(0.7,  1.0),   # horizontal moderate
    }
    Di_map = {
        "Pumping Unit": random.uniform(0.0004, 0.0008),  # slower initial decline
        "ESP":          random.uniform(0.0008, 0.0015),  # steeper early decline
        "Gas Lift":     random.uniform(0.0006, 0.0012),
    }
    # Economic minimum floor — well stays on production at low stable rate
    floor_map = {
        "Pumping Unit": random.uniform(8.0,  20.0),  # bbl/day — low-cost vertical
        "ESP":          random.uniform(30.0, 80.0),  # bbl/day — higher opex threshold
        "Gas Lift":     random.uniform(15.0, 40.0),  # bbl/day
    }

    b     = b_map[lift]
    Di    = Di_map[lift]
    floor = floor_map[lift]

    # Hyperbolic decline — flattens to long-tail rather than reaching zero
    theoretical = base_rate / np.power(1.0 + b * Di * t, 1.0 / b)
    # Apply economic floor — no PRODUCING day goes below this
    theoretical = np.maximum(theoretical, floor)

    prod_noise   = np.random.normal(1.0, variance, days)
    rates        = np.clip(theoretical * prod_noise, floor * 0.5, None)

    # ── Water and gas production ──────────────────────────────────────────────
    PROD_PARAMS_INLINE = {
        "Pumping Unit": {"water_base":(50.0,800.0),"gas_base":(10.0,150.0),
                         "water_noise":0.06,"gas_noise":0.05,
                         "water_decline":0.0002,"gas_decline":0.0008,"water_rise":0.0001},
        "ESP":          {"water_base":(200.0,5000.0),"gas_base":(50.0,500.0),
                         "water_noise":0.05,"gas_noise":0.04,
                         "water_decline":0.0001,"gas_decline":0.0007,"water_rise":0.00015},
        "Gas Lift":     {"water_base":(100.0,2000.0),"gas_base":(100.0,2000.0),
                         "water_noise":0.07,"gas_noise":0.06,
                         "water_decline":0.00015,"gas_decline":0.0006,"water_rise":0.00012},
    }
    pp           = PROD_PARAMS_INLINE[lift]
    water_base_v = random.uniform(*pp["water_base"])
    gas_base_v   = random.uniform(*pp["gas_base"])
    water_th     = water_base_v*(1+pp["water_rise"]*t)*np.exp(-pp["water_decline"]*t)
    gas_th       = gas_base_v*np.exp(-pp["gas_decline"]*t)
    water_rates  = np.clip(water_th*np.random.normal(1.0,pp["water_noise"],days), 1.0, None)
    gas_rates    = np.clip(gas_th  *np.random.normal(1.0,pp["gas_noise"],  days), 0.5, None)

    test_day_of_month = random.randint(5, 25)

    normals     = {k: v for k, v in SENSOR_NORMALS[lift].items()
                   if k not in ('allocated_oil', 'test_oil', 'gross_stroke_len', 'net_stroke_len')}
    sensor_keys = list(normals.keys())

    history       = []
    status        = "PRODUCING"
    days_online   = 0
    downtime_rem  = 0
    failure_cause = ""

    # Deterioration state
    detr_active      = False
    detr_total_days  = 0
    detr_elapsed     = 0
    detr_failure_def = None
    # Per-sensor running state values (for random walk)
    sensor_state     = {}
    # Per-sensor random onset day within the window (staggered — not all start day 1)
    sensor_onset     = {}
    # Gross/net stroke running state
    gross_cur        = gross_base
    net_cur          = net_base

    for i, current_date in enumerate(dates):
        date_str    = current_date.strftime("%Y-%m-%d")
        is_test_day = (current_date.day == test_day_of_month)
        row         = {"date": date_str}

        if status == "PRODUCING":
            days_online += 1

            # ── Hazard function ──────────────────────────────────────────────
            fail_prob = 0.00005
            if lift == "ESP" and days_online > 450:
                fail_prob += 0.003 * ((days_online - 450) / 100)
            elif lift == "Pumping Unit" and days_online > 300:
                fail_prob += 0.002 * ((days_online - 300) / 100)
            elif lift == "Gas Lift" and days_online > 600:
                fail_prob += 0.001 * ((days_online - 600) / 150)

            # ── Trigger new deterioration event ─────────────────────────────
            if not detr_active and random.random() < fail_prob:
                detr_active      = True
                detr_failure_def = random.choice(FAILURES[lift])
                detr_total_days  = random.randint(detr_failure_def["window_min"],
                                                  detr_failure_def["window_max"])
                detr_elapsed     = 0
                trigger_sens     = detr_failure_def["trigger_sensor"]

                # Initialize each sensor at its current healthy value
                # Stagger onset: trigger sensor starts immediately,
                # secondary sensors start at random day within first 40% of window
                sensor_state = {s: init_sensor_state(s, normals) for s in sensor_keys}
                sensor_onset = {}
                for s in sensor_keys:
                    if s == trigger_sens:
                        sensor_onset[s] = 0   # trigger starts immediately
                    else:
                        # Secondary sensors onset randomly — some early, some late
                        sensor_onset[s] = random.randint(
                            int(detr_total_days * 0.10),
                            int(detr_total_days * 0.50)
                        )

                # Stroke lengths
                gross_cur = gross_base
                net_cur   = net_base

            # ── Advance sensor states ────────────────────────────────────────
            if detr_active:
                detr_elapsed += 1
                trigger_sens  = detr_failure_def["trigger_sensor"]
                new_sensor_vals = {}

                for s in sensor_keys:
                    new_val = step_deteriorating_sensor(
                        sensor_state[s], s, normals,
                        detr_elapsed, detr_total_days,
                        is_trigger=(s == trigger_sens),
                        onset_day=sensor_onset[s]
                    )
                    sensor_state[s]    = new_val
                    new_sensor_vals[s] = new_val

                if lift == "Pumping Unit":
                    gross_cur, net_cur = step_gross_net_stroke(
                        gross_cur, net_cur, gross_base, net_base,
                        detr_elapsed, detr_total_days,
                        onset_day=sensor_onset.get("gross_stroke_len", 0)
                    )

                # ── Trigger failure at end of window ────────────────────────
                if detr_elapsed >= detr_total_days:
                    status        = "SHUT-IN"
                    failure_cause = detr_failure_def["cause"]
                    downtime_rem  = random.randint(4, 12)
                    detr_active   = False
                    detr_failure_def = None
                    sensor_state  = {}
                    sensor_onset  = {}
                    row["allocated_oil"]   = 0.0
                    row["test_oil"]        = None
                    if lift == "Pumping Unit":
                        row["gross_stroke_len"] = 0.0
                        row["net_stroke_len"]   = 0.0
                    for s in sensor_keys:
                        row[s] = 0.0
                    row["run_status"]    = "SHUT-IN"
                    row["failure_cause"] = failure_cause
                    row["allocated_water"] = 0.0
                    row["allocated_gas"]   = 0.0
                    history.append(row)
                    continue

                sensor_vals = new_sensor_vals

            else:
                # Normal healthy readings
                sensor_vals = {s: normal_reading(s, normals) for s in sensor_keys}
                if lift == "Pumping Unit":
                    gross_cur = max(1.0, float(np.random.normal(gross_base, gross_base * 0.01)))
                    net_cur   = max(1.0, float(min(
                        np.random.normal(net_base, net_base * 0.015),
                        gross_cur * 0.98
                    )))

            oil = round(float(rates[i]), 2)
            row["allocated_oil"] = oil

            if is_test_day:
                test_var        = random.uniform(0.05, 0.12)
                test_dir        = random.choice([1, -1])
                row["test_oil"] = round(oil * (1 + test_dir * test_var), 2)
            else:
                row["test_oil"] = None

            if lift == "Pumping Unit":
                row["gross_stroke_len"] = round(gross_cur, 2)
                row["net_stroke_len"]   = round(net_cur, 2)

            for s, v in sensor_vals.items():
                row[s] = v

            row["run_status"]    = "PRODUCING"
            row["failure_cause"] = ""
            row["allocated_water"] = round(float(water_rates[i]), 2)
            row["allocated_gas"]   = round(float(gas_rates[i]),   2)

        else:  # SHUT-IN
            row["allocated_oil"]   = 0.0
            row["test_oil"]        = None
            if lift == "Pumping Unit":
                row["gross_stroke_len"] = 0.0
                row["net_stroke_len"]   = 0.0
            for s in sensor_keys:
                row[s] = 0.0
            downtime_rem -= 1
            row["run_status"]    = "SHUT-IN"
            row["failure_cause"] = failure_cause
            row["allocated_water"] = 0.0
            row["allocated_gas"]   = 0.0
            if downtime_rem <= 0:
                status        = "PRODUCING"
                days_online   = 0
                failure_cause = ""
                gross_cur     = gross_base
                net_cur       = net_base

        history.append(row)
    return history

def build_wells():
    wells    = []
    lift_idx = 0
    for lease in LEASES:
        for w in range(1, WELLS_PER_LEASE + 1):
            well_num  = LEASES.index(lease) * WELLS_PER_LEASE + w
            lift      = LIFT_POOL[lift_idx % len(LIFT_POOL)]
            lift_idx += 1
            suffix    = "V" if lift == "Pumping Unit" else "H"
            asset_id  = f"{lease}-12-{well_num}-{suffix}"
            # Gross stroke base: 120-168 inches, net is 85-92% of gross
            gross_base = random.uniform(120.0, 168.0)
            net_base   = gross_base * random.uniform(0.85, 0.92)
            wells.append({
                "asset_id":   asset_id,
                "lease":      lease,
                "lift":       lift,
                "spud_date":  random_spud(),
                "base_rate":  random.uniform(300, 1200),
                "gross_base": gross_base,
                "net_base":   net_base,
            })
    return wells

def get_personnel(lease):
    return {
        "prod_technician": {
            "name":  f"{lease} Production Tech",
            "role":  "Production Technician",
            "email": f"prodtech.{lease.lower()}@redbluffresources.com"
        },
        "well_perf_engineer": {
            "name":  random.choice(["Derek Tatum", "Kristen Albright"]),
            "role":  "Well Performance Engineer",
            "email": "wellperf@redbluffresources.com"
        }
    }

if __name__ == "__main__":
    print("Red Bluff Resources — Digital Twin")
    print(f"60 wells | End date: {END_DATE.strftime('%Y-%m-%d')}")
    print("─" * 60)

    wells  = build_wells()
    output = {"company": COMPANY, "assets": {}}

    for i, w in enumerate(wells):
        aid = w["asset_id"]
        print(f"  [{i+1:>2}/60] {aid} ({w['lift']}, spud {w['spud_date'].strftime('%Y-%m-%d')})...")
        history = simulate_well(
            aid, w["lift"], w["spud_date"],
            w["base_rate"], w["gross_base"], w["net_base"]
        )
        output["assets"][aid] = {
            "type":               "Well",
            "lift":               w["lift"],
            "assigned_personnel": get_personnel(w["lease"]),
            "history":            history,
            "financial_ledger":   []
        }

    with open("red_bluff_engineering.json", "w") as f:
        json.dump(output, f, indent=2)

    total_days     = sum(len(a["history"]) for a in output["assets"].values())
    total_failures = sum(
        sum(1 for d in a["history"] if d["run_status"] == "SHUT-IN" and d["failure_cause"])
        for a in output["assets"].values()
    )
    print(f"\n✓ Done")
    print(f"  Assets:           {len(output['assets'])}")
    print(f"  Total daily rows: {total_days:,}")
    print(f"  Failure events:   {total_failures:,}")
