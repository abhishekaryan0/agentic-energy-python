import json
import numpy as np
import random
from datetime import datetime, timedelta

# ─── CONFIG ──────────────────────────────────────────────────────────────────
END_DATE = datetime(2026, 6, 8)
COMPANY = "South Texas Energy Partners"

try:
    from config import STEP_ORGANIZATION_ID as ORGANIZATION_ID
except ImportError:
    ORGANIZATION_ID = "d270f89a-aada-4ff2-adb0-68b3e8087565"

# ── AREA GROUPINGS ─────────────────────────────────────────────────────────────
# Company-internal operating area groupings — fixed per lease
# Modifiable by operator in the platform
AREA_MAP = {
    "Bravo":    "Brush Country Area",
    "Maverick": "Rio Grande Area",
    "Carrizo":  "Karnes Area",
    "Cotulla":  "Rio Grande Area",
    "Pearsall": "Brush Country Area",
}

# ── IP FLUSH WELLS ────────────────────────────────────────────────────────────
# 20 wells fixed permanently — 4 per lease.
# Cannot overlap with the 15 clean wells.
# Each gets a 60-90 day flush period at 2-4x normal starting rate.
# Water also elevated during flush — frac fluid returns from formation.
# IP flush wells — fixed permanently with exact well IDs including suffix
# Lift pool is now fixed so suffixes never change
IP_FLUSH_WELL_NUMBERS = [
    # Bravo lease — Brush Country Area
    "Bravo-1H-8-H",    "Bravo-1H-12-H",    "Bravo-1H-14-H",    "Bravo-1H-17-H",
    # Maverick lease — Rio Grande Area
    "Maverick-1H-28-H","Maverick-1H-30-H", "Maverick-1H-40-H", "Maverick-1H-37-H",
    # Carrizo lease — Karnes Area
    "Carrizo-1H-53-H", "Carrizo-1H-54-H",  "Carrizo-1H-55-V",  "Carrizo-1H-57-H",
    # Cotulla lease — Rio Grande Area
    "Cotulla-1H-67-H", "Cotulla-1H-73-H",  "Cotulla-1H-75-H",  "Cotulla-1H-80-H",
    # Pearsall lease — Brush Country Area
    "Pearsall-1H-88-H","Pearsall-1H-90-H", "Pearsall-1H-92-H", "Pearsall-1H-99-H",
]

# ─── LEASE AND LIFT CONFIGURATION ─────────────────────────────────────────────
LEASES          = ["Bravo", "Maverick", "Carrizo", "Cotulla", "Pearsall"]
WELLS_PER_LEASE = 20

# LIFT POOL — fixed permanently. Same lift type for every well every rebuild.
# 55 ESP / 35 Gas Lift / 10 Pumping Unit — 11/7/2 per lease
# Order matches well numbers 1-20 per lease sequentially.
LIFT_POOL = [
    # Bravo (wells 1-20)
    "ESP",          # Bravo-1H-1-H
    "ESP",          # Bravo-1H-2-H
    "Gas Lift",     # Bravo-1H-3-H
    "ESP",          # Bravo-1H-4-H
    "ESP",          # Bravo-1H-5-H
    "ESP",          # Bravo-1H-6-H
    "Gas Lift",     # Bravo-1H-7-H
    "Gas Lift",     # Bravo-1H-8-H
    "Gas Lift",     # Bravo-1H-9-H
    "ESP",          # Bravo-1H-10-H
    "Pumping Unit", # Bravo-1H-11-V
    "ESP",          # Bravo-1H-12-H
    "Gas Lift",     # Bravo-1H-13-H
    "Gas Lift",     # Bravo-1H-14-H
    "Gas Lift",     # Bravo-1H-15-H
    "ESP",          # Bravo-1H-16-H
    "ESP",          # Bravo-1H-17-H
    "Pumping Unit", # Bravo-1H-18-V
    "ESP",          # Bravo-1H-19-H
    "ESP",          # Bravo-1H-20-H
    # Maverick (wells 21-40)
    "ESP",          # Maverick-1H-21-H
    "Gas Lift",     # Maverick-1H-22-H
    "ESP",          # Maverick-1H-23-H
    "ESP",          # Maverick-1H-24-H
    "ESP",          # Maverick-1H-25-H
    "Gas Lift",     # Maverick-1H-26-H
    "Gas Lift",     # Maverick-1H-27-H
    "ESP",          # Maverick-1H-28-H
    "ESP",          # Maverick-1H-29-H
    "ESP",          # Maverick-1H-30-H
    "ESP",          # Maverick-1H-31-H
    "ESP",          # Maverick-1H-32-H
    "Gas Lift",     # Maverick-1H-33-H
    "Pumping Unit", # Maverick-1H-34-V
    "ESP",          # Maverick-1H-35-H
    "Gas Lift",     # Maverick-1H-36-H
    "ESP",          # Maverick-1H-37-H
    "Pumping Unit", # Maverick-1H-38-V
    "Gas Lift",     # Maverick-1H-39-H
    "Gas Lift",     # Maverick-1H-40-H
    # Carrizo (wells 41-60)
    "ESP",          # Carrizo-1H-41-H
    "ESP",          # Carrizo-1H-42-H
    "ESP",          # Carrizo-1H-43-H
    "ESP",          # Carrizo-1H-44-H
    "Gas Lift",     # Carrizo-1H-45-H
    "Gas Lift",     # Carrizo-1H-46-H
    "ESP",          # Carrizo-1H-47-H
    "Gas Lift",     # Carrizo-1H-48-H
    "ESP",          # Carrizo-1H-49-H
    "Gas Lift",     # Carrizo-1H-50-H
    "Gas Lift",     # Carrizo-1H-51-H
    "ESP",          # Carrizo-1H-52-H
    "ESP",          # Carrizo-1H-53-H
    "ESP",          # Carrizo-1H-54-H
    "Pumping Unit", # Carrizo-1H-55-V
    "Pumping Unit", # Carrizo-1H-56-V
    "Gas Lift",     # Carrizo-1H-57-H
    "Gas Lift",     # Carrizo-1H-58-H
    "ESP",          # Carrizo-1H-59-H
    "ESP",          # Carrizo-1H-60-H
    # Cotulla (wells 61-80)
    "Pumping Unit", # Cotulla-1H-61-V
    "ESP",          # Cotulla-1H-62-H
    "Gas Lift",     # Cotulla-1H-63-H
    "ESP",          # Cotulla-1H-64-H
    "Gas Lift",     # Cotulla-1H-65-H
    "ESP",          # Cotulla-1H-66-H
    "Gas Lift",     # Cotulla-1H-67-H
    "ESP",          # Cotulla-1H-68-H
    "Gas Lift",     # Cotulla-1H-69-H
    "ESP",          # Cotulla-1H-70-H
    "ESP",          # Cotulla-1H-71-H
    "Gas Lift",     # Cotulla-1H-72-H
    "Gas Lift",     # Cotulla-1H-73-H
    "Gas Lift",     # Cotulla-1H-74-H
    "ESP",          # Cotulla-1H-75-H
    "ESP",          # Cotulla-1H-76-H
    "ESP",          # Cotulla-1H-77-H
    "ESP",          # Cotulla-1H-78-H
    "Pumping Unit", # Cotulla-1H-79-V
    "ESP",          # Cotulla-1H-80-H
    # Pearsall (wells 81-100)
    "ESP",          # Pearsall-1H-81-H
    "ESP",          # Pearsall-1H-82-H
    "Gas Lift",     # Pearsall-1H-83-H
    "Pumping Unit", # Pearsall-1H-84-V
    "Gas Lift",     # Pearsall-1H-85-H
    "ESP",          # Pearsall-1H-86-H
    "ESP",          # Pearsall-1H-87-H
    "Gas Lift",     # Pearsall-1H-88-H
    "Gas Lift",     # Pearsall-1H-89-H
    "ESP",          # Pearsall-1H-90-H
    "ESP",          # Pearsall-1H-91-H
    "ESP",          # Pearsall-1H-92-H
    "Gas Lift",     # Pearsall-1H-93-H
    "Gas Lift",     # Pearsall-1H-94-H
    "Pumping Unit", # Pearsall-1H-95-V
    "ESP",          # Pearsall-1H-96-H
    "Gas Lift",     # Pearsall-1H-97-H
    "ESP",          # Pearsall-1H-98-H
    "ESP",          # Pearsall-1H-99-H
    "ESP",          # Pearsall-1H-100-H
]


def random_spud():
    # Eagle Ford development peaked 2014-2020, still active
    bucket = random.random()
    if bucket < 0.15:
        return datetime(2013, 1, 1) + timedelta(days=random.randint(0, 730))
    elif bucket < 0.50:
        return datetime(2015, 1, 1) + timedelta(days=random.randint(0, 1095))
    elif bucket < 0.80:
        return datetime(2018, 1, 1) + timedelta(days=random.randint(0, 1095))
    else:
        return datetime(2021, 1, 1) + timedelta(days=random.randint(0, 730))

# ─── EAGLE FORD FAILURE LIBRARY ──────────────────────────────────────────────
# CRITICAL: failure_cause field is REMOVED — the AI must detect these
# from sensor patterns alone. No labels. The sensor deterioration signature
# IS the training signal.
#
# Eagle Ford-specific failures — all subtle, hard to detect:
#
# ESP failures:
#   Scale Buildup — carbonate scale slowly coats impellers over months.
#     Head drops 0.5-2% per week. No sudden change. Hard to distinguish
#     from normal decline. Trigger: pump_discharge_pressure_psi declining
#     while intake_pressure stable.
#   Gas Interference — free gas slugging through pump intermittently.
#     Intermittent vibration spikes, erratic intake pressure, current
#     swings. Hard to distinguish from normal variability.
#   Emulsion Block — oil/water emulsion increases fluid viscosity.
#     Motor current rises slowly, efficiency drops. No mechanical failure.
#     Trigger: motor_current_amps rising while production flat or declining.
#
# Gas Lift failures:
#   Paraffin Deposition — paraffinic Eagle Ford crude deposits in tubing.
#     Wellhead temperature drops gradually as flow restriction increases.
#     Production index (oil per psi drawdown) declines slowly.
#     Trigger: wellhead_temp_f declining, casing_injection_pressure rising.
#   Injection Valve Erosion — valve seat erodes slowly from sand/fluid.
#     Injection rate creeps up as valve loses seal, tubing pressure drops.
#     Very gradual — 90-150 day signature. Hard to separate from reservoir.
#
# Pumping Unit failures:
#   Downhole Sensor Drift — the sensor itself is failing, not the well.
#     One sensor drifts out of range while others remain stable.
#     The AI must learn that correlated sensor drift = equipment failure
#     but uncorrelated single-sensor drift = instrumentation failure.
#   Tubing Leak — small pinhole leak develops slowly in production tubing.
#     Casing pressure rises slowly as gas migrates to annulus.
#     Production drops slowly. No obvious mechanical signature.

FAILURES = {
    "ESP": [
        {
            # Scale buildup — very slow head degradation, no sudden event
            # Trigger: discharge pressure declining while intake stable
            "window_min": 90, "window_max": 180,
            "trigger_sensor": "pump_discharge_pressure_psi",
            "secondary_direction": "discharge_drops_intake_stable",
        },
        {
            # Gas interference — intermittent slugging, erratic sensors
            # Trigger: vibration spikes with intake pressure swings
            "window_min": 30, "window_max": 75,
            "trigger_sensor": "lateral_vibration_hz",
            "secondary_direction": "erratic",
        },
        {
            # Emulsion block — rising motor current, flat/declining production
            # Trigger: motor_current_amps rising slowly over months
            "window_min": 60, "window_max": 120,
            "trigger_sensor": "motor_load_amps",
            "secondary_direction": "current_up_production_flat",
        },
    ],
    "Gas Lift": [
        {
            # Paraffin deposition — wellhead temp drops, injection pressure rises
            # Trigger: wellhead_temp_f declining
            "window_min": 45, "window_max": 90,
            "trigger_sensor": "wellhead_flowing_temp_f",
            "secondary_direction": "temp_down_pressure_up",
        },
        {
            # Injection valve erosion — very gradual, 90-150 day signature
            # Trigger: injection_rate_mscfd creeping up as valve loses seal
            "window_min": 90, "window_max": 150,
            "trigger_sensor": "lift_gas_rate_mscfd",
            "secondary_direction": "rate_up_tubing_down",
        },
    ],
    "Pumping Unit": [
        {
            # Downhole sensor drift — instrumentation failure, not well failure
            # Single sensor drifts while others stable
            "window_min": 20, "window_max": 60,
            "trigger_sensor": "downhole_pump_intake_psi",
            "secondary_direction": "single_sensor_drift",
        },
        {
            # Tubing leak — slow casing pressure rise, gradual production drop
            # Trigger: casing_pressure_psi rising slowly
            "window_min": 60, "window_max": 120,
            "trigger_sensor": "annular_casing_pressure_psi",
            "secondary_direction": "casing_up_tubing_down",
        },
    ],
}

# ─── SENSOR NORMAL RANGES ────────────────────────────────────────────────────
# Eagle Ford specifics:
# - Higher GOR wells — more gas, different pressure profiles
# - Deeper wells (~8,000-13,000 ft TVD) — higher pump intake pressures
# - Higher initial rates — Eagle Ford ESP wells can produce 500-3000+ BPD early
# - Hotter reservoir (~250-300°F BHT) — higher motor temps

SENSOR_NORMALS = {
    "ESP": {
        "gross_oil_production_bbl":              (1.0,    3000.0),   # Eagle Ford higher rate wells
        "well_test_rate_bopd":                   (0.0,    0.0),
        "flowing_tubing_pressure_psi":(300.0,  800.0),    # higher tubing pressure — deeper wells
        "annular_casing_pressure_psi":(50.0,   400.0),    # higher GOR = higher casing pressure
        "motor_winding_temp_f":       (180.0,  280.0),    # hotter — deeper reservoir
        "motor_load_amps":         (30.0,   180.0),
        "vfd_output_frequency_hz":    (45.0,   65.0),
        "pump_intake_pressure_psi":   (500.0,  4000.0),   # deeper = higher PIP
        "pump_discharge_pressure_psi":(1500.0, 6000.0),   # higher lift requirement
        "lateral_vibration_hz":       (0.0,    1.25),
    },
    "Gas Lift": {
        "gross_oil_production_bbl":                 (1.0,    2500.0),
        "well_test_rate_bopd":                      (0.0,    0.0),
        "annular_casing_pressure_psi":   (800.0,  2000.0),  # higher casing pressure Eagle Ford
        "lift_gas_injection_pressure_psi":(1200.0,2200.0),  # higher injection pressure — deeper
        "surface_tubing_pressure_psi":   (100.0,  400.0),
        "lift_gas_rate_mscfd":           (500.0,  8000.0),  # higher injection rates Eagle Ford
        "wellhead_flowing_temp_f":       (80.0,   220.0),   # Eagle Ford warmer wellhead
        "production_choke_opening_pct":  (0.0,    100.0),
        "compressor_inlet_temp_f":       (80.0,   130.0),
    },
    "Pumping Unit": {
        "gross_oil_production_bbl":        (1.0,    500.0),     # lower rate — mature vertical re-entries
        "well_test_rate_bopd":             (0.0,    0.0),
        "polished_rod_stroke_length_in":(20.0,600.0),
        "effective_plunger_stroke_in":  (20.0,600.0),
        "pumping_speed_spm":    (0.5,    12.0),
        "volumetric_efficiency_pct":    (0.0,100.0),
        "annular_casing_pressure_psi":  (50.0,3000.0),
        "wellhead_tubing_pressure_psi": (50.0,200.0),
        "surface_motor_amperage":(10.0,  150.0),
        "motor_drive_frequency_hz":(57.0, 62.0),
        "downhole_pump_intake_psi":(0.0,  3000.0),
        "gearbox_motor_temp_f": (100.0,  160.0),
    },
}

SENSOR_FAILURE_DIRECTION = {
    # ESP
    "pump_discharge_pressure_psi": "down",   # scale buildup — head degrades
    "lateral_vibration_hz":        "up",     # gas interference — vibration spikes
    "motor_load_amps":              "up",     # emulsion block — current rises
    "pump_intake_pressure_psi":    "down",   # intake drops as pump struggles
    "flowing_tubing_pressure_psi": "down",
    "annular_casing_pressure_psi": "up",
    "motor_winding_temp_f":         "up",
    "vfd_output_frequency_hz":      "down",
    # Gas Lift
    "wellhead_flowing_temp_f":      "down",   # paraffin — temp drops
    "lift_gas_rate_mscfd":          "up",     # valve erosion — rate creeps up
    "lift_gas_injection_pressure_psi":"down",
    "surface_tubing_pressure_psi":  "down",
    "production_choke_opening_pct": "down",
    # Pumping Unit
    "downhole_pump_intake_psi":     "up",     # sensor drift — can go either way
    "annular_casing_pressure_psi":  "up",     # tubing leak — casing pressure rises
    "volumetric_efficiency_pct":    "down",
    "pumping_speed_spm":            "down",
    "polished_rod_stroke_length_in":"down",
    "effective_plunger_stroke_in":  "down",
}

def normal_reading(sensor, normals):
    lo, hi = normals[sensor]
    mid    = (lo + hi) / 2
    noise  = (hi - lo) * 0.04
    val    = float(np.random.normal(mid, noise))
    return round(float(np.clip(val, lo, hi)), 2)

def init_sensor_state(sensor, normals):
    lo, hi = normals[sensor]
    mid    = (lo + hi) / 2
    noise  = (hi - lo) * 0.04
    return float(np.clip(np.random.normal(mid, noise), lo, hi))

def step_deteriorating_sensor(current_val, sensor, normals, day, total_window,
                               is_trigger, onset_day):
    lo, hi        = normals[sensor]
    rng           = hi - lo
    direction     = SENSOR_FAILURE_DIRECTION.get(sensor, "up")

    days_active   = max(0, day - onset_day)
    if days_active <= 0:
        return round(float(np.clip(np.random.normal(current_val, rng * 0.025), lo, hi)), 2)

    active_window = max(total_window - onset_day, 1)
    progress      = min(days_active / active_window, 1.0)

    # Eagle Ford failures are designed to be MORE subtle — slower drift
    # so the AI has to work harder to detect them
    base_drift    = (0.006 if is_trigger else 0.002) * rng
    if progress > 0.85:
        accel     = 1.0 + (progress - 0.85) * 10.0
    else:
        accel     = 1.0 + progress * 0.4   # even flatter early signal
    drift_scale   = base_drift * accel

    bias = drift_scale * progress if direction == "up" else -drift_scale * progress

    # More recoveries — gas interference and scale buildup have frequent partial recoveries
    recovery_prob = max(0.0, 0.55 - progress * 0.55)  # higher recovery rate — Eagle Ford design
    if random.random() < recovery_prob:
        bias *= -random.uniform(0.5, 1.2)

    noise   = rng * (0.030 + progress * (0.025 if is_trigger else 0.012))  # more noise
    new_val = current_val + bias + float(np.random.normal(0, noise))
    # Sensor-specific absolute minimum floors — realistic lower bounds
    SENSOR_MINIMUMS = {
        "pump_intake_pressure_psi":      20.0,
        "pump_discharge_pressure_psi":  100.0,
        "flowing_tubing_pressure_psi":   15.0,
        "annular_casing_pressure_psi":    5.0,
        "motor_winding_temp_f":         100.0,
        "motor_load_amps":                5.0,
        "vfd_output_frequency_hz":       30.0,
        "lateral_vibration_hz":           0.005,
        "lift_gas_rate_mscfd":           20.0,
        "lift_gas_injection_pressure_psi":50.0,
        "surface_tubing_pressure_psi":   15.0,
        "wellhead_flowing_temp_f":       40.0,
        "production_choke_opening_pct":   0.5,
        "compressor_inlet_temp_f":       40.0,
        "pumping_speed_spm":              0.1,
        "volumetric_efficiency_pct":      1.0,
        "downhole_pump_intake_psi":      10.0,
        "gearbox_motor_temp_f":          60.0,
        "surface_motor_amperage":         2.0,
        "motor_drive_frequency_hz":      55.0,
    }
    SENSOR_MAXIMUMS = {
        "motor_load_amps":               280.0,  # overload trip point
        "lift_gas_rate_mscfd":         15000.0,  # physical injection limit
        "motor_winding_temp_f":          380.0,  # motor burn threshold
        "pump_discharge_pressure_psi": 8000.0,   # max burst pressure
        "production_choke_opening_pct": 100.0,  # cannot exceed fully open
        "volumetric_efficiency_pct":    100.0,  # pump fillage cannot exceed 100%
        "vfd_output_frequency_hz":      75.0,   # VFD hard upper limit
    }
    floor_val = SENSOR_MINIMUMS.get(sensor, 0.01)
    ceil_val  = SENSOR_MAXIMUMS.get(sensor, hi * 2.2)
    new_val = max(new_val, floor_val)
    new_val = min(new_val, ceil_val)

    return round(new_val, 2)

def step_gross_net_stroke(gross_cur, net_cur, gross_base, net_base,
                           day, total_window, onset_day):
    days_active = max(0, day - onset_day)
    if days_active <= 0:
        gross = max(1.0, float(np.random.normal(gross_base, gross_base * 0.01)))
        net   = max(1.0, float(min(np.random.normal(net_base, net_base * 0.012),
                                    gross * 0.98)))
        return round(gross, 2), round(net, 2)

    active_window = max(total_window - onset_day, 1)
    progress      = min(days_active / active_window, 1.0)
    g_drift = gross_base * 0.002 * (1.0 + progress * 2.5)
    n_drift = net_base   * 0.004 * (1.0 + progress * 3.0)
    g_bias  = -g_drift * progress
    n_bias  = -n_drift * progress
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

    days = len(dates)
    t    = np.arange(days)

    # Eagle Ford decline characteristics:
    # Higher b values — steeper early decline, longer flat tail (tight oil behavior)
    variance_map = {"Gas Lift": 0.045, "Pumping Unit": 0.035, "ESP": 0.025}
    variance     = variance_map[lift]

    b_map = {
        "Pumping Unit": random.uniform(0.4, 0.7),
        "ESP":          random.uniform(1.0, 1.5),   # steeper Eagle Ford horizontal decline
        "Gas Lift":     random.uniform(0.8, 1.2),
    }
    Di_map = {
        "Pumping Unit": random.uniform(0.0003, 0.0007),
        "ESP":          random.uniform(0.0010, 0.0020),  # steeper early Eagle Ford
        "Gas Lift":     random.uniform(0.0008, 0.0015),
    }
    floor_map = {
        "Pumping Unit": random.uniform(5.0,  15.0),
        "ESP":          random.uniform(40.0, 100.0),  # higher economic floor Eagle Ford
        "Gas Lift":     random.uniform(20.0,  50.0),
    }

    b     = b_map[lift]
    Di    = Di_map[lift]
    floor = floor_map[lift]

    theoretical  = base_rate / np.power(1.0 + b * Di * t, 1.0 / b)
    theoretical  = np.maximum(theoretical, floor)
    prod_noise   = np.random.normal(1.0, variance, days)
    rates        = np.clip(theoretical * prod_noise, floor * 0.5, None)

    # Eagle Ford water and gas — higher GOR, water cut rises faster
    PROD_PARAMS_INLINE = {
        "Pumping Unit": {"water_base":(30.0,500.0),"gas_base":(20.0,200.0),
                         "water_noise":0.06,"gas_noise":0.05,
                         "water_decline":0.0001,"gas_decline":0.0006,"water_rise":0.00015},
        "ESP":          {"water_base":(300.0,8000.0),"gas_base":(100.0,2000.0),
                         "water_noise":0.05,"gas_noise":0.04,
                         "water_decline":0.00008,"gas_decline":0.0008,"water_rise":0.0002},
        "Gas Lift":     {"water_base":(200.0,4000.0),"gas_base":(500.0,5000.0),
                         "water_noise":0.06,"gas_noise":0.05,
                         "water_decline":0.0001,"gas_decline":0.0005,"water_rise":0.00018},
    }
    pp           = PROD_PARAMS_INLINE[lift]
    water_base_v = random.uniform(*pp["water_base"])
    gas_base_v   = random.uniform(*pp["gas_base"])
    water_th     = water_base_v*(1+pp["water_rise"]*t)*np.exp(-pp["water_decline"]*t)
    gas_th       = gas_base_v*np.exp(-pp["gas_decline"]*t)
    water_rates  = np.clip(water_th*np.random.normal(1.0,pp["water_noise"],days), 1.0, None)
    gas_rates    = np.clip(gas_th  *np.random.normal(1.0,pp["gas_noise"],  days), 0.5, None)

    test_day_of_month = random.randint(5, 25)

    # IP flush parameters — set per well, applied if this well is in IP_FLUSH_WELL_NUMBERS
    # Exact match — lift pool and suffixes are now fixed permanently
    ip_flush_active  = asset_id in IP_FLUSH_WELL_NUMBERS
    ip_flush_window  = random.randint(60, 90) if ip_flush_active else 0
    ip_flush_mult    = random.uniform(2.0, 4.0) if ip_flush_active else 1.0

    normals     = {k: v for k, v in SENSOR_NORMALS[lift].items()
                   if k not in ('gross_oil_production_bbl', 'well_test_rate_bopd', 'gross_stroke_len', 'net_stroke_len')}
    sensor_keys = list(normals.keys())

    history       = []
    status        = "PRODUCING"
    days_online   = 0
    downtime_rem  = 0

    detr_active      = False
    detr_total_days  = 0
    detr_elapsed     = 0
    detr_failure_def = None
    sensor_state     = {}
    sensor_onset     = {}
    gross_cur        = gross_base
    net_cur          = net_base

    for i, current_date in enumerate(dates):
        date_str    = current_date.strftime("%Y-%m-%d")
        is_test_day = (current_date.day == test_day_of_month)
        row         = {"date": date_str}

        if status == "PRODUCING":
            days_online += 1

            # Hazard function — Eagle Ford ESPs fail more frequently due to scale/gas
            fail_prob = 0.00006
            if lift == "ESP" and days_online > 365:
                fail_prob += 0.004 * ((days_online - 365) / 100)
            elif lift == "Gas Lift" and days_online > 500:
                fail_prob += 0.0015 * ((days_online - 500) / 100)
            elif lift == "Pumping Unit" and days_online > 400:
                fail_prob += 0.001 * ((days_online - 400) / 100)

            if not detr_active and random.random() < fail_prob:
                detr_active      = True
                detr_failure_def = random.choice(FAILURES[lift])
                detr_total_days  = random.randint(detr_failure_def["window_min"],
                                                  detr_failure_def["window_max"])
                detr_elapsed     = 0
                trigger_sens     = detr_failure_def["trigger_sensor"]

                sensor_state = {s: init_sensor_state(s, normals) for s in sensor_keys}
                sensor_onset = {}
                for s in sensor_keys:
                    if s == trigger_sens:
                        sensor_onset[s] = 0
                    else:
                        sensor_onset[s] = random.randint(
                            int(detr_total_days * 0.15),
                            int(detr_total_days * 0.55)
                        )
                gross_cur = gross_base
                net_cur   = net_base

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
                        onset_day=sensor_onset.get("polished_rod_stroke_length_in", 0)
                    )

                if detr_elapsed >= detr_total_days:
                    status        = "SHUT-IN"
                    downtime_rem  = random.randint(5, 14)
                    detr_active   = False
                    detr_failure_def = None
                    sensor_state  = {}
                    sensor_onset  = {}
                    row["gross_oil_production_bbl"]   = 0.0
                    row["well_test_rate_bopd"]        = None
                    if lift == "Pumping Unit":
                        row["polished_rod_stroke_length_in"] = 0.0
                        row["effective_plunger_stroke_in"]   = 0.0
                    for s in sensor_keys:
                        row[s] = 0.0
                    row["run_status"]      = "SHUT-IN"
                    row["produced_water_volume_bbl"] = 0.0
                    row["casinghead_gas_mcf"]   = 0.0
                    history.append(row)
                    continue

                sensor_vals = new_sensor_vals

            else:
                sensor_vals = {s: normal_reading(s, normals) for s in sensor_keys}
                if lift == "Pumping Unit":
                    gross_cur = max(1.0, float(np.random.normal(gross_base, gross_base * 0.01)))
                    net_cur   = max(1.0, float(min(
                        np.random.normal(net_base, net_base * 0.015),
                        gross_cur * 0.98
                    )))

            oil = round(float(rates[i]), 2)
            # Apply IP flush multiplier in early flush window
            if ip_flush_active and i < ip_flush_window:
                # Steep decay from 2-4x back to normal over flush window
                flush_factor = ip_flush_mult * (1.0 - i / ip_flush_window) + 1.0
                oil = round(oil * flush_factor, 2)
            row["gross_oil_production_bbl"] = oil

            if is_test_day:
                test_var        = random.uniform(0.05, 0.15)
                test_dir        = random.choice([1, -1])
                row["well_test_rate_bopd"] = round(oil * (1 + test_dir * test_var), 2)
            else:
                row["well_test_rate_bopd"] = None

            if lift == "Pumping Unit":
                row["polished_rod_stroke_length_in"] = round(gross_cur, 2)
                row["effective_plunger_stroke_in"]   = round(net_cur, 2)

            for s, v in sensor_vals.items():
                row[s] = v

            row["run_status"]      = "PRODUCING"
            # Water also elevated during IP flush — frac fluid returns
            water_val = float(water_rates[i])
            if ip_flush_active and i < ip_flush_window:
                water_flush = ip_flush_mult * 1.5 * (1.0 - i / ip_flush_window) + 1.0
                water_val   = water_val * water_flush
            row["produced_water_volume_bbl"] = round(water_val, 2)
            row["casinghead_gas_mcf"]   = round(float(gas_rates[i]),   2)

        else:  # SHUT-IN
            row["gross_oil_production_bbl"]   = 0.0
            row["well_test_rate_bopd"]        = None
            if lift == "Pumping Unit":
                row["polished_rod_stroke_length_in"] = 0.0
                row["effective_plunger_stroke_in"]   = 0.0
            for s in sensor_keys:
                row[s] = 0.0
            downtime_rem -= 1
            row["run_status"]      = "SHUT-IN"
            row["produced_water_volume_bbl"] = 0.0
            row["casinghead_gas_mcf"]   = 0.0
            if downtime_rem <= 0:
                status        = "PRODUCING"
                days_online   = 0
                gross_cur     = gross_base
                net_cur       = net_base

        history.append(row)
    return history

# ─── LEASE TYPE AND NRI/WI ────────────────────────────────────────────────────
# Fixed per-lease structure — decoupled fields
# Lease type and NRI are independent — NRI reflects negotiated agreement

def assign_lease_ownership():
    """Fixed per-lease NRI/WI — permanent every rebuild.
    South Texas Energy Partners lease structure.
    """
    return {
        "Bravo":    {"working_interest": 1.0, "lease_type": "Fee Land",         "net_revenue_interest": 0.750},
        "Maverick": {"working_interest": 1.0, "lease_type": "University Land",  "net_revenue_interest": 0.875},
        "Carrizo":  {"working_interest": 1.0, "lease_type": "PSF Land",         "net_revenue_interest": 0.375},
        "Cotulla":  {"working_interest": 1.0, "lease_type": "Railroad ROW",     "net_revenue_interest": 1.000},
        "Pearsall": {"working_interest": 1.0, "lease_type": "Fee Land",         "net_revenue_interest": 0.750},
    }

# ─── ESP PUMP CURVES ──────────────────────────────────────────────────────────
# Eagle Ford ESPs run harder — deeper, higher flow rates, higher temps
# Base Curve Method — store at 60 Hz, apply Affinity Laws at runtime
# Physics-consistent synthetic values — NOT from official manufacturer spec sheets

ESP_PUMP_MODELS = [
    {
        # High-volume deep Eagle Ford ESP — 5.5" casing horizontal completion
        "manufacturer": "Centrilift",
        "model_series": "GN5600",
        "stage_count_range": (80, 160),
        "min_flow_bpd": 2000, "max_flow_bpd": 6000, "bep_bpd": 4000,
        "curve_points": [
            {"flow_rate_bpd": 1500, "head_ft_per_stage": 26.8, "efficiency_percent": 35.0, "brake_horsepower": 0.846},
            {"flow_rate_bpd": 2500, "head_ft_per_stage": 25.4, "efficiency_percent": 55.0, "brake_horsepower": 0.850},
            {"flow_rate_bpd": 4000, "head_ft_per_stage": 22.8, "efficiency_percent": 70.0, "brake_horsepower": 0.960},
            {"flow_rate_bpd": 5200, "head_ft_per_stage": 18.5, "efficiency_percent": 61.0, "brake_horsepower": 1.162},
            {"flow_rate_bpd": 6000, "head_ft_per_stage": 12.4, "efficiency_percent": 42.0, "brake_horsepower": 1.305},
        ],
        "poly": {"a": -0.00000092, "b": 0.00445, "c": 24.62},
    },
    {
        # Medium-volume Eagle Ford ESP — workhorse model
        "manufacturer": "REDA",
        "model_series": "D3000N",
        "stage_count_range": (60, 130),
        "min_flow_bpd": 1200, "max_flow_bpd": 4000, "bep_bpd": 2800,
        "curve_points": [
            {"flow_rate_bpd": 800,  "head_ft_per_stage": 32.1, "efficiency_percent": 32.0, "brake_horsepower": 0.591},
            {"flow_rate_bpd": 1500, "head_ft_per_stage": 30.5, "efficiency_percent": 54.0, "brake_horsepower": 0.624},
            {"flow_rate_bpd": 2800, "head_ft_per_stage": 27.2, "efficiency_percent": 70.0, "brake_horsepower": 0.801},
            {"flow_rate_bpd": 3500, "head_ft_per_stage": 22.4, "efficiency_percent": 61.0, "brake_horsepower": 0.947},
            {"flow_rate_bpd": 4000, "head_ft_per_stage": 16.8, "efficiency_percent": 44.0, "brake_horsepower": 1.125},
        ],
        "poly": {"a": -0.00000281, "b": 0.00698, "c": 29.18},
    },
    {
        # Lower-volume ESP for mature Eagle Ford wells
        "manufacturer": "Centrilift",
        "model_series": "FC1200",
        "stage_count_range": (60, 120),
        "min_flow_bpd": 800,  "max_flow_bpd": 1600, "bep_bpd": 1200,
        "curve_points": [
            {"flow_rate_bpd": 600,  "head_ft_per_stage": 34.2, "efficiency_percent": 42.0, "brake_horsepower": 0.360},
            {"flow_rate_bpd": 900,  "head_ft_per_stage": 32.8, "efficiency_percent": 61.0, "brake_horsepower": 0.356},
            {"flow_rate_bpd": 1200, "head_ft_per_stage": 29.4, "efficiency_percent": 73.5, "brake_horsepower": 0.354},
            {"flow_rate_bpd": 1450, "head_ft_per_stage": 25.1, "efficiency_percent": 65.0, "brake_horsepower": 0.412},
            {"flow_rate_bpd": 1600, "head_ft_per_stage": 19.8, "efficiency_percent": 49.0, "brake_horsepower": 0.476},
        ],
        "poly": {"a": -0.00001548, "b": 0.02031, "c": 27.41},
    },
]

def generate_pump_curve(well_id):
    model       = random.choice(ESP_PUMP_MODELS)
    stage_count = random.randint(*model["stage_count_range"])
    return {
        "pump_metadata": {
            "manufacturer":              model["manufacturer"],
            "model":                     model["model_series"],
            "stage_count":               stage_count,
            "baseline_frequency_hz":     60,
            "min_operating_flow_bpd":    model["min_flow_bpd"],
            "max_operating_flow_bpd":    model["max_flow_bpd"],
            "best_efficiency_point_bpd": model["bep_bpd"],
        },
        "curve_data_60hz": model["curve_points"],
        "polynomial_coefficients": {
            "head_curve": {
                "a": model["poly"]["a"],
                "b": model["poly"]["b"],
                "c": model["poly"]["c"],
            }
        },
        "affinity_law_notes": (
            "To recalculate at speed N2 from baseline 60 Hz (N1): "
            "Q2 = Q1*(N2/N1), H2 = H1*(N2/N1)^2, P2 = P1*(N2/N1)^3"
        ),
    }

def build_wells(lease_ownership):
    wells    = []
    lift_idx = 0
    for lease in LEASES:
        ownership = lease_ownership[lease]
        for w in range(1, WELLS_PER_LEASE + 1):
            well_num   = LEASES.index(lease) * WELLS_PER_LEASE + w
            lift       = LIFT_POOL[lift_idx % len(LIFT_POOL)]
            lift_idx  += 1
            suffix     = "V" if lift == "Pumping Unit" else "H"
            asset_id   = f"{lease}-1H-{well_num}-{suffix}"
            gross_base = random.uniform(120.0, 168.0)
            net_base   = gross_base * random.uniform(0.85, 0.92)
            # Eagle Ford higher initial rates
            base_rate  = random.uniform(400, 2800) if lift == "ESP" else \
                         random.uniform(200, 1500) if lift == "Gas Lift" else \
                         random.uniform(50, 300)
            wells.append({
                "asset_id":   asset_id,
                "lease":      lease,
                "lift":       lift,
                "spud_date":  random_spud(),
                "base_rate":  base_rate,
                "gross_base": gross_base,
                "net_base":   net_base,
                "ownership":  ownership,
            })
    return wells

def get_personnel(lease):
    engineers = {
        "Bravo":    "Marcus Reyes",
        "Maverick": "Sandra Ochoa",
        "Carrizo":  "Marcus Reyes",
        "Cotulla":  "Sandra Ochoa",
        "Pearsall": "Marcus Reyes",
    }
    return {
        "prod_technician": {
            "name":  f"{lease} Production Tech",
            "role":  "Production Technician",
            "email": f"prodtech.{lease.lower()}@steppartners.com"
        },
        "well_perf_engineer": {
            "name":  engineers[lease],
            "role":  "Well Performance Engineer",
            "email": "wellperf@steppartners.com"
        }
    }

if __name__ == "__main__":
    print("South Texas Energy Partners — Digital Twin")
    print(f"100 wells across 5 leases | End date: {END_DATE.strftime('%Y-%m-%d')}")
    print("Eagle Ford Shale — South Texas")
    print("─" * 60)

    lease_ownership = assign_lease_ownership()
    print("Lease NRI/WI assignments:")
    for lease, own in lease_ownership.items():
        print(f"  {lease:<10} WI={own['working_interest']*100:.1f}%  NRI={own['net_revenue_interest']*100:.1f}%  ({own['lease_type']})")

    wells  = build_wells(lease_ownership)
    output = {"company": COMPANY, "assets": {}}

    for i, w in enumerate(wells):
        aid = w["asset_id"]
        print(f"  [{i+1:>3}/100] {aid} ({w['lift']}, spud {w['spud_date'].strftime('%Y-%m-%d')})...")
        history    = simulate_well(aid, w["lift"], w["spud_date"],
                                   w["base_rate"], w["gross_base"], w["net_base"])
        pump_curve = generate_pump_curve(aid) if w["lift"] == "ESP" else None

        asset_dict = {
            "type":                 "Well",
            "lift":                 w["lift"],
            "organization_id":      ORGANIZATION_ID,
            "area":                 AREA_MAP.get(w["lease"], ""),
            "working_interest":     w["ownership"]["working_interest"],
            "lease_type":           w["ownership"]["lease_type"],
            "net_revenue_interest": w["ownership"]["net_revenue_interest"],
            "assigned_personnel":   get_personnel(w["lease"]),
            "history":              history,
            "financial_ledger":     []
        }
        if pump_curve is not None:
            asset_dict["pump_curve"] = pump_curve
        output["assets"][aid] = asset_dict

    with open("south_texas_engineering.json", "w") as f:
        json.dump(output, f, indent=2)

    total_days     = sum(len(a["history"]) for a in output["assets"].values())
    total_failures = sum(
        sum(1 for d in a["history"] if d["run_status"] == "SHUT-IN")
        for a in output["assets"].values()
    )
    print(f"\n✓ Done")
    print(f"  Assets:           {len(output['assets'])}")
    print(f"  Total daily rows: {total_days:,}")
    print(f"  Failure events:   {total_failures:,}")
