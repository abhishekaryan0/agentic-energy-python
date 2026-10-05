"""
STEP V2 digital twin engine — same role as v1 generate_digital_twin.py.

Used by v2/pipeline.py to generate wells/supporting history directly into
Snowflake (*_V2 tables). Does not write STEP_V2_Master.json.
"""

from __future__ import annotations

import collections
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

_V2_DIR = Path(__file__).resolve().parent
if str(_V2_DIR) not in sys.path:
    sys.path.insert(0, str(_V2_DIR))

try:
    from config import STEP_ORGANIZATION_ID as ORGANIZATION_ID
except ImportError:
    ORGANIZATION_ID = (
        os.getenv("STEP_ORGANIZATION_ID")
        or os.getenv("ORGANIZATION_ID")
        or "d270f89a-aada-4ff2-adb0-68b3e8087565"
    )


def usd(x):
    """Round a currency amount to cents, half-up, as a plain Python float."""
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


random.seed(20260708)
np.random.seed(20260708)

COMPANY = "South Texas Energy Partners"
END_DATE = datetime(2026, 7, 8)
START_MIN       = datetime(2008, 1, 1)
START_MAX       = datetime(2011, 12, 31)

# Working interest and royalty burden per lease. WI was previously 1.0 on all
# 120 assets, which made the WI math check vacuous (total == gross on every
# invoice) and gave the model no partial-interest case. NRI is now derived as
# WI*(1-royalty) rather than hardcoded, so the old Cotulla NRI of 1.000 (which
# implies a zero royalty burden — not a real Eagle Ford lease) and Carrizo's
# 0.375 are replaced by defensible South Texas values.
LEASES = {
    "Bravo":    {"wi":1.000,"royalty":0.2500,"lt":"Fee Land",       "area":"Brush Country Area","county":"La Salle",
                 "dist":{"ESP":11,"Gas Lift":7,"Pumping Unit":2}},
    "Maverick": {"wi":0.750,"royalty":0.2000,"lt":"University Land","area":"Rio Grande Area",   "county":"Maverick",
                 "dist":{"ESP":11,"Gas Lift":7,"Pumping Unit":2}},
    "Carrizo":  {"wi":0.500,"royalty":0.2250,"lt":"PSF Land",       "area":"Karnes Area",       "county":"Karnes",
                 "dist":{"ESP":11,"Gas Lift":7,"Pumping Unit":2}},
    "Cotulla":  {"wi":0.875,"royalty":0.1875,"lt":"Railroad ROW",   "area":"Rio Grande Area",   "county":"Dimmit",
                 "dist":{"ESP":11,"Gas Lift":7,"Pumping Unit":2}},
    "Pearsall": {"wi":0.625,"royalty":0.2500,"lt":"Fee Land",       "area":"Brush Country Area","county":"Frio",
                 "dist":{"ESP":11,"Gas Lift":7,"Pumping Unit":2}},
}
for _k,_v in LEASES.items():
    _v["nri"]=round(_v["wi"]*(1.0-_v["royalty"]),6)
PERSONNEL = {
    "Bravo":"Marcus Reyes","Maverick":"Sandra Ochoa",
    "Carrizo":"Marcus Reyes","Cotulla":"Sandra Ochoa","Pearsall":"Marcus Reyes",
}

# 10 clean wells — 2 per lease (1 ESP + 1 Gas Lift).
# SCATTERED through each lease's 20-well block. They used to be wells 001/002 of
# every lease, so "well number ends 001 or 002" identified the healthy cohort
# with perfect accuracy without a model ever reading a sensor.
LEASE_BASE = {"Bravo":0,"Maverick":20,"Carrizo":40,"Cotulla":60,"Pearsall":80}
# (0-based slot in the lease pool for the clean ESP, clean Gas Lift)
CLEAN_SLOTS = {"Bravo":(5,14), "Maverick":(3,11), "Carrizo":(8,16),
               "Cotulla":(2,12), "Pearsall":(6,17)}
CLEAN_WELLS = {lz:[f"{lz}-1H-{LEASE_BASE[lz]+i+1:03d}-V2-H" for i in slots]
               for lz,slots in CLEAN_SLOTS.items()}
ALL_CLEAN = {w for ws in CLEAN_WELLS.values() for w in ws}
if len(ALL_CLEAN)!=10: raise SystemExit("clean well selection no longer totals 10")

# ── Ordinary (non-failure) well downtime ─────────────────────────────────────
# Every well, including the clean cohort, takes routine outages: weather, power
# loss, offset frac hits, facility upsets, pumper error, scheduled surface work.
# Without this the 10 clean wells run 100% uptime for 16+ straight years and
# run_status alone separates the healthy cohort perfectly.
# The long tail is REQUIRED and is allocated by construction, not by a per-outage
# coin flip: a short-only routine distribution leaves "any outage >= N days is a
# failure" a perfect rule, and "any well with no long outage" a perfect clean
# cohort detector.
ROUTINE_OUTAGES_PER_YEAR = (1.0, 2.4)
ROUTINE_SHORT_DAYS       = (1, 3)
ROUTINE_LONG_DAYS        = (4, 24)
ROUTINE_LONG_FRAC        = 0.28

def routine_outages(sd, ed):
    """Ordinary non-failure downtime. {start_date: duration_days}."""
    out={}; total=(ed-sd).days
    if total<=0: return out
    n=max(1,int(round(total/365.25*random.uniform(*ROUTINE_OUTAGES_PER_YEAR))))
    n_long=min(n,max(2,int(round(n*ROUTINE_LONG_FRAC))))
    kinds=[ROUTINE_LONG_DAYS]*n_long+[ROUTINE_SHORT_DAYS]*(n-n_long)
    random.shuffle(kinds)
    for rng in kinds:
        st=sd+timedelta(days=random.randint(0,max(0,total-1)))
        out[st.strftime('%Y-%m-%d')]=random.randint(*rng)
    return out

# Vendors
VENDORS = [
    "CPS Energy - South Texas","South Texas Energy Partners Internal",
    "Monarch Chemical Services","Lone Star Telemetry","Eagle Ford Water Services",
    "South Texas Hot Oil","Laredo Roustabout Services","Frontier Well Services",
    "South Texas Compression LLC","South Texas Electrical LLC",
]
LOE_VENDORS   = ["South Texas Energy Partners Internal","Monarch Chemical Services",
                  "Lone Star Telemetry","Eagle Ford Water Services","South Texas Hot Oil",
                  "Laredo Roustabout Services","CPS Energy - South Texas"]
WO_VENDORS    = ["Frontier Well Services","South Texas Hot Oil","Laredo Roustabout Services"]
SUPP_VENDORS  = {"Facility":["South Texas Electrical LLC","South Texas Energy Partners Internal"],
                 "Compressor":["South Texas Compression LLC","South Texas Electrical LLC"],
                 "Tank Battery":["South Texas Energy Partners Internal","Eagle Ford Water Services"],
                 "SWD Well":["Eagle Ford Water Services","Frontier Well Services"]}

# Invoice counters per year
_inv_counters = defaultdict(int)
_sup_counters = defaultdict(int)

def next_invoice(year, prefix="STEP"):
    _inv_counters[year] += 1
    return f"{prefix}-{year}-{_inv_counters[year]:05d}"

def next_sup_invoice(year):
    _sup_counters[year] += 1
    return f"STEP-{year}-SUP-{_sup_counters[year]:05d}"

_rev_counters = defaultdict(int)
def next_revenue(year):
    _rev_counters[year] += 1
    return f"STEP-REV-{year}-{_rev_counters[year]:05d}"

# ── Price deck — South Texas / Eagle Ford ────────────────────────────────────
# Eagle Ford crude moves on Gulf Coast pricing (LLS / Magellan East Houston),
# which trades at a PREMIUM to WTI Cushing because it reaches waterborne markets.
# That is the opposite of a Permian barrel, which was discounted at Midland for
# most of this period. The premium was very wide while Cushing was landlocked
# (2011-2014) and compressed once Gulf Coast takeaway and the export ban repeal
# arrived. Gas prices off Houston Ship Channel, which sits within a few cents of
# Henry Hub — Eagle Ford never saw anything like the Waha collapse.
WTI_MONTHLY = {
    2008: [92.97,95.39,105.45,112.58,125.40,133.88,133.37,116.67,104.11,76.61,57.31,41.12],
    2014: [94.62,100.82,100.80,102.07,102.18,105.79,103.59,96.54,93.21,84.40,75.79,59.29],
    2015: [47.22,50.58,47.82,54.45,59.27,59.82,50.90,42.87,45.48,46.22,42.44,37.19],
    2016: [31.68,30.32,37.55,40.75,46.71,48.76,44.65,44.72,45.18,49.78,45.66,51.97],
    2020: [57.52,50.54,29.21,16.55,28.56,38.31,40.71,42.34,39.63,39.40,40.94,47.02],
    2021: [52.00,59.04,62.33,61.72,65.17,71.38,72.49,67.73,71.65,81.48,79.15,71.71],
    2022: [83.22,91.64,108.50,101.78,109.55,114.84,101.62,91.48,84.26,87.55,84.37,76.44],
}
WTI_ANNUAL = {2009:61.95,2010:79.48,2011:94.88,2012:94.05,2013:97.98,2017:50.80,
              2018:65.23,2019:56.99,2023:77.58,2024:76.60,2025:70.10,2026:68.40}
HH_ANNUAL  = {2008:8.86,2009:3.94,2010:4.37,2011:4.00,2012:2.75,2013:3.73,2014:4.37,
              2015:2.62,2016:2.52,2017:2.99,2018:3.15,2019:2.56,2020:2.03,2021:3.89,
              2022:6.42,2023:2.54,2024:2.19,2025:3.40,2026:3.55}
# LLS / MEH premium over WTI Cushing ($/bbl).
LLS_PREMIUM = {2008: 2.20,2009: 1.60,2010: 2.80,2011:15.90,2012:15.60,2013: 9.40,
               2014: 4.30,2015: 2.60,2016: 1.70,2017: 2.40,2018: 4.20,2019: 5.30,
               2020: 1.80,2021: 1.50,2022: 2.60,2023: 2.10,2024: 1.90,2025: 1.80,2026: 1.75}
# Houston Ship Channel basis vs Henry Hub ($/mcf) — small, occasionally positive.
HSC_BASIS   = {2008:-0.10,2009:-0.08,2010:-0.06,2011:-0.05,2012:-0.04,2013:-0.03,
               2014: 0.02,2015:-0.05,2016:-0.06,2017:-0.05,2018:-0.08,2019:-0.12,
               2020:-0.06,2021: 0.05,2022: 0.18,2023:-0.02,2024:-0.09,2025: 0.04,2026: 0.03}

# Texas fiscal terms — same statute as any other Texas lease.
SEV_TAX_OIL   = 0.046    # 4.6% of market value
SEV_TAX_GAS   = 0.075    # 7.5% of market value
AD_VALOREM    = 0.011    # ~1.1% of value, county assessed
# Eagle Ford sits on dense gathering with short hauls to Corpus Christi and
# Houston, so oil transport is well below a trucked Permian barrel.
OIL_TRANSPORT = 1.35     # $/bbl gathering
GAS_GPT       = 0.58     # $/mcf gathering, processing, treating
# Eagle Ford casinghead gas is rich, richest in the Karnes condensate window.
NGL_YIELD_BY_AREA = {"Karnes Area":(95.0,145.0),"Brush Country Area":(70.0,115.0),
                     "Rio Grande Area":(55.0,95.0)}
NGL_PCT_WTI   = 0.38     # NGL barrel realizes ~38% of WTI

def _monthly_from_annual(anchor, rng, vol=0.075):
    """12 monthly values as a random walk normalized to the annual mean."""
    w=[]; x=1.0
    for _ in range(12):
        x*=(1.0+rng.gauss(0,vol)); w.append(x)
    m=sum(w)/12.0
    return [anchor*v/m for v in w]

def build_price_deck():
    """Monthly realized-price deck keyed 'YYYY-MM'. Deterministic under the seed."""
    rng=random.Random(20260708)
    deck={}
    for yr in range(2008,2027):
        wti=WTI_MONTHLY.get(yr) or _monthly_from_annual(WTI_ANNUAL[yr],rng,0.085)
        hh =_monthly_from_annual(HH_ANNUAL[yr],rng,0.110)
        for i in range(12):
            pr=LLS_PREMIUM[yr]*(1.0+rng.gauss(0,0.16))
            oil=max(1.00,wti[i]+pr)
            hb=HSC_BASIS[yr]+rng.gauss(0,0.05)
            gas=max(0.04,hh[i]+hb)
            deck[f"{yr}-{i+1:02d}"]={
                "wti_cushing_usd_bbl":round(wti[i],2),
                "lls_premium_usd_bbl":round(pr,2),
                "realized_oil_usd_bbl":round(oil,2),
                "henry_hub_usd_mcf":round(hh[i],3),
                "hsc_basis_usd_mcf":round(hb,3),
                "realized_gas_usd_mcf":round(gas,3),
                "realized_ngl_usd_bbl":round(wti[i]*NGL_PCT_WTI,2)}
    return deck

PRICE_DECK=build_price_deck()

# ── Failure types ─────────────────────────────────────────────────────────────
ESP_TYPES = [
    ("Gas Interference - Minor",            60, 180),
    ("Gas Interference - Severe",           90, 270),
    ("Thermal Burnout - Gradual",          120, 365),
    ("Thermal Burnout - Rapid",             20,  60),
    ("Scale Buildup - Slow",               180, 365),
    ("Scale Buildup - Fast",                60, 120),
    ("Emulsion Block - Gradual",           120, 365),
    ("Emulsion Block - Acute",              30,  90),
    ("Underperformance - PIP High Stable", 120, 365),
    ("Underperformance - PIP High Declining",90,270),
]
GL_TYPES = [
    ("Paraffin Deposition - Mechanical",         45,  90),
    ("Paraffin Deposition - Corrosion-Assisted", 90, 270),
    ("Injection Valve Erosion - Sand",           60, 150),
    ("Injection Valve Erosion - Chemical",       90, 270),
    ("Tubing Hole - Pinhole",                    90, 270),
    ("Tubing Hole - Large",                      30,  90),
]
PU_TYPES = [
    ("Downhole Sensor Drift - Single", 20,  60),
    ("Downhole Sensor Drift - Multi",  30,  90),
    ("Tubing Leak - Pinhole",          90, 270),
    ("Tubing Leak - Section",          30,  90),
    ("Underperformance - Runtime",    120, 365),
    # Rod-string failures — the most common beam-pump failure in the field.
    ("Rod Wear - Coupling",            90, 270),
    ("Rod Wear - Tubing Contact",     120, 330),
    ("Rod Part - Fatigue Break",       20,  60),
    ("Rod Part - Corrosion Break",     60, 180),
]

# ── Failure-mode selection tuning ─────────────────────────────────────────────
# Sensor evidence BIASES which failure mode comes next; it does not hard-select
# one family. A hard cascade collapses the mix onto whichever condition fires
# most often (a temp threshold of 220 sat below the median degraded motor temp of
# ~264F and produced 73% thermal burnout). Adjust these to retune the mix.
SEL_WEIGHT      = 2.5    # multiplier applied to families a sensor condition implicates
SEL_REPEAT_WT   = 0.35   # weight on repeating the current mode (possible, less likely)
ESP_PIP_HIGH    = 800    # psi  — high pump intake pressure
ESP_TEMP_HIGH   = 270    # F    — hot motor (recalibrated from 220; see note above)
ESP_VIB_HIGH    = 1.0    # Hz   — high lateral vibration
ESP_DISC_LOW    = 2000   # psi  — low discharge pressure
GL_GAP_LOW      = 300    # psi  — casing/tubing pressure gap
GL_WHT_LOW      = 120    # F    — wellhead flowing temp
GL_INJ_HIGH     = 800    # psi  — lift gas injection pressure
PU_CAS_HIGH     = 250    # psi  — casing pressure
PU_RUNTIME_HIGH = 22     # hrs  — daily runtime
PU_FILL_LOW     = 72     # %    — pump fillage / volumetric efficiency

def _pick(types, current_scenario, boosts):
    """Weighted choice over `types`. `boosts` maps a substring to True when the
    sensor condition implicating that family holds."""
    w=[]
    for t in types:
        x=1.0
        if t[0]==current_scenario: x*=SEL_REPEAT_WT
        for frag,active in boosts:
            if active and frag in t[0]: x*=SEL_WEIGHT
        w.append(x)
    return random.choices(types,weights=w,k=1)[0]

def sensor_driven_next(lift, current_scenario, sensor_state, age_years):
    """
    Bias the next failure mode by the sensor state at the time of failure and by
    well age. Every mode stays reachable, so failure classes remain balanced for
    training while succession is still driven by sensor behaviour.

    NOTE: must be called BEFORE post-workover baselines are reset, or every
    threshold below is compared against healthy values and cannot trip.
    """
    if lift == "ESP":
        pip  = sensor_state.get('pip', 400)
        temp = sensor_state.get('temp', 190)
        vib  = sensor_state.get('vib', 0.3)
        disc = sensor_state.get('disc', 3000)
        return _pick(ESP_TYPES, current_scenario, [
            ("Emulsion",        pip>ESP_PIP_HIGH  and age_years>8),
            ("Underperformance",pip>ESP_PIP_HIGH  and age_years>8),
            ("Thermal",         temp>ESP_TEMP_HIGH and age_years>5),
            ("Gas Interference",vib>ESP_VIB_HIGH),
            ("Scale",           disc<ESP_DISC_LOW and age_years>10),
        ])

    elif lift == "Gas Lift":
        gap  = sensor_state.get('gap', 800)
        wht  = sensor_state.get('wht', 150)
        inj  = sensor_state.get('inj', 500)
        return _pick(GL_TYPES, current_scenario, [
            ("Tubing Hole", gap<GL_GAP_LOW and age_years>6),
            ("Paraffin",    wht<GL_WHT_LOW and age_years>4),
            ("Erosion",     inj>GL_INJ_HIGH),
        ])

    else:  # Pumping Unit
        fill    = sensor_state.get('fill', 80)
        cas     = sensor_state.get('cas', 150)
        runtime = sensor_state.get('runtime', 20)
        return _pick(PU_TYPES, current_scenario, [
            ("Tubing Leak",     cas>PU_CAS_HIGH and age_years>8),
            ("Underperformance",runtime>PU_RUNTIME_HIGH and age_years>6),
            ("Sensor Drift",    age_years>12),
            ("Rod Wear",        fill<PU_FILL_LOW and age_years>4),
            ("Rod Part",        fill<PU_FILL_LOW and age_years>9),
        ])

def stamp(row):
    """Attach row-level organization_id and timestamp to a history row.
    Applied at every append site so all asset types share one schema."""
    row["organization_id"]=ORGANIZATION_ID
    row["timestamp"]=row["date"]+" 00:00:00"
    return row

def hyp(qi,di,b,t):
    # math.exp, not np.exp: anything on this path feeds production volumes, which
    # feed invoice money. A numpy scalar there rounds half-cents differently from
    # Python and serialises to JSON silently. The b==0 branch is unreachable while
    # b is drawn from 0.7-1.5, but it must stay numpy-free in case b is retuned.
    if b==0: return qi*math.exp(-di*t)
    return qi/((1+b*di*t)**(1.0/b))

def cl(v,lo,hi): return max(lo,min(hi,v))
def rd(s,e): return s+timedelta(days=random.randint(0,(e-s).days))

# ── Correlated sensor noise ───────────────────────────────────────────────────
# Real daily SCADA is strongly persistent: today's motor temperature is mostly
# yesterday's motor temperature. Drawing an independent gaussian per day gives
# lag-1 autocorrelation ~0.00 and teaches a model white noise where the physical
# process is red. Each named channel carries its own AR(1) state:
#     e_t = rho*e_{t-1} + sqrt(1-rho^2)*N(0,1)
# which is stationary with unit variance, so value = v*(1 + p*e_t) keeps exactly
# the same marginal spread as the old nx() while restoring persistence.
AR_RHO_MIN, AR_RHO_MAX = 0.80, 0.93

class ARNoise:
    def __init__(self): self.rho=0.86; self.e={}
    def reset(self, rho=None):
        self.rho = random.uniform(AR_RHO_MIN,AR_RHO_MAX) if rho is None else rho
        self.e={}
    def dev(self, k):
        r=self.rho
        prev=self.e.get(k)
        if prev is None: prev=random.gauss(0,1)
        cur=r*prev+((1.0-r*r)**0.5)*random.gauss(0,1)
        self.e[k]=cur
        return cur

_AR=ARNoise()

def nx(v,p=0.025,k=None):
    """Multiplicative noise. Pass k to draw from that channel's AR(1) state
    (persistent, realistic); omit k for an independent draw."""
    if k is None: return v*(1+random.gauss(0,p))
    return v*(1+p*_AR.dev(k))

# ── Seasonality ───────────────────────────────────────────────────────────────
# South Texas ambient: ~52F January mean, ~85F July mean.
AMB_MEAN, AMB_AMP = 68.5, 17.0

def ambient_f(d):
    doy=d.timetuple().tm_yday
    return AMB_MEAN+AMB_AMP*math.sin(2*math.pi*(doy-110)/365.25)

def amb_delta(d):
    """Deviation from annual mean — added to ambient-influenced sensors."""
    return ambient_f(d)-AMB_MEAN

# ── Hard freeze events ────────────────────────────────────────────────────────
# An 18-year South Texas dataset with no freeze curtailment is conspicuous.
# Wells are curtailed (run_status CURTAILED), not shut in, so freeze days are
# never counted as failure/workover events and clean wells stay clean.
FREEZE_EVENTS = [
    (datetime(2011, 2, 1),  datetime(2011, 2, 5),  0.45),  # Feb 2011 ERCOT event
    (datetime(2018, 1, 16), datetime(2018, 1, 18), 0.70),  # brief hard freeze
    (datetime(2021, 2, 11), datetime(2021, 2, 20), 0.08),  # Winter Storm Uri
    (datetime(2022,12, 22), datetime(2022,12, 26), 0.55),  # Winter Storm Elliott
]

def freeze_state(d):
    """Returns (production_multiplier, ambient_override_f) or (1.0, None)."""
    for s,e,sev in FREEZE_EVENTS:
        if s<=d<=e:
            span=max(1,(e-s).days)
            # coldest in the middle of the event
            phase=abs((d-s).days-span/2.0)/(span/2.0+1e-9)
            depth=1.0-0.65*phase
            mult=sev+(1.0-sev)*(1.0-depth)
            return cl(mult,0.02,1.0), 20.0+25.0*phase
    return 1.0, None

# ── Pump curves for ESP ───────────────────────────────────────────────────────
PUMP_MODELS = [
    {"model":"Centrilift GN5600","manufacturer":"Centrilift","bep_bpd":4000,
     "min_bpd":2000,"max_bpd":6000,"stages":120,
     "poly":{"a":-0.000012,"b":0.02,"c":45.0}},
    {"model":"REDA D3000N","manufacturer":"REDA","bep_bpd":2800,
     "min_bpd":1200,"max_bpd":4000,"stages":100,
     "poly":{"a":-0.000018,"b":0.025,"c":38.0}},
    {"model":"Centrilift FC1200","manufacturer":"Centrilift","bep_bpd":1200,
     "min_bpd":800,"max_bpd":1600,"stages":150,
     "poly":{"a":-0.000045,"b":0.035,"c":28.0}},
]
BPD_TO_GPM = 42.0/1440.0

def make_pump_curve(model_def):
    m=model_def; a=m['poly']['a']; b=m['poly']['b']; c=m['poly']['c']
    stages=m['stages']
    flows=[m['min_bpd'],m['bep_bpd']*0.7,m['bep_bpd'],m['bep_bpd']*1.2,m['max_bpd']]
    curve=[]
    for q in flows:
        h=a*q**2+b*q+c; h=max(5,h)
        eff=85*(1-((q-m['bep_bpd'])/(m['bep_bpd']*0.8))**2)
        eff=cl(eff,20,90)
        q_gpm=q*BPD_TO_GPM
        bhp=round((q_gpm*h)/(3960*(eff/100)),3)
        curve.append({"flow_rate_bpd":round(q,0),"head_ft_per_stage":round(h,2),
                      "efficiency_percent":round(eff,1),"brake_horsepower":bhp})
    return {
        "manufacturer":m['manufacturer'],"model":m['model'],
        "stages":stages,"baseline_frequency_hz":60,
        "min_operating_flow_bpd":m['min_bpd'],"max_operating_flow_bpd":m['max_bpd'],
        "best_efficiency_point_bpd":m['bep_bpd'],
        "polynomial_coefficients":{"head_curve":m['poly']},
        "curve_data_60hz":curve,
    }

def assign_pump_model(qi):
    if qi > 1200: return PUMP_MODELS[0]
    elif qi > 600: return PUMP_MODELS[1]
    else: return PUMP_MODELS[2]

# ── Concurrent failure modes ──────────────────────────────────────────────────
# Real wells do not fail one tidy mode at a time. A scaling pump runs hotter, a
# rod string wearing against tubing eventually parts, paraffin and valve erosion
# progress together. Each well therefore carries a PRIMARY degrading mode and,
# on P_OVERLAP of episodes, a CONCURRENT secondary mode that starts partway
# through and degrades alongside it. Both write to the same sensors on the same
# day; their effects are summed as deltas off the healthy baseline, then noise
# is applied once, so an overlapping episode is genuinely superposed rather than
# one mode overwriting the other.
P_OVERLAP   = 0.32   # probability an episode grows a second, concurrent mode
CO_START    = (0.25, 0.70)  # where in the primary's window the second one begins


def _co_pick(types, primary):
    """Choose a concurrent mode from a different family than the primary."""
    fam = primary.split(" - ")[0]
    alt = [t for t in types if not t[0].startswith(fam)]
    return random.choice(alt or types)


# ── ESP history ───────────────────────────────────────────────────────────────
def esp_deltas(scenario, frac, base):
    """Additive sensor deltas off baseline, plus an oil multiplier, for one mode."""
    pb, db, ab, tb, hb, vb = base
    d = {'pip': 0.0, 'disc': 0.0, 'amps': 0.0, 'temp': 0.0, 'hz': 0.0, 'vib': 0.0}
    om = 1.0
    if "Gas Interference" in scenario:
        sev = "Severe" in scenario
        d['pip'] += frac * random.uniform(300, 800 if sev else 450)
        d['amps'] += random.gauss(0, frac * (15 if sev else 7))
        d['temp'] += frac * random.uniform(25, 90 if sev else 40)
        d['hz'] += random.gauss(0, frac * (4 if sev else 2))
        d['vib'] += frac * random.uniform(0.3, 2.5 if sev else 1.0)
        om *= (1 - frac * 0.10)
    elif "Thermal Burnout" in scenario:
        d['temp'] += frac * random.uniform(60, 130 if "Rapid" in scenario else 90)
        d['amps'] += random.gauss(0, frac * 25)
        d['disc'] -= frac * random.uniform(300, 1200)
        om *= (1 - frac * 0.08)
    elif "Scale Buildup" in scenario:
        d['disc'] -= frac * random.uniform(400, 1800 if "Slow" in scenario else 900)
        d['amps'] += frac * random.uniform(5, 18)
        om *= (1 - frac * 0.12)
    elif "Emulsion Block" in scenario:
        d['amps'] += frac * random.uniform(20, 60 if "Acute" in scenario else 35)
        d['temp'] += frac * random.uniform(15, 50)
        om *= (1 - frac * 0.14)
    elif "Underperformance" in scenario:
        d['pip'] += max(0.0, 1000.0 - pb) * frac + frac * 500
        if "Declining" in scenario:
            d['disc'] -= frac * random.uniform(200, 600)
        om *= (1 - frac * 0.06)
    return d, om


ESP_DOWN = {"Gas Interference": (5, 21), "Thermal Burnout": (8, 25),
            "Scale Buildup": (6, 15), "Emulsion Block": (7, 20)}


def _esp_down(sc):
    for k, v in ESP_DOWN.items():
        if k in sc:
            return random.randint(*v)
    return random.randint(6, 18)


def gen_esp(aid, spud, is_clean):
    # Vintage-calibrated. These wells spud 2008-2011: early Eagle Ford, short
    # laterals and small fracs. An early Eagle Ford well made roughly 300-700
    # bopd IP30, not the 1,200-2,000 of a 2014+ completion.
    qi=random.uniform(120,820); di=random.uniform(0.0015,0.006); b=random.uniform(0.7,1.5)
    wc_=random.uniform(0.25,0.75); gor=random.uniform(400,3000)
    pb=random.uniform(200,600); db=random.uniform(1800,6000)
    ab=random.uniform(45,180); tb=random.uniform(170,215)
    hb=random.uniform(48,63); vb=random.uniform(0.05,0.6)
    ftp0=random.uniform(100,800); acp0=random.uniform(80,500)

    pump_model = assign_pump_model(qi)
    pump_curve = make_pump_curve(pump_model)

    sec_floor=random.uniform(0.50,0.62)
    sec_t=random.uniform(2800,4600); sec_w=random.uniform(250,600)

    _AR.reset()
    history=[]; shutin_events=[]; current=spud; t=0
    status="PRODUCING"; downtime=0
    routine=routine_outages(spud,END_DATE)
    scenario=random.choice(ESP_TYPES)[0]
    wmin,wmax=next((s[1],s[2]) for s in ESP_TYPES if s[0]==scenario)
    next_fail=random.randint(120,300); fwin=random.randint(wmin,max(wmin,min(wmax,180)))
    detr=0; done=False
    co_sc=None; co_win=0; co_detr=0; co_at=0
    sensor_state={'pip':pb,'temp':tb,'amps':ab,'vib':vb,'disc':db}

    while current<=END_DATE:
        _ds=current.strftime("%Y-%m-%d")
        # Ordinary non-failure downtime. Not a failure event, no workover invoice.
        if status=="PRODUCING" and _ds in routine:
            status="SHUT-IN"; downtime=routine[_ds]
        age_years=t/365.0
        sec=sec_floor+(1.0-sec_floor)/(1.0+math.exp(cl((t-sec_t)/sec_w,-60.0,60.0)))
        fmult,famb=freeze_state(current); ad=amb_delta(current)
        oil=hyp(qi,di,b,t)*sec*fmult
        water=oil*(wc_/(1-wc_))*(1+t/10000); gas=oil*gor/1000

        dd={'pip':0.0,'disc':0.0,'amps':0.0,'temp':0.0,'hz':0.0,'vib':0.0}

        if status=="PRODUCING" and not is_clean:
            if not done and t>=next_fail:
                detr+=1; frac=min(1.0,detr/fwin)
                is_under="Underperformance" in scenario
                d1,om=esp_deltas(scenario,frac,(pb,db,ab,tb,hb,vb))
                for k in dd: dd[k]+=d1[k]
                oil*=om

                # concurrent mode: starts partway in, degrades alongside
                if co_sc is None and co_at and detr>=co_at:
                    pick=_co_pick(ESP_TYPES,scenario)
                    co_sc=pick[0]; co_win=random.randint(pick[1],max(pick[1],min(pick[2],200)))
                    co_detr=0
                if co_sc:
                    co_detr+=1; cf=min(1.0,co_detr/co_win)
                    d2,om2=esp_deltas(co_sc,cf,(pb,db,ab,tb,hb,vb))
                    for k in dd: dd[k]+=d2[k]
                    oil*=om2

                trip = (detr>=fwin and not is_under) or (co_sc is not None and co_detr>=co_win)
                if trip:
                    active=[scenario]+([co_sc] if co_sc else [])
                    dt_days=_esp_down(scenario)
                    if co_sc: dt_days=int(dt_days*random.uniform(1.25,1.6))
                    status="SHUT-IN"; downtime=dt_days
                    shutin_events.append({"date":current.strftime("%Y-%m-%d"),
                                          "scenario":scenario,"concurrent":active,
                                          "downtime_days":downtime})
                    done=True
                elif detr>=fwin and is_under:
                    failing=[s for s in ESP_TYPES if "Underperformance" not in s[0]]
                    nsc=random.choice(failing)
                    scenario=nsc[0]; fwin=random.randint(nsc[1],nsc[2])
                    detr=0; next_fail=t+30
                    co_sc=None; co_at=int(fwin*random.uniform(*CO_START)) if random.random()<P_OVERLAP else 0

        # The trip above may have just flipped status to SHUT-IN. Re-test it here
        # rather than writing the producing row from inside the block that
        # changed it, or the failure day emits run_status SHUT-IN alongside live
        # production, sensors and frequency (and the outage runs one day longer
        # than downtime_days). GL and PU already re-test; this makes ESP match.
        # Convention: the trip day is the FIRST ZERO day.
        if status=="PRODUCING":
            pip=nx(pb+dd['pip'],k='pip');   disc=nx(db+dd['disc'],k='disc')
            amps=nx(ab+dd['amps'],k='amps');temp=nx(tb+dd['temp'],k='temp')+0.25*ad
            hz=nx(hb+dd['hz'],0.01,k='hz'); vib=nx(vb+dd['vib'],k='vib')
            ftp=nx(ftp0,k='ftp');           acp=nx(acp0,k='acp')
            if famb is not None: temp-=0.5*(AMB_MEAN-famb)*0.15
            pip=cl(pip,10,8000); disc=cl(disc,100,10000); amps=cl(amps,5,280); temp=cl(temp,100,295)

            row={
                "date":current.strftime("%Y-%m-%d"),
                "run_status":("CURTAILED" if fmult<0.9 else status),
                "gross_oil_production_bbl":round(max(0,nx(oil,k='oil')),2),
                "well_test_rate_bopd":round(max(0,nx(oil,0.06,k='wt')),2),
                "flowing_tubing_pressure_psi":round(cl(ftp,30,2000),1),
                "annular_casing_pressure_psi":round(cl(acp,20,1500),1),
                "motor_winding_temp_f":round(cl(temp,100,380),1),
                "motor_load_amps":round(cl(amps,5,280),2),
                "vfd_output_frequency_hz":round(cl(hz,30,75),1),
                "pump_intake_pressure_psi":round(cl(pip,10,8000),1),
                "pump_discharge_pressure_psi":round(cl(disc,100,9500),1),
                "lateral_vibration_hz":round(cl(vib,0,5),3),
                "produced_water_volume_bbl":round(max(0,nx(water,k='wat')),2),
                "casinghead_gas_mcf":round(max(0,nx(gas,k='gas')),3),
            }
            sensor_state={'pip':pip,'temp':temp,'amps':amps,'vib':vib,'disc':disc}

        elif status=="SHUT-IN":
            downtime-=1
            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":"SHUT-IN",
                "gross_oil_production_bbl":0.0,"well_test_rate_bopd":0.0,
                "flowing_tubing_pressure_psi":0.0,
                "annular_casing_pressure_psi":round(cl(nx(acp0,0.05),20,800),1),
                "motor_winding_temp_f":round(cl(nx(95,0.03,k='tidle')+0.6*ad,60,200),1),
                "motor_load_amps":0.0,"vfd_output_frequency_hz":0.0,
                "pump_intake_pressure_psi":0.0,"pump_discharge_pressure_psi":0.0,
                "lateral_vibration_hz":0.0,
                "produced_water_volume_bbl":0.0,"casinghead_gas_mcf":0.0,
            }
            if downtime<=0:
                status="PRODUCING"; detr=0; done=False
                # Select from the sensor state as it stood at failure (still
                # degraded) — before baselines are reset, or nothing can trip.
                nsc=sensor_driven_next("ESP",scenario,sensor_state,t/365.0)
                scenario=nsc[0]; fwin=random.randint(nsc[1],nsc[2])
                next_fail=t+random.randint(60,250)
                co_sc=None; co_detr=0
                co_at=int(fwin*random.uniform(*CO_START)) if random.random()<P_OVERLAP else 0
                pb=random.uniform(200,600); db=random.uniform(1800,5500)
                ab=random.uniform(45,160); tb=random.uniform(170,212); hb=random.uniform(48,63)
                sensor_state={'pip':pb,'temp':tb,'amps':ab,'vib':vb,'disc':db}

        history.append(stamp(row)); current+=timedelta(days=1); t+=1

    return history, pump_curve, shutin_events


# ── GL history ────────────────────────────────────────────────────────────────
def gl_deltas(scenario, frac, base):
    ib_, cb_, ipb_, tub0, wht0 = base
    d = {'inj': 0.0, 'cas': 0.0, 'injp': 0.0, 'wht': 0.0}
    om = 1.0
    if "Paraffin" in scenario:
        corr = "Corrosion" in scenario
        d['wht'] -= frac * random.uniform(15, 45 * (1.5 if corr else 1.0))
        if corr:
            d['cas'] += frac * random.uniform(20, 80)
        om *= (1 - frac * 0.25)
    elif "Erosion" in scenario:
        sp = 1.8 if "Sand" in scenario else 1.0
        d['inj'] += frac * random.uniform(50, 300) * sp
        d['injp'] -= frac * random.uniform(50, 200)
        om *= (1 - frac * 0.15)
    elif "Tubing Hole" in scenario:
        large = "Large" in scenario
        gl_ = frac * (cb_ - tub0) * (0.88 if large else 0.55)
        d['cas'] -= gl_
        d['injp'] -= gl_ * 0.60
        d['inj'] -= ib_ * frac * 0.04
        d['wht'] -= frac * 10
        om *= (1 - frac * (0.40 if large else 0.20))
    return d, om


def gen_gl(aid, spud, is_clean):
    # Vintage-calibrated for 2008-2011 spuds — see the note in gen_esp.
    qi=random.uniform(60,520); di=random.uniform(0.001,0.005); b=random.uniform(0.7,1.4)
    wc_=random.uniform(0.30,0.75); gor=random.uniform(600,4000)
    ib=random.uniform(150,1000); cb=random.uniform(700,1800)
    tub0=random.uniform(150,500); wht0=random.uniform(110,190)
    chk0=random.uniform(35,80); ipb=random.uniform(850,2000)

    sec_floor=random.uniform(0.40,0.54)
    sec_t=random.uniform(2800,4600); sec_w=random.uniform(250,600)

    _AR.reset()
    history=[]; shutin_events=[]; current=spud; t=0
    status="PRODUCING"; downtime=0
    routine=routine_outages(spud,END_DATE)
    scenario=random.choice(GL_TYPES)[0]
    wmin,wmax=next((s[1],s[2]) for s in GL_TYPES if s[0]==scenario)
    next_fail=random.randint(120,280); fwin=random.randint(wmin,max(wmin,min(wmax,150)))
    detr=0; done=False
    co_sc=None; co_win=0; co_detr=0; co_at=0
    ib_=ib; cb_=cb; ipb_=ipb
    sensor_state={'gap':cb-tub0,'wht':wht0,'inj':ib}

    while current<=END_DATE:
        _ds=current.strftime("%Y-%m-%d")
        # Ordinary non-failure downtime. Not a failure event, no workover invoice.
        if status=="PRODUCING" and _ds in routine:
            status="SHUT-IN"; downtime=routine[_ds]
        sec=sec_floor+(1.0-sec_floor)/(1.0+math.exp(cl((t-sec_t)/sec_w,-60.0,60.0)))
        fmult,famb=freeze_state(current); ad=amb_delta(current)
        oil=hyp(qi,di,b,t)*sec*fmult
        water=oil*(wc_/(1-wc_))*(1+t/9000); gas=oil*gor/1000
        dd={'inj':0.0,'cas':0.0,'injp':0.0,'wht':0.0}

        if status=="PRODUCING" and not is_clean:
            if not done and t>=next_fail:
                detr+=1; frac=min(1.0,detr/fwin)
                d1,om=gl_deltas(scenario,frac,(ib_,cb_,ipb_,tub0,wht0))
                for k in dd: dd[k]+=d1[k]
                oil*=om
                if co_sc is None and co_at and detr>=co_at:
                    pick=_co_pick(GL_TYPES,scenario)
                    co_sc=pick[0]; co_win=random.randint(pick[1],max(pick[1],min(pick[2],180)))
                    co_detr=0
                if co_sc:
                    co_detr+=1; cf=min(1.0,co_detr/co_win)
                    d2,om2=gl_deltas(co_sc,cf,(ib_,cb_,ipb_,tub0,wht0))
                    for k in dd: dd[k]+=d2[k]
                    oil*=om2
                if detr>=fwin or (co_sc is not None and co_detr>=co_win):
                    active=[scenario]+([co_sc] if co_sc else [])
                    dt_days=random.randint(5,18)
                    if co_sc: dt_days=int(dt_days*random.uniform(1.25,1.6))
                    status="SHUT-IN"; downtime=dt_days
                    shutin_events.append({"date":current.strftime("%Y-%m-%d"),
                                          "scenario":scenario,"concurrent":active,
                                          "downtime_days":downtime}); done=True

        if status=="PRODUCING":
            chk=nx(chk0,0.02,k='chk')
            inj=nx(ib_+dd['inj'],k='inj');  cas=nx(cb_+dd['cas'],k='cas')
            tub=nx(tub0,k='tub');           injp=nx(ipb_+dd['injp'],k='injp')
            wht=nx(wht0+dd['wht'],k='wht')+0.45*ad
            if famb is not None: wht-=(AMB_MEAN-famb)*0.35
            inj=cl(inj,0,15000); cas=cl(cas,0,3000); injp=cl(injp,0,3000); wht=cl(wht,30,350)
            gap=cl(cas-tub,0,3000)
            gor_val=round(gas*1000/max(oil,1.0),0) if oil>1.0 else 0.0
            sensor_state={'gap':gap,'wht':wht,'inj':inj}
            row={
                "date":current.strftime("%Y-%m-%d"),
                "run_status":("CURTAILED" if fmult<0.9 else "PRODUCING"),
                "gross_oil_production_bbl":round(max(0,nx(oil,k='oil')),2),
                "well_test_rate_bopd":round(max(0,nx(oil,0.06,k='wt')),2),
                "annular_casing_pressure_psi":round(cl(cas,0,3000),1),
                "lift_gas_injection_pressure_psi":round(cl(injp,0,3000),1),
                "surface_tubing_pressure_psi":round(cl(tub,0,1500),1),
                "lift_gas_rate_mscfd":round(cl(inj,0,15000),1),
                "wellhead_flowing_temp_f":round(cl(wht,30,350),1),
                "production_choke_opening_pct":round(cl(chk,0,100),1),
                "compressor_inlet_temp_f":round(cl(nx(95,0.05,k='cit')+0.5*ad,40,200),1),
                "produced_water_volume_bbl":round(max(0,nx(water,k='wat')),2),
                "casinghead_gas_mcf":round(max(0,nx(gas,k='gas')),3),
                "casing_tubing_pressure_gap_psi":round(gap,1),
                "gas_oil_ratio_scf_bbl":gor_val,
            }
        else:
            downtime-=1
            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":"SHUT-IN",
                "gross_oil_production_bbl":0.0,"well_test_rate_bopd":0.0,
                "annular_casing_pressure_psi":round(cl(nx(cb_*0.25,k='cas'),0,3000),1),
                "lift_gas_injection_pressure_psi":0.0,
                "surface_tubing_pressure_psi":round(cl(nx(tub0*0.6,k='tub'),0,1500),1),
                "lift_gas_rate_mscfd":0.0,
                "wellhead_flowing_temp_f":round(cl(nx(75,0.05,k='whtidle')+0.6*ad,30,200),1),
                "production_choke_opening_pct":0.0,
                "compressor_inlet_temp_f":round(cl(nx(65,0.05,k='cit')+0.5*ad,40,150),1),
                "produced_water_volume_bbl":0.0,"casinghead_gas_mcf":0.0,
                "casing_tubing_pressure_gap_psi":0.0,"gas_oil_ratio_scf_bbl":0.0,
            }
            if downtime<=0:
                status="PRODUCING"; detr=0; done=False
                cb_=random.uniform(700,1800); ib_=random.uniform(150,1000); ipb_=random.uniform(850,2000)
                nsc=sensor_driven_next("Gas Lift",scenario,sensor_state,t/365.0)
                scenario=nsc[0]; fwin=random.randint(nsc[1],nsc[2])
                next_fail=t+random.randint(60,250)
                co_sc=None; co_detr=0
                co_at=int(fwin*random.uniform(*CO_START)) if random.random()<P_OVERLAP else 0

        history.append(stamp(row)); current+=timedelta(days=1); t+=1

    return history, shutin_events


# ── PU history ────────────────────────────────────────────────────────────────
def pu_deltas(scenario, frac, base):
    fb, pb, cb, rb, stroke0, spm0, amps0, temp0, tub0 = base
    d = {'fill': 0.0, 'pip': 0.0, 'cas': 0.0, 'runtime': 0.0,
         'stroke': 0.0, 'spm': 0.0, 'amps': 0.0, 'temp': 0.0, 'tub': 0.0}
    om = 1.0
    if "Rod Part" in scenario:
        # Rod string parts: the plunger stops carrying fluid. Load comes off the
        # motor (amps collapse), pump fillage goes to nothing, fluid builds over
        # the pump so intake pressure climbs, and production stops even though
        # the surface unit keeps stroking.
        rapid = "Fatigue" in scenario
        sh = frac ** (0.6 if rapid else 1.0)
        d['fill'] -= sh * fb * random.uniform(0.80, 0.95)
        d['amps'] -= sh * amps0 * random.uniform(0.45, 0.68)
        d['pip'] += sh * random.uniform(300, 900)
        d['runtime'] += sh * random.uniform(0.5, 2.0)
        d['tub'] -= sh * tub0 * random.uniform(0.30, 0.55)
        om *= (1 - sh * random.uniform(0.80, 0.93))
    elif "Rod Wear" in scenario:
        # Coupling/tubing wear before a part: erratic load, slow fillage loss.
        d['amps'] += random.gauss(0, frac * 6.0) - frac * amps0 * 0.05
        d['fill'] -= frac * random.uniform(5, 16)
        d['cas'] += frac * random.uniform(5, 30)
        d['temp'] += frac * random.uniform(3, 14)
        om *= (1 - frac * 0.14)
    elif "Sensor Drift" in scenario:
        d['pip'] += frac * random.uniform(200, 800)
        if "Multi" in scenario:
            d['amps'] += frac * random.uniform(5, 20)
    elif "Tubing Leak" in scenario:
        sp = 2.0 if "Section" in scenario else 1.0
        d['cas'] += frac * random.uniform(50, 250) * sp
        om *= (1 - frac * (0.35 if "Section" in scenario else 0.20))
    elif "Underperformance" in scenario:
        d['runtime'] += frac * random.uniform(0.5, 2.5)
        d['fill'] -= frac * random.uniform(4, 12)
        om *= (1 - frac * 0.18)
    return d, om


def gen_pu(aid, spud, is_clean):
    # Trimmed at the top: a beam pump lifting much past ~450 bopd of oil is
    # outside normal capacity even at high fluid volumes.
    qi=random.uniform(20,210); di=random.uniform(0.001,0.005); b=random.uniform(0.7,1.4)
    wc_=random.uniform(0.40,0.85); gor=random.uniform(150,900)
    spm0=random.uniform(3,14); stroke0=random.uniform(42,108)
    fill0=random.uniform(65,95); pip0=random.uniform(150,1100)
    rt0=random.uniform(16,23); cas0=random.uniform(60,350)
    tub0=random.uniform(30,150); amps0=random.uniform(10,55); temp0=random.uniform(120,195)

    sec_floor=random.uniform(0.35,0.48)
    sec_t=random.uniform(3500,5300); sec_w=random.uniform(300,700)

    _AR.reset()
    history=[]; shutin_events=[]; current=spud; t=0
    status="PRODUCING"; downtime=0
    routine=routine_outages(spud,END_DATE)
    scenario=random.choice(PU_TYPES)[0]
    wmin,wmax=next((s[1],s[2]) for s in PU_TYPES if s[0]==scenario)
    next_fail=random.randint(100,280); fwin=random.randint(wmin,max(wmin,min(wmax,90)))
    detr=0; done=False
    co_sc=None; co_win=0; co_detr=0; co_at=0
    fb=fill0; pb=pip0; cb=cas0; rb=rt0
    sensor_state={'pip':pb,'cas':cb,'runtime':rb,'fill':fb}

    while current<=END_DATE:
        _ds=current.strftime("%Y-%m-%d")
        # Ordinary non-failure downtime. Not a failure event, no workover invoice.
        if status=="PRODUCING" and _ds in routine:
            status="SHUT-IN"; downtime=routine[_ds]
        sec=sec_floor+(1.0-sec_floor)/(1.0+math.exp(cl((t-sec_t)/sec_w,-60.0,60.0)))
        fmult,famb=freeze_state(current); ad=amb_delta(current)
        oil=hyp(qi,di,b,t)*sec*fmult
        water=oil*(wc_/(1-wc_))*(1+t/8000); gas=oil*gor/1000
        dd={'fill':0.0,'pip':0.0,'cas':0.0,'runtime':0.0,'stroke':0.0,
            'spm':0.0,'amps':0.0,'temp':0.0,'tub':0.0}
        base=(fb,pb,cb,rb,stroke0,spm0,amps0,temp0,tub0)

        if status=="PRODUCING" and not is_clean:
            if not done and t>=next_fail:
                detr+=1; frac=min(1.0,detr/fwin)
                is_under="Underperformance" in scenario
                d1,om=pu_deltas(scenario,frac,base)
                for k in dd: dd[k]+=d1[k]
                oil*=om
                if co_sc is None and co_at and detr>=co_at:
                    pick=_co_pick(PU_TYPES,scenario)
                    co_sc=pick[0]; co_win=random.randint(pick[1],max(pick[1],min(pick[2],200)))
                    co_detr=0
                if co_sc:
                    co_detr+=1; cf=min(1.0,co_detr/co_win)
                    d2,om2=pu_deltas(co_sc,cf,base)
                    for k in dd: dd[k]+=d2[k]
                    oil*=om2
                trip=(detr>=fwin and not is_under) or (co_sc is not None and co_detr>=co_win)
                if trip:
                    active=[scenario]+([co_sc] if co_sc else [])
                    dt_days=random.randint(7,18) if "Rod" in scenario else random.randint(4,12)
                    if co_sc: dt_days=int(dt_days*random.uniform(1.25,1.6))
                    status="SHUT-IN"; downtime=dt_days
                    shutin_events.append({"date":current.strftime("%Y-%m-%d"),
                                          "scenario":scenario,"concurrent":active,
                                          "downtime_days":downtime}); done=True
                elif detr>=fwin and is_under:
                    failing=[s for s in PU_TYPES if "Underperformance" not in s[0]]
                    nsc=random.choice(failing)
                    scenario=nsc[0]; fwin=random.randint(nsc[1],nsc[2])
                    detr=0; next_fail=t+30
                    co_sc=None; co_at=int(fwin*random.uniform(*CO_START)) if random.random()<P_OVERLAP else 0

        if status=="PRODUCING":
            spm=nx(spm0+dd['spm'],k='spm');   stroke=nx(stroke0+dd['stroke'],k='stroke')
            fill=nx(fb+dd['fill'],k='fill');  pip=nx(pb+dd['pip'],k='pip')
            runtime=nx(rb+dd['runtime'],0.02,k='rt'); cas=nx(cb+dd['cas'],k='cas')
            tub=nx(tub0+dd['tub'],k='tub');   amps=nx(amps0+dd['amps'],k='amps')
            temp=nx(temp0+dd['temp'],k='temp')+0.35*ad
            if famb is not None: temp-=(AMB_MEAN-famb)*0.25
            fill=cl(fill,0,100); amps=cl(amps,0,80); pip=cl(pip,0,5000)
            cas=cl(cas,0,1800); tub=cl(tub,0,800); runtime=cl(runtime,0,24)
            sensor_state={'pip':pip,'cas':cas,'runtime':runtime,'fill':fill}
            row={
                "date":current.strftime("%Y-%m-%d"),
                "run_status":("CURTAILED" if fmult<0.9 else "PRODUCING"),
                "gross_oil_production_bbl":round(max(0,nx(oil,k='oil')),2),
                "well_test_rate_bopd":round(max(0,nx(oil,0.06,k='wt')),2),
                "polished_rod_stroke_length_in":round(cl(stroke,0,180),1),
                "effective_plunger_stroke_in":round(cl(stroke*fill/100,0,180),1),
                "pumping_speed_spm":round(cl(spm,0,20),2),
                "volumetric_efficiency_pct":round(fill,1),
                "annular_casing_pressure_psi":round(cas,1),
                "wellhead_tubing_pressure_psi":round(tub,1),
                "surface_motor_amperage":round(amps,2),
                "motor_drive_frequency_hz":round(cl(nx(60,0.01,k='hz'),30,75),1),
                "downhole_pump_intake_psi":round(pip,1),
                "gearbox_motor_temp_f":round(cl(temp,60,300),1),
                "produced_water_volume_bbl":round(max(0,nx(water,k='wat')),2),
                "casinghead_gas_mcf":round(max(0,nx(gas,k='gas')),3),
                "daily_runtime_hrs":round(runtime,1),
            }
        else:
            downtime-=1
            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":"SHUT-IN",
                "gross_oil_production_bbl":0.0,"well_test_rate_bopd":0.0,
                "polished_rod_stroke_length_in":0.0,"effective_plunger_stroke_in":0.0,
                "pumping_speed_spm":0.0,"volumetric_efficiency_pct":0.0,
                "annular_casing_pressure_psi":round(cl(nx(cas0,0.05,k='cas'),0,1800),1),
                "wellhead_tubing_pressure_psi":round(cl(nx(tub0,0.05,k='tub'),0,800),1),
                "surface_motor_amperage":0.0,"motor_drive_frequency_hz":0.0,
                "downhole_pump_intake_psi":0.0,
                "gearbox_motor_temp_f":round(cl(nx(95,0.03,k='tidle')+0.6*ad,60,200),1),
                "produced_water_volume_bbl":0.0,"casinghead_gas_mcf":0.0,
                "daily_runtime_hrs":0.0,
            }
            if downtime<=0:
                status="PRODUCING"; detr=0; done=False
                fb=random.uniform(62,92); pb=random.uniform(150,900)
                cb=random.uniform(60,350); rb=random.uniform(16,23)
                nsc=sensor_driven_next("Pumping Unit",scenario,sensor_state,t/365.0)
                scenario=nsc[0]; fwin=random.randint(nsc[1],nsc[2])
                next_fail=t+random.randint(60,250)
                co_sc=None; co_detr=0
                co_at=int(fwin*random.uniform(*CO_START)) if random.random()<P_OVERLAP else 0
                sensor_state={'pip':pb,'cas':cb,'runtime':rb,'fill':fb}

        history.append(stamp(row)); current+=timedelta(days=1); t+=1

    return history, shutin_events

# ── Invoice generation ────────────────────────────────────────────────────────
# Costs were previously uniform random and statistically independent of the
# well (LOE corr with monthly oil = -0.03; a sensor-drift job and a large
# tubing hole drew from the same 25k-120k range). Cost is now driven by the
# things that actually drive it: fluid volumes handled, well age, lift type,
# job scope, downtime length, and the year's cost index.

# Service-cost index, 2008=1.00. Captures the 2009 and 2015-16 service-price
# collapses and the 2021-23 inflation run-up.
COST_INDEX = {
    2008:1.00,2009:0.78,2010:0.85,2011:0.95,2012:1.02,2013:1.05,2014:1.12,
    2015:0.82,2016:0.74,2017:0.86,2018:0.95,2019:0.97,2020:0.71,2021:0.88,
    2022:1.14,2023:1.27,2024:1.31,2025:1.35,2026:1.38,
}
def cost_index(y): return COST_INDEX.get(y,1.38)

# Fixed monthly overhead and per-barrel handling cost by lift type.
LIFT_ECON = {
    "ESP":         {"fixed":6200.0,"oil_bbl":3.10,"wtr_bbl":1.05,"power":0.85},
    "Gas Lift":    {"fixed":4800.0,"oil_bbl":2.55,"wtr_bbl":0.95,"power":0.42},
    "Pumping Unit":{"fixed":3900.0,"oil_bbl":2.20,"wtr_bbl":0.88,"power":0.55},
}

# Workover scope by failure family: (base_usd, per_downtime_day_usd).
# Rig-and-pull ESP jobs dominate; surface/sensor work is an order cheaper.
WO_SCOPE = [
    ("Thermal Burnout",  148000.0, 5200.0),   # pull tubing, new motor+pump
    ("Scale Buildup",     72000.0, 3100.0),   # acid/scale squeeze, may pull
    ("Emulsion Block",    58000.0, 2700.0),   # chemical treat + flush
    ("Gas Interference",  44000.0, 2400.0),   # gas handling / setpoint + pull
    ("Underperformance",  39000.0, 2100.0),
    ("Tubing Hole",       86000.0, 3600.0),   # tubing string replacement
    ("Tubing Leak",       81000.0, 3500.0),
    ("Injection Valve",   47000.0, 2200.0),   # gas lift valve change-out
    ("Paraffin",          26000.0, 1500.0),   # hot oil / cutting
    ("Rod Part",          64000.0, 2900.0),   # fish and replace rod string
    ("Rod Wear",          38000.0, 1900.0),   # rod/coupling change-out
    ("Downhole Sensor",   14000.0,  900.0),   # instrument swap
]
def wo_scope(scenario):
    for frag,base,per in WO_SCOPE:
        if frag in scenario: return base,per
    return 45000.0,2200.0

# Which vendors are actually capable of each workover scope. Previously every
# workover drew from one flat 3-name list regardless of the job, so a downhole
# instrument swap and a rod-string fishing job could bill to the same roustabout
# crew. Names are drawn from the VENDORS registry below-of-record; the build
# asserts every vendor written to an invoice appears in VENDORS.
# Note the three VENDORS not listed here are deliberate: CPS Energy is the power
# utility, Eagle Ford Water Services hauls water, and the Internal crews carry
# routine LOE — none of them run workovers.
WO_VENDOR_BY_SCOPE = [
    ("Thermal Burnout",  ["Frontier Well Services","South Texas Electrical LLC"]),
    ("Scale Buildup",    ["Frontier Well Services","Monarch Chemical Services"]),
    ("Emulsion Block",   ["Monarch Chemical Services","South Texas Hot Oil"]),
    ("Gas Interference", ["Frontier Well Services","South Texas Compression LLC"]),
    ("Underperformance", ["Frontier Well Services","Laredo Roustabout Services"]),
    ("Tubing Hole",      ["Frontier Well Services","Laredo Roustabout Services"]),
    ("Tubing Leak",      ["Frontier Well Services","Laredo Roustabout Services"]),
    ("Injection Valve",  ["Frontier Well Services","South Texas Compression LLC"]),
    ("Paraffin",         ["South Texas Hot Oil","Monarch Chemical Services"]),
    ("Rod Part",         ["Frontier Well Services","Laredo Roustabout Services"]),
    ("Rod Wear",         ["Frontier Well Services","Laredo Roustabout Services"]),
    ("Downhole Sensor",  ["Lone Star Telemetry","South Texas Electrical LLC"]),
]
def wo_vendors(scenario):
    for frag,vs in WO_VENDOR_BY_SCOPE:
        if frag in scenario: return vs
    return WO_VENDORS

# Scheduled (non-failure) well service, by lift. Every well gets this, including
# the clean cohort. Without it the 10 clean wells carry ZERO Workover invoices
# for their whole life while every other well carries 5-17, so the financial
# ledger alone identifies the healthy cohort with perfect accuracy — the same
# leak as 100% uptime, just moved from run_status into the invoices.
# These are real preventive jobs, priced well below a failure repair, and they
# reveal nothing about a failure mode because they are lift-appropriate routine
# maintenance rather than a repair of anything.
PREVENTIVE_SCOPES = {
    "ESP": [("Downhole gauge and cable inspection",        9000.0, 26000.0,
             ["Frontier Well Services","South Texas Electrical LLC"]),
            ("Chemical inhibitor squeeze — scale program", 7000.0, 21000.0,
             ["Monarch Chemical Services"])],
    "Gas Lift": [("Gas lift survey and valve resetting",   8000.0, 24000.0,
             ["Frontier Well Services","South Texas Compression LLC"]),
            ("Injection gas quality and dehy check",       6000.0, 17000.0,
             ["South Texas Compression LLC"])],
    "Pumping Unit": [("Rod string inspection and torque survey", 9500.0, 27000.0,
             ["Frontier Well Services","Laredo Roustabout Services"]),
            ("Paraffin hot oil treatment — scheduled",     5500.0, 16000.0,
             ["South Texas Hot Oil","Monarch Chemical Services"])],
}
# Cadence is drawn PER WELL, across a wide band, because operators run different
# maintenance intensity on different wells. A single shared cadence gave every
# clean well 11-14 preventive jobs against 16-31 on the rest — so having closed
# the "zero workovers" leak, workover COUNT immediately became the same proxy.
# A per-well program makes the two distributions overlap.
# Not all scheduled work is cheap. Operators pull and replace ESPs at expected
# run life, replace rod strings on a program, and re-run tubing on schedule —
# planned jobs that cost the same as the failure they are meant to pre-empt.
# Without this tier, preventive spend tops out far below failure spend and
# lifetime workover DOLLARS separate the cohorts perfectly even once the count
# overlaps: clean $104-424k against $737k-2.8M on everything else.
PREVENTIVE_MAJOR = {
    "ESP": [("ESP change-out — scheduled run-life replacement", 62000.0, 155000.0,
             ["Frontier Well Services","South Texas Electrical LLC"])],
    "Gas Lift": [("Mandrel and valve string replacement — scheduled", 44000.0, 118000.0,
             ["Frontier Well Services","South Texas Compression LLC"])],
    "Pumping Unit": [("Rod string and pump replacement — scheduled program", 48000.0, 132000.0,
             ["Frontier Well Services","Laredo Roustabout Services"])],
}
# Both the cadence AND the major-work share are drawn PER WELL, across wide
# bands, because operators run very different maintenance programs well to well:
# a high-value producer gets an aggressive schedule with regular planned pulls, a
# marginal one gets the minimum. This has to be wide. With a single shared
# intensity the clean cohort's lifetime workover spend clustered in a narrow band
# well below the failing wells and lifetime DOLLARS separated the cohorts almost
# perfectly — and a narrow-but-passing overlap turned out to be luck: it
# collapsed from 24% to 1% on an unrelated change that merely shifted the RNG.
P_MAJOR_BAND    = (0.15, 0.58)  # per-well share of scheduled jobs that are major
PREVENTIVE_BAND = (85, 430)     # per-well base cadence, days

def make_well_invoices(aid, history, shutin_events, wi, lift):
    ledger=[]; econ=LIFT_ECON[lift]
    # Aggregate the month's actual fluid volumes and uptime.
    months={}
    for d in history:
        m=d['date'][:7]
        e=months.setdefault(m,{'oil':0.0,'wtr':0.0,'days':0,'up':0,'first':d['date']})
        e['oil']+=d.get('gross_oil_production_bbl',0.0)
        e['wtr']+=d.get('produced_water_volume_bbl',0.0)
        e['days']+=1
        if d['run_status']!='SHUT-IN': e['up']+=1
    age0=datetime.strptime(history[0]['date'],'%Y-%m-%d')
    for m in sorted(months):
        e=months[m]
        # A fully idle month used to be skipped. The MISSING invoice was itself an
        # unambiguous "this well was down all month" label, which undoes the
        # de-labeling work everywhere else in the build. Idle months bill fixed
        # cost: the pumper still drives the route and the lease still costs money.
        year=int(m[:4]); ci=cost_index(year)
        age=(datetime.strptime(e['first'],'%Y-%m-%d')-age0).days/365.25
        # older wells cost more to keep running: workovers, chemical, failures
        agef=1.0+0.030*age
        uptime=e['up']/max(1,e['days'])
        gross=(econ['fixed']*uptime
               +econ['oil_bbl']*e['oil']
               +econ['wtr_bbl']*e['wtr']
               +econ['power']*(e['oil']+e['wtr']))*agef*ci
        gross=usd(max(1500.0,gross*random.gauss(1.0,0.11)))
        ledger.append({
            "invoice_id":next_invoice(year),
            "invoice_date":e['first'],
            "category":"Monthly LOE",
            "vendor":random.choice(LOE_VENDORS),
            "service_description":f"Monthly lease operating expenses — {aid} — {m}",
            "gross_cost_usd":gross,
            "working_interest":wi,
            "total_usd":usd(gross*wi),
        })
    # Scheduled service on a cadence, independent of any failure.
    _h0=datetime.strptime(history[0]['date'],'%Y-%m-%d')
    _hN=datetime.strptime(history[-1]['date'],'%Y-%m-%d')
    _cad_lo=random.randint(*PREVENTIVE_BAND)
    _cad_hi=_cad_lo+random.randint(110,380)
    _p_major=random.uniform(*P_MAJOR_BAND)
    _pd=_h0+timedelta(days=random.randint(60,_cad_hi))
    while _pd<=_hN:
        _tier = PREVENTIVE_MAJOR if random.random()<_p_major else PREVENTIVE_SCOPES
        # The scope name is deliberately NOT written to the invoice — descriptions
        # stay generic so a scheduled job cannot be told from a repair. It is
        # unpacked and dropped on purpose.
        _desc,plo,phi,pvend=random.choice(_tier[lift])
        _py=_pd.year
        pg=usd(max(3000.0,random.uniform(plo,phi)*cost_index(_py)*random.gauss(1.0,0.10)))
        ledger.append({
            "invoice_id":next_invoice(_py),
            "invoice_date":_pd.strftime('%Y-%m-%d'),
            "category":"Workover",
            "vendor":random.choice(pvend),
            "service_description":f"Well services — {aid}",
            "gross_cost_usd":pg,
            "working_interest":wi,
            "total_usd":usd(pg*wi),
        })
        _pd+=timedelta(days=random.randint(_cad_lo,_cad_hi))

    for ev in shutin_events:
        sd=ev["date"]; year=int(sd[:4])
        # The dominant (most expensive) job drives both the cost and who does it,
        # so an overlapping episode bills the vendor for the bigger scope.
        scs=ev.get("concurrent",[ev["scenario"]])
        dom=max(scs,key=lambda s: wo_scope(s)[0])
        base,per=wo_scope(dom)
        gross=usd(max(5000.0,(base+per*ev["downtime_days"])
                      *cost_index(year)*random.gauss(1.0,0.14)))
        ledger.append({
            "invoice_id":next_invoice(year),
            "invoice_date":sd,
            "category":"Workover",
            "vendor":random.choice(wo_vendors(dom)),
            "service_description":f"Well services — {aid}",
            "gross_cost_usd":gross,
            "working_interest":wi,
            "total_usd":usd(gross*wi),
        })
    return ledger

def make_revenue(aid, history, nri, ngl_yield):
    """
    Monthly revenue derived from the well's OWN produced volumes. Gross volumes
    are valued at the realized Gulf Coast price, the operator's share is NRI
    (royalty already netted), then Texas severance, ad valorem and midstream
    deducts come off to give net revenue.

    RECONCILING THE LEDGER:
      oil_volume_bbl  x realized_oil_usd_bbl = gross_oil_revenue_usd
      residue_gas_mcf x realized_gas_usd_mcf = gross_gas_revenue_usd
      ngl_volume_bbl  x realized_ngl_usd_bbl = gross_ngl_revenue_usd
    Use residue_gas_mcf, NOT gas_volume_mcf, for the gas line. gas_volume_mcf is
    wet casinghead gas and is larger because the NGLs are still in it. Eagle Ford
    gas is rich, so the shrink here is bigger than a dry-gas play would show.
    """
    ledger=[]
    mo_o=defaultdict(float); mo_g=defaultdict(float); mo_w=defaultdict(float)
    for d in history:
        if d['run_status'] not in ('PRODUCING','CURTAILED'): continue
        m=d['date'][:7]
        mo_o[m]+=d.get('gross_oil_production_bbl',0.0)
        mo_g[m]+=d.get('casinghead_gas_mcf',0.0)
        mo_w[m]+=d.get('produced_water_volume_bbl',0.0)
    # Every month the well existed gets a record, including fully idle months
    # (all zeros). A missing month would flag a full-month outage on its own.
    for m in sorted({d['date'][:7] for d in history}):
        p=PRICE_DECK.get(m)
        if not p:
            # Silently skipping would drop the month and reintroduce the missing
            # month label. Fires only if END_DATE moves past the deck coverage.
            raise SystemExit(f"price deck has no entry for {m} — extend build_price_deck()")
        yr=int(m[:4])
        o=mo_o.get(m,0.0); g=mo_g.get(m,0.0); w=mo_w.get(m,0.0)
        ngl=g/1000.0*ngl_yield              # bbl NGL from wet gas
        g_res=max(0.0,g-ngl*1.15)           # residue gas after NGL shrink
        # Value the volumes AS PUBLISHED. Pricing full-precision floats and then
        # rounding the volumes separately leaves small records un-tie-out-able.
        o=round(o,2); g_res=round(g_res,3); ngl=round(ngl,2)

        gr_o=o*p["realized_oil_usd_bbl"]; gr_g=g_res*p["realized_gas_usd_mcf"]
        gr_n=ngl*p["realized_ngl_usd_bbl"]
        gross=gr_o+gr_g+gr_n
        n_o=gr_o*nri; n_g=gr_g*nri; n_n=gr_n*nri
        n_gross=n_o+n_g+n_n
        sev=n_o*SEV_TAX_OIL+(n_g+n_n)*SEV_TAX_GAS
        adval=max(0.0,n_gross)*AD_VALOREM
        gath=o*nri*OIL_TRANSPORT+g_res*nri*GAS_GPT
        ledger.append({
            "revenue_id":next_revenue(yr),"production_month":m,
            "oil_volume_bbl":o,"gas_volume_mcf":round(g,3),"residue_gas_mcf":g_res,
            "ngl_volume_bbl":ngl,"water_volume_bbl":round(w,2),
            "realized_oil_usd_bbl":p["realized_oil_usd_bbl"],
            "realized_gas_usd_mcf":p["realized_gas_usd_mcf"],
            "realized_ngl_usd_bbl":p["realized_ngl_usd_bbl"],
            "gross_oil_revenue_usd":round(gr_o,2),
            "gross_gas_revenue_usd":round(gr_g,2),
            "gross_ngl_revenue_usd":round(gr_n,2),
            "gross_revenue_usd":round(gross,2),
            "net_revenue_interest":nri,
            "net_revenue_before_deducts_usd":round(n_gross,2),
            "severance_tax_usd":round(sev,2),
            "ad_valorem_tax_usd":round(adval,2),
            "gathering_processing_usd":round(gath,2),
            "net_revenue_usd":round(n_gross-sev-adval-gath,2),
        })
    return ledger

# Supporting-asset monthly cost scales with what the asset actually handled.
SUPP_ECON = {
    "Facility":    {"fixed":5200.0,"rate":0.55,"field":"custody_transfer_rate_bblhr","mult":24.0},
    "Compressor":  {"fixed":6100.0,"rate":0.34,"field":"gas_throughput_mscfd","mult":1.0},
    "Tank Battery":{"fixed":2400.0,"rate":0.28,"field":"ot_1_run_ticket_bbl","mult":3.0},
    "SWD Well":    {"fixed":4300.0,"rate":0.61,"field":"daily_disposal_volume_bbl","mult":1.0},
}

def make_supp_invoices(aid, atype, history, wi):
    ledger=[]
    vendors=SUPP_VENDORS.get(atype,["South Texas Energy Partners Internal"])
    ec=SUPP_ECON.get(atype,{"fixed":4000.0,"rate":0.0,"field":None,"mult":1.0})
    months={}
    for d in history:
        m=d['date'][:7]
        e=months.setdefault(m,{'vol':0.0,'days':0,'up':0,'first':d['date']})
        if ec['field']: e['vol']+=d.get(ec['field'],0.0)*ec['mult']
        e['days']+=1
        if d['run_status']!='DOWN': e['up']+=1
    for m in sorted(months):
        e=months[m]; year=int(m[:4]); ci=cost_index(year)
        uptime=e['up']/max(1,e['days'])
        gross=(ec['fixed']*uptime+ec['rate']*e['vol'])*ci
        gross=usd(max(800.0,gross*random.gauss(1.0,0.10)))
        ledger.append({
            "invoice_id":next_sup_invoice(year),
            "invoice_date":e['first'],
            "category":"Monthly LOE",
            "vendor":random.choice(vendors),
            "service_description":f"Monthly operating expenses — {aid} — {m}",
            "gross_cost_usd":gross,
            "working_interest":wi,
            "total_usd":usd(gross*wi),
        })
    return ledger

# ── Supporting assets ─────────────────────────────────────────────────────────
def lease_daily(wells_dict, lease):
    by={}
    for k,v in wells_dict.items():
        if not k.startswith(lease): continue
        for d in v['history']:
            dt=d['date']
            if dt not in by: by[dt]={'oil':0,'water':0,'gas':0}
            by[dt]['oil']  +=d.get('gross_oil_production_bbl',0)
            by[dt]['water']+=d.get('produced_water_volume_bbl',0)
            by[dt]['gas']  +=d.get('casinghead_gas_mcf',0)
    return by

def make_loc(info):
    return {"basin":"Gulf Coast Basin","sub_basin":"Eagle Ford Shale","state":"TX",
            "county":info["county"],"region":"South Texas"}

def make_pers(lease):
    eng=PERSONNEL[lease]
    return {"well_perf_engineer":{"name":eng,"role":"Well Performance Engineer",
                                   "email":f"{eng.lower().replace(' ','.')}@steppartners.com"},
            "prod_technician":   {"name":f"{lease} Production Tech","role":"Production Technician",
                                   "email":f"prodtech.{lease.lower()}@steppartners.com"}}

def make_facility(wells_dict, lease, info, sd, ed):
    _AR.reset()
    prod=lease_daily(wells_dict,lease); h=[]; c=sd; wb=random.uniform(82,94); po=pw=pg=0.0
    down=0; day=0; next_down=random.randint(250,700)
    while c<=ed:
        dt=c.strftime('%Y-%m-%d'); dp=prod.get(dt,{'oil':0,'water':0,'gas':0})
        ad=amb_delta(c)
        oil=dp['oil']; water=dp['water']; gas=dp['gas']
        if down==0 and day>=next_down:
            down=random.randint(1,3); next_down=day+random.randint(250,700)
        running = down==0
        if down>0: down-=1
        day+=1
        if not running:
            h.append(stamp({"date":dt,"wellhead_pressure_psi":0.0,
                      "two_phase_separator_pressure_psi":0.0,
                      "three_phase_separator_pressure_psi":0.0,
                      "gas_sales_line_pressure_psi":0.0,
                      "tester_1_pressure_psi":0.0,"tester_1_temp_f":round(cl(nx(90,0.04,k='t1t')+0.8*ad,50,180),1),
                      "tester_1_oil_rate_bpd":0.0,"tester_1_water_rate_bpd":0.0,"tester_1_gas_rate_mcfd":0.0,
                      "tester_2_pressure_psi":0.0,"tester_2_temp_f":round(cl(nx(88,0.04,k='t2t')+0.8*ad,50,180),1),
                      "tester_2_oil_rate_bpd":0.0,"tester_2_water_rate_bpd":0.0,"tester_2_gas_rate_mcfd":0.0,
                      "tester_3_pressure_psi":0.0,"tester_3_temp_f":round(cl(nx(87,0.04,k='t3t')+0.8*ad,50,180),1),
                      "tester_3_oil_rate_bpd":0.0,"tester_3_water_rate_bpd":0.0,"tester_3_gas_rate_mcfd":0.0,
                      "prior_day_oil_production_bbl":round(po,2),"prior_day_water_production_bbl":round(pw,2),
                      "prior_day_gas_production_mcf":round(pg,3),
                      "sales_gas_rate_mscfd":0.0,
                      "sales_gas_temp_f":round(cl(nx(80,0.03,k='sgt')+0.7*ad,50,130),1),
                      "sales_gas_pressure_psi":0.0,
                      "custody_transfer_rate_bblhr":0.0,
                      "custody_transfer_temp_f":round(cl(nx(92,0.03,k='ctt')+0.7*ad,60,140),1),
                      "crude_gravity_api":round(cl(nx(42,0.02,k='api'),35,52),1),
                      "basic_sediment_water_pct":round(cl(nx(0.25,0.08,k='bsw'),0.0,2.0),3),
                      "produced_water_transfer_rate_bpd":0.0,
                      "water_transfer_discharge_psi":0.0,
                      "produced_water_temp_f":round(cl(nx(100,0.03,k='pwt')+0.6*ad,70,160),1),
                      "facility_oil_tank_level_ft":round(cl(nx(12,0.06,k='fot'),2,22),1),
                      "facility_water_tank_level_ft":round(cl(nx(10,0.06,k='fwt'),2,22),1),
                      "run_status":"DOWN"}))
            po=oil; pw=water; pg=gas; c+=timedelta(days=1); continue
        wh=cl(nx(wb,k='wh'),78,98); p2=cl(nx(wh-random.uniform(6,12)),60,wh-5)
        p3=cl(nx(p2-random.uniform(6,12)),40,p2-5); gs=cl(nx(p3-random.uniform(6,15)),20,p3-5)
        t1=cl(nx(oil*0.15,0.08),0,5000); t2=cl(nx(oil*0.12,0.08),0,5000); t3=cl(nx(oil*0.10,0.08),0,5000)
        h.append(stamp({"date":dt,"wellhead_pressure_psi":round(wh,1),
                  "two_phase_separator_pressure_psi":round(p2,1),
                  "three_phase_separator_pressure_psi":round(p3,1),
                  "gas_sales_line_pressure_psi":round(gs,1),
                  "tester_1_pressure_psi":round(cl(nx(p2*0.9),30,p2),1),"tester_1_temp_f":round(cl(nx(145,0.04,k='t1t')+0.5*ad,90,220),1),
                  "tester_1_oil_rate_bpd":round(t1,1),"tester_1_water_rate_bpd":round(cl(nx(t1*0.8,0.10),0,8000),1),"tester_1_gas_rate_mcfd":round(cl(nx(t1*0.4,0.10),0,5000),1),
                  "tester_2_pressure_psi":round(cl(nx(p2*0.88),30,p2),1),"tester_2_temp_f":round(cl(nx(140,0.04,k='t2t')+0.5*ad,90,220),1),
                  "tester_2_oil_rate_bpd":round(t2,1),"tester_2_water_rate_bpd":round(cl(nx(t2*0.8,0.10),0,8000),1),"tester_2_gas_rate_mcfd":round(cl(nx(t2*0.4,0.10),0,5000),1),
                  "tester_3_pressure_psi":round(cl(nx(p2*0.85),30,p2),1),"tester_3_temp_f":round(cl(nx(138,0.04,k='t3t')+0.5*ad,90,220),1),
                  "tester_3_oil_rate_bpd":round(t3,1),"tester_3_water_rate_bpd":round(cl(nx(t3*0.8,0.10),0,8000),1),"tester_3_gas_rate_mcfd":round(cl(nx(t3*0.4,0.10),0,5000),1),
                  "prior_day_oil_production_bbl":round(po,2),"prior_day_water_production_bbl":round(pw,2),"prior_day_gas_production_mcf":round(pg,3),
                  "sales_gas_rate_mscfd":round(cl(nx(gas*0.85,0.04),0,50000),2),
                  "sales_gas_temp_f":round(cl(nx(92,0.03,k='sgt')+0.6*ad,60,130),1),
                  "sales_gas_pressure_psi":round(cl(nx(gs*1.1,0.02),gs,800),1),
                  "custody_transfer_rate_bblhr":round(cl(nx(oil/24,0.05),0,500),2),
                  "custody_transfer_temp_f":round(cl(nx(105,0.03,k='ctt')+0.6*ad,70,140),1),
                  "crude_gravity_api":round(cl(nx(42,0.02,k='api'),35,52),1),
                  "basic_sediment_water_pct":round(cl(nx(0.25,0.08,k='bsw'),0.0,2.0),3),
                  "produced_water_transfer_rate_bpd":round(cl(nx(water*0.9,0.04),0,20000),1),
                  "water_transfer_discharge_psi":round(cl(nx(180,0.05),100,400),1),
                  "produced_water_temp_f":round(cl(nx(118,0.03,k='pwt')+0.5*ad,80,160),1),
                  "facility_oil_tank_level_ft":round(cl(nx(12,0.06,k='fot'),2,22),1),
                  "facility_water_tank_level_ft":round(cl(nx(10,0.06,k='fwt'),2,22),1),
                  "run_status":"PRODUCING"}))
        po=oil; pw=water; pg=gas; c+=timedelta(days=1)
    return h

def make_compressor(wells_dict, lease, sd, ed):
    _AR.reset()
    prod=lease_daily(wells_dict,lease); h=[]; c=sd
    su=random.uniform(28,55); di=random.uniform(180,350); rp=random.uniform(900,1200); lu=random.uniform(42,58)
    down=0; day=0; next_down=random.randint(120,400)
    while c<=ed:
        dt=c.strftime('%Y-%m-%d'); gas=prod.get(dt,{}).get('gas',0); ad=amb_delta(c)
        if down==0 and day>=next_down:
            down=random.randint(1,5); next_down=day+random.randint(120,400)
        running = down==0
        if down>0: down-=1
        day+=1
        if not running:
            h.append(stamp({"date":dt,"compressor_suction_pressure_psi":round(cl(nx(su*1.25,k='su'),15,80),1),
                      "compressor_suction_temp_f":round(cl(nx(78,0.04,k='sut')+0.8*ad,45,130),1),
                      "compressor_discharge_pressure_psi":0.0,
                      "compressor_discharge_temp_f":round(cl(nx(95,0.04,k='dit')+0.8*ad,50,180),1),
                      "gas_throughput_mscfd":0.0,
                      "compressor_engine_rpm":0.0,
                      "fuel_gas_rate_mcfd":0.0,
                      "compressor_engine_temp_f":round(cl(nx(92,0.04,k='et')+0.8*ad,50,180),1),
                      "compressor_lube_oil_pressure_psi":0.0,
                      "daily_runtime_hrs":0.0,
                      "run_status":"DOWN"}))
            c+=timedelta(days=1); continue
        h.append(stamp({"date":dt,"compressor_suction_pressure_psi":round(cl(nx(su,k='su'),15,80),1),
                  "compressor_suction_temp_f":round(cl(nx(88,0.04,k='sut')+0.7*ad,60,130),1),
                  "compressor_discharge_pressure_psi":round(cl(nx(di,k='dp'),120,500),1),
                  "compressor_discharge_temp_f":round(cl(nx(215,0.04,k='dit')+0.35*ad,150,300),1),
                  "gas_throughput_mscfd":round(cl(nx(gas*0.9,0.05,k='thr'),0,50000),1),
                  "compressor_engine_rpm":round(cl(nx(rp,k='rpm'),700,1400),0),
                  "fuel_gas_rate_mcfd":round(cl(nx(gas*0.04,0.06,k='fg'),0,2000),2),
                  "compressor_engine_temp_f":round(cl(nx(195,0.04,k='et')+0.3*ad,140,260),1),
                  "compressor_lube_oil_pressure_psi":round(cl(nx(lu,k='lop'),30,75),1),
                  "daily_runtime_hrs":round(cl(nx(22.5,0.02,k='rt'),0,24),1),
                  "run_status":"PRODUCING"}))
        c+=timedelta(days=1)
    return h

# ── Tank battery physical spec ────────────────────────────────────────────────
OT_CAP_BBL, OT_HEIGHT_FT = 2500.0, 24.0   # 3 oil stock tanks per battery
WT_CAP_BBL, WT_HEIGHT_FT = 3000.0, 24.0   # produced-water tank
GB_CAP_BBL, GB_HEIGHT_FT = 1500.0, 20.0   # gun barrel / free-water knockout
OT_BPF, WT_BPF, GB_BPF = OT_CAP_BBL/OT_HEIGHT_FT, WT_CAP_BBL/WT_HEIGHT_FT, GB_CAP_BBL/GB_HEIGHT_FT
ALARM_FRAC  = 0.90        # float switch trips at 90% of shell height
MAX_FILL    = (0.94,0.97) # usable fill band — tanks keep freeboard, never gauge full
TANK_SPECS = {"oil_tank_count":3,"oil_tank_capacity_bbl":OT_CAP_BBL,
              "oil_tank_height_ft":OT_HEIGHT_FT,"oil_tank_bbl_per_ft":round(OT_BPF,2),
              "water_tank_capacity_bbl":WT_CAP_BBL,"water_tank_height_ft":WT_HEIGHT_FT,
              "water_tank_bbl_per_ft":round(WT_BPF,2),
              "gun_barrel_capacity_bbl":GB_CAP_BBL,"gun_barrel_height_ft":GB_HEIGHT_FT,
              "gun_barrel_bbl_per_ft":round(GB_BPF,2),
              "high_level_alarm_setpoint_pct":round(ALARM_FRAC*100)}

def make_tank(wells_dict, lease, sd, ed):
    """Production-coupled tank battery.

    Oil and water inflow come from the lease's own wells, and run tickets are the
    volumes actually hauled out, so battery throughput reconciles with well
    production (sum of tickets == oil produced, less closing inventory).

    Levels are end-of-day gauge readings in feet. high_level_alarm trips when a
    custody-transfer or water-transfer disruption backs inventory up past the
    setpoint — a physically caused event, not a random flag.
    """
    _AR.reset()
    prod=lease_daily(wells_dict,lease); h=[]; c=sd
    # reference throughput for the gun barrel operating band
    _oils=sorted(d['oil'] for d in prod.values()) or [1.0]
    oil_ref=max(1.0,_oils[int(0.95*(len(_oils)-1))])
    _wtrs=sorted(d['water'] for d in prod.values()) or [1.0]
    wtr_ref=max(1.0,_wtrs[int(0.95*(len(_wtrs)-1))])
    # Shell sizes stay standard — these are ordinary 2,500 bbl stock tanks and
    # every battery publishes the same nameplate. What IS sized to the lease is
    # the daily custody-transfer and water-transfer capacity: an operator
    # contracts trucking and pipeline capacity to match its own volume.
    # This matters physically. With a flat haul capacity, a transfer disruption
    # backed inventory up on the high-rate leases and the low-rate leases simply
    # kept up, so their high-level float never tripped once in 18 years and the
    # alarm signal was dead on exactly the leases where inventory management is
    # tightest. Proportional capacity makes a disruption back up by the same
    # fraction of throughput everywhere.
    OT_HAUL=max(300.0,oil_ref*random.uniform(0.42,0.62))     # per tank, bbl/d
    WT_HAUL=max(600.0,wtr_ref*random.uniform(1.05,1.45))     # bbl/d
    inv=[OT_CAP_BBL*random.uniform(.30,.45) for _ in range(3)]
    winv=WT_CAP_BBL*random.uniform(.30,.45)
    ginv=GB_CAP_BBL*random.uniform(.35,.55)
    split=[0.36,0.34,0.30]; disrupt=0; wdisrupt=0
    down=0; next_down=random.randint(250,700); day=0
    while c<=ed:
        dt=c.strftime('%Y-%m-%d'); dp=prod.get(dt,{'oil':0,'water':0,'gas':0})
        oil=dp['oil']; water=dp['water']; ad=amb_delta(c)

        # hauling / transfer availability
        if disrupt==0  and random.random()<1/150: disrupt=random.randint(2,8)
        if wdisrupt==0 and random.random()<1/260: wdisrupt=random.randint(1,3)
        of=random.uniform(.05,.35) if disrupt>0  else 1.0
        wf=random.uniform(.05,.35) if wdisrupt>0 else 1.0
        if disrupt>0:  disrupt-=1
        if wdisrupt>0: wdisrupt-=1

        # Battery maintenance outage: the oil custody-transfer train (LACT /
        # transfer pump) is down, so no oil is hauled out while inflow from the
        # wells continues and inventory backs up. Water disposal is a separate
        # system and keeps running — the produced-water tank holds well under one
        # day of lease water, so halting it too would force the outage to abort
        # almost every time and downtime would be vanishingly rare.
        # An outage cannot be held if an oil tank would overtop — the operator
        # ends it and hauls — which keeps every DOWN row free of oil run tickets,
        # so run_status never contradicts the payload. The guard tests the LOWEST
        # possible jittered ceiling (MAX_FILL[0]) so the forced-diversion branch
        # below can never fire on a DOWN day.
        if down==0 and day>=next_down:
            down=random.randint(1,3); next_down=day+random.randint(250,700)
        maint = down>0
        if maint and any(inv[i]+oil*split[i] > OT_CAP_BBL*MAX_FILL[0] for i in range(3)):
            maint=False; down=0
        if maint:
            of=0.0
            down-=1
        day+=1

        # Gun barrel sits upstream and passes oil through: its level reflects the
        # operating setpoint scaled by throughput, with no cumulative drift (an
        # integrated drift saturates against the clamps and pins the level).
        gfrac=cl(0.32+0.30*(oil/oil_ref)+random.gauss(0,0.035),0.18,0.82)
        ginv=GB_CAP_BBL*gfrac

        alarm=0; tk=[]
        for i in range(3):
            avail=inv[i]+oil*split[i]
            tgt=OT_CAP_BBL*random.uniform(.28,.44)
            haul=min(max(0.0,avail-tgt),OT_HAUL*of)
            inv[i]=avail-haul
            ceil=OT_CAP_BBL*random.uniform(*MAX_FILL)   # freeboard, jittered
            if inv[i]>ceil:                            # excess diverted, not stored
                haul+=inv[i]-ceil; inv[i]=ceil
            if inv[i]/OT_BPF>ALARM_FRAC*OT_HEIGHT_FT: alarm=1
            tk.append(haul)

        wavail=winv+water
        wtgt=WT_CAP_BBL*random.uniform(.24,.42)
        whaul=min(max(0.0,wavail-wtgt),WT_HAUL*wf)
        winv=wavail-whaul
        wceil=WT_CAP_BBL*random.uniform(*MAX_FILL)
        if winv>wceil:
            whaul+=winv-wceil; winv=wceil
        if winv/WT_BPF>ALARM_FRAC*WT_HEIGHT_FT: alarm=1

        h.append(stamp({"date":dt,
                  "ot_1_level_ft":round(inv[0]/OT_BPF,2),"ot_1_temperature_f":round(cl(nx(105,0.03,k='ot1t')+0.55*ad,60,145),1),
                  "ot_1_run_ticket_bbl":round(tk[0],1),
                  "ot_2_level_ft":round(inv[1]/OT_BPF,2),"ot_2_temperature_f":round(cl(nx(103,0.03,k='ot2t')+0.55*ad,60,145),1),
                  "ot_2_run_ticket_bbl":round(tk[1],1),
                  "ot_3_level_ft":round(inv[2]/OT_BPF,2),"ot_3_temperature_f":round(cl(nx(102,0.03,k='ot3t')+0.55*ad,60,145),1),
                  "ot_3_run_ticket_bbl":round(tk[2],1),
                  "wt_1_level_ft":round(winv/WT_BPF,2),"wt_1_temperature_f":round(cl(nx(95,0.03,k='wt1t')+0.6*ad,50,135),1),
                  "wt_1_disposal_ticket_bbl":round(whaul,1),
                  "gb_1_level_ft":round(ginv/GB_BPF,2),"gb_1_temperature_f":round(cl(nx(100,0.03,k='gb1t')+0.55*ad,55,135),1),
                  "high_level_alarm":alarm,"run_status":("DOWN" if maint else "PRODUCING")}))
        c+=timedelta(days=1)
    return h

def make_swd(wells_dict, lease, info, sd, ed):
    """Salt-water disposal well.

    Surface injection pressure responds to cumulative injected volume: as the
    disposal interval fills, pressure climbs, and periodic acid stimulations
    knock it back down. A flat 1400 psi across 18 years and 6.4 million barrels
    (the previous behaviour) removes the single most useful SWD maintenance
    signal there is.
    """
    _AR.reset()
    prod=lease_daily(wells_dict,lease); h=[]; c=sd; cum=0.0
    ib=random.uniform(800,2500)
    p_base=random.uniform(1180,1520)          # clean-formation injection pressure
    p_span=random.uniform(620,1150)           # pressure climb at full fill
    cum_ref=random.uniform(4.0e6,9.0e6)       # cum bbl at which span is reached
    stim_credit=0.0                           # pressure relief from stimulation
    next_stim=random.randint(700,1400); day=0
    stim_days=set()
    down=0; next_down=random.randint(200,600)
    while c<=ed:
        dt=c.strftime('%Y-%m-%d'); ad=amb_delta(c)
        water=prod.get(dt,{}).get('water',0)

        # scheduled/unscheduled pump maintenance
        if down==0 and day>=next_down:
            down=random.randint(1,4); next_down=day+random.randint(200,600)
        running = down==0
        if down>0: down-=1

        d=cl(nx(min(ib,water*0.95),0.04,k='rate'),0,5000) if running else 0.0
        cum+=d

        # acid stimulation: relieves most of the accumulated pressure rise
        if day>=next_stim:
            stim_credit=p_span*(cum/cum_ref)*0.62
            next_stim=day+random.randint(700,1600); stim_days.add(dt)
        fill_press=max(0.0,p_span*(cum/cum_ref)-stim_credit)
        surf=nx(p_base+fill_press,0.035,k='surf') if running else nx(p_base*0.18,0.05,k='surfidle')

        h.append(stamp({"date":dt,
                  "surface_injection_pressure_psi":round(cl(surf,80,3400),1),
                  "swdwellhead_pressure_psi":round(cl(surf*0.965,70,3300),1),
                  "tubing_pressure_psi":round(cl(surf*0.985,75,3350),1),
                  "casing_pressure_psi":round(cl(nx(120,0.05,k='csg'),50,400),1),
                  "injection_temp_f":round(cl(nx(95,0.03,k='itemp')+0.35*ad,60,140),1),
                  "daily_disposal_volume_bbl":round(d,1),
                  "cumulative_disposal_volume_bbl":round(cum,1),
                  "pump_speed_spm":round(cl(nx(42,0.03,k='spm'),20,70),1) if running else 0.0,
                  "pump_rpm":round(cl(nx(380,0.03,k='rpm'),200,600),0) if running else 0.0,
                  "injection_pump_suction_psi":round(cl(nx(45,0.04,k='suc'),20,100),1) if running else 0.0,
                  "injection_pump_discharge_psi":round(cl(surf*1.035,80,3500),1) if running else 0.0,
                  "lube_oil_pressure_psi":round(cl(nx(48,0.04,k='lop'),30,75),1) if running else 0.0,
                  "lube_oil_temp_f":round(cl(nx(175,0.03,k='lot')+0.2*ad,120,220),1) if running else round(cl(nx(95,0.04,k='lot')+0.6*ad,60,140),1),
                  "vibration_high_freq_hz":round(cl(nx(0.8,0.08,k='vib'),0,5),3) if running else 0.0,
                  "packing_leak_rate_gpm":round(cl(nx(0.02,0.10,k='pk'),0,0.5),3),
                  "lube_oil_volume_gal":round(cl(nx(12,0.04,k='lov'),5,20),1),
                  "stimulation_performed":1 if dt in stim_days else 0,
                  "run_status":"PRODUCING" if running else "DOWN"}))
        c+=timedelta(days=1); day+=1
    return h


# ── V1-style pipeline API (used by v2/pipeline.py) ────────────────────────────

def simulate_well(asset_id, lift, spud_date, is_clean=False):
    """Generate one well history like v1 simulate_well.

    Returns (history, shutin_events, pump_curve_or_None).
    """
    if isinstance(spud_date, datetime):
        spud = spud_date
    else:
        spud = datetime.strptime(str(spud_date)[:10], "%Y-%m-%d")
    if lift == "ESP":
        hist, pump_curve, shutin = gen_esp(asset_id, spud, is_clean)
        return hist, shutin, pump_curve
    if lift == "Gas Lift":
        hist, shutin = gen_gl(asset_id, spud, is_clean)
        return hist, shutin, None
    hist, shutin = gen_pu(asset_id, spud, is_clean)
    return hist, shutin, None


def build_wells():
    """Build the fixed 100-well V2 catalog (asset_id, lease, lift, spud, WI/NRI)."""
    random.seed(20260708)
    np.random.seed(20260708)
    wells = []
    for lease, info in LEASES.items():
        pool = []
        for lift, count in info["dist"].items():
            pool.extend([lift] * count)
        random.shuffle(pool)
        s_esp, s_gl = CLEAN_SLOTS[lease]
        if pool[s_esp] != "ESP":
            cand = [i for i, lift in enumerate(pool) if lift == "ESP" and i != s_gl]
            if cand:
                j = cand[0]
                pool[s_esp], pool[j] = pool[j], pool[s_esp]
        if pool[s_gl] != "Gas Lift":
            cand = [i for i, lift in enumerate(pool) if lift == "Gas Lift" and i != s_esp]
            if cand:
                j = cand[0]
                pool[s_gl], pool[j] = pool[j], pool[s_gl]
        for i, lift in enumerate(pool):
            suffix = "-V" if lift == "Pumping Unit" else "-H"
            well_no = LEASE_BASE[lease] + i + 1
            aid = f"{lease}-1H-{well_no:03d}-V2{suffix}"
            spud = rd(START_MIN, START_MAX)
            wells.append(
                {
                    "asset_id": aid,
                    "lease": lease,
                    "lift": lift,
                    "spud_date": spud.strftime("%Y-%m-%d"),
                    "is_clean": aid in ALL_CLEAN,
                    "working_interest": info["wi"],
                    "net_revenue_interest": info["nri"],
                    "area": info["area"],
                    "lease_type": info["lt"],
                    "county": info["county"],
                }
            )
    return wells
