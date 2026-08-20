import json, os, random, sys
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter

random.seed(20260708)
np.random.seed(20260708)

OUTPUT_FILE = os.environ.get("RBNR_OUTPUT", "RBNR_V2_Master.json")
END_DATE    = datetime(2026, 7, 8)
START_MIN   = datetime(2008, 1, 1)
START_MAX   = datetime(2010, 12, 31)

# Build scale. Full is the default and is what ships.
#   full  100 wells, 2008-2026 histories  (~60s, ~1 GB peak, ~650 MB output)
#   mid   100 wells, 2019-2026 histories  — real well counts, a third of the rows
#   test   16 wells, 2019-2026 histories  — smoke test only
# mid is the useful one for iterating: it exercises every check with enough
# sample size to be meaningful, without paying for the full history.
SCALE = os.environ.get("RBNR_SCALE", "full")
# An unrecognised value used to fall through to a full 100-well build while
# quietly downgrading MIN_INV from 20,000 to 100 and skipping five verification
# checks — so RBNR_SCALE=fulll produced a complete, ALL CLEAR, exit-0 artifact
# with weakened validation. Reject anything not on the list.
if SCALE not in ("full","mid","test"):
    raise SystemExit(f"RBNR_SCALE={SCALE!r} is not valid — use full, mid, or test")
if SCALE in ("test","mid"):
    START_MIN = datetime(2019, 1, 1); START_MAX = datetime(2019, 6, 30)

COMPANY         = "Red Bluff Resources"
ORGANIZATION_ID = "RED_BLUFF_RESOURCES"

# ── Leases ────────────────────────────────────────────────────────────────────
# NRI is WI x (1 - royalty), set by lease type:
#   Chaparral 0.750 = 1.0 x (1 - 0.250) Fee Land
#   Mesquite  0.800 = 1.0 x (1 - 0.200) University Land
#   Caliche   0.781 = 1.0 x (1 - 0.219) PSF Land
#   Llano     0.812 = 1.0 x (1 - 0.188) Railroad ROW
# Delaware Basin: Chaparral (Reeves), Mesquite (Loving)
# Midland Basin:  Caliche (Midland), Llano (Ector)
LEASES = {
    "Chaparral": {
        "basin": "Permian Basin", "sub_basin": "Delaware Basin",
        "state": "TX", "county": "Reeves County", "region": "West Texas",
        "nri": 0.750, "wi": 1.0, "lease_type": "Fee Land",
        "engineer": "Jake Thornton",
        # Permian Delaware: more ESP and rod pump, some gas lift
        "dist": {"ESP": 7, "Gas Lift": 5, "Pumping Unit": 13},
        "swd_pump": "Triplex",
    },
    "Mesquite": {
        "basin": "Permian Basin", "sub_basin": "Delaware Basin",
        "state": "TX", "county": "Loving County", "region": "West Texas",
        "nri": 0.800, "wi": 1.0, "lease_type": "University Land",
        "engineer": "Maria Castillo",
        "dist": {"ESP": 8, "Gas Lift": 4, "Pumping Unit": 13},
        "swd_pump": "HPS",
    },
    "Caliche": {
        "basin": "Permian Basin", "sub_basin": "Midland Basin",
        "state": "TX", "county": "Midland County", "region": "West Texas",
        "nri": 0.781, "wi": 1.0, "lease_type": "PSF Land",
        "engineer": "Jake Thornton",
        # Midland Basin: heavier rod pump
        "dist": {"ESP": 6, "Gas Lift": 3, "Pumping Unit": 16},
        "swd_pump": "Triplex",
    },
    "Llano": {
        "basin": "Permian Basin", "sub_basin": "Midland Basin",
        "state": "TX", "county": "Ector County", "region": "West Texas",
        "nri": 0.812, "wi": 1.0, "lease_type": "Railroad ROW",
        "engineer": "Maria Castillo",
        "dist": {"ESP": 4, "Gas Lift": 5, "Pumping Unit": 16},
        "swd_pump": "HPS",
    },
}

if SCALE == "test":
    for _d in LEASES.values():
        _d["dist"] = {"ESP":1, "Gas Lift":1, "Pumping Unit":2}

N_WELLS = sum(sum(d["dist"].values()) for d in LEASES.values())
N_ASSETS = N_WELLS + 4*len(LEASES)
if SCALE == "full" and N_WELLS != 100:
    raise SystemExit("lease distribution no longer totals 100 wells")

# ── Clean wells — 2-3 per lease, 10 total ────────────────────────────────────
# Clean wells have NO failure physics applied. They are NOT 100% uptime: like
# every other well they take ordinary non-failure downtime (see routine_outages),
# because a well that never goes down is trivially identifiable as the healthy
# cohort no matter how well the financial ledger is de-labeled.
# Selected by WELL NUMBER, not by full asset ID: the -H/-V suffix depends on the
# shuffled lift pool, so matching on full IDs made the count seed-dependent
# (10 at seed 20260708, but 2-6 at other seeds). Well numbers are stable.
# Scattered through each lease's number range instead of clustered at the
# start. Previously 1/2/3, 26/28, 51/52, 76/77/83, so "low well number in lease"
# correlated with the healthy cohort.
CLEAN_WELL_NUMBERS = {
    "Chaparral": [4, 12, 21],
    "Mesquite":  [33, 44],
    "Caliche":   [58, 69],
    "Llano":     [81, 88, 96],
}
if SCALE == "test":
    CLEAN_WELL_NUMBERS = {"Chaparral":[2], "Mesquite":[6], "Caliche":[10], "Llano":[14]}
N_CLEAN = sum(len(v) for v in CLEAN_WELL_NUMBERS.values())
if SCALE == "full" and N_CLEAN != 10:
    raise SystemExit("clean well selection no longer totals 10")
ALL_CLEAN_NUMS = {n for ns in CLEAN_WELL_NUMBERS.values() for n in ns}

# ── Wellbore geometry ────────────────────────────────────────────────────────
# Target horizontal fraction by sub-basin (Delaware drilled horizontal earlier
# and harder than Midland over this 2008-2026 vintage).
HORIZ_FRAC = {"Delaware Basin": 0.72, "Midland Basin": 0.56}
# Relative preference for landing on a horizontal, by lift method. Higher = more
# likely to be horizontal. Rod pump on horizontals is common in the Permian,
# especially in later life, so Pumping Unit is low but not zero.
# Kept close to 1.0 on purpose: a stronger skew starves the vertical cell on the
# smaller lift classes and makes the geometry comparison meaningless.
H_AFFINITY = {"ESP": 1.12, "Gas Lift": 1.06, "Pumping Unit": 1.0}

# Wellbore geometry now drives production. Previously Horizontal/Vertical
# and lateral_length_ft were assigned and stored but never touched qi, so a
# 10,500 ft lateral produced exactly like a vertical.
# These are 2008-2010 spud dates, so they are early-generation Permian laterals:
# uplift is modest (roughly 1.55-2.6x a vertical) and decline is steeper, which is
# the characteristic shale trade-off. Modern 2020s laterals would be far higher.
LATERAL_MIN_FT = 4500
LATERAL_MAX_FT = 10500
H_QI_AT_MIN    = 1.55     # qi multiplier at a 4,500 ft lateral
H_QI_AT_MAX    = 2.60     # qi multiplier at a 10,500 ft lateral
H_DI_MULT      = 1.55     # horizontals decline faster
H_B_BONUS      = 0.15     # and hold a slightly longer hyperbolic tail

def geom_params(lateral_ft):
    """(qi_mult, di_mult, b_bonus) from wellbore geometry."""
    if lateral_ft <= 0:
        return 1.0, 1.0, 0.0
    f = (lateral_ft - LATERAL_MIN_FT) / float(LATERAL_MAX_FT - LATERAL_MIN_FT)
    f = max(0.0, min(1.0, f))   # inlined: cl() is defined further down the file
    return H_QI_AT_MIN + (H_QI_AT_MAX - H_QI_AT_MIN) * f, H_DI_MULT, H_B_BONUS

# ── Failure sub-types per lift ─────────────────────────────────────────────────
# (name, det_min_days, det_max_days, shutin_min, shutin_max)
# Sub-types are physics labels for internal use only — NOT written to JSON

# PU sub-types — Permian rod pump specific
PU_TYPES = [
    # Parted Rods - Mechanical: sudden fatigue failure, minimal pre-signal
    ("Parted Rods - Mechanical",    5,   30,  10, 25),
    # Parted Rods - Corrosion: H2S/CO2 weakens rod first, casing pressure rises
    ("Parted Rods - Corrosion",    90,  270,  12, 30),
    # Worn Pump - Sand: abrasive wear, slow fillage decline
    ("Worn Pump - Sand",          120,  365,   8, 20),
    # Worn Pump - Scale: calcium carbonate, different fillage pattern
    ("Worn Pump - Scale",         150,  365,  10, 25),
    # Stuck Pump - Sand: fast, SPM drops, fillage zeroes
    ("Stuck Pump - Sand",          10,   45,   8, 18),
    # Rod String Vibration: current and stroke oscillate
    ("Rod String Vibration",       30,  120,   6, 15),
]

# ESP sub-types — Permian Basin
ESP_TYPES = [
    # Motor Overtemp - Gradual: insulation wear over months
    ("Motor Overtemp - Gradual",  120,  365,   8, 20),
    # Motor Overtemp - Rapid: high amp event, fast shutdown
    ("Motor Overtemp - Rapid",     15,   60,   8, 18),
    # Pump Cavitation - Gas: gas interference driving cavitation
    ("Pump Cavitation - Gas",      60,  210,   8, 18),
    # Pump Cavitation - Scale: scale restricting intake
    ("Pump Cavitation - Scale",   120,  365,  10, 22),
]

# GL sub-types — Permian Basin mandrel/valve failures
GL_TYPES = [
    # Mandrel Washout - Sand: faster erosion
    ("Mandrel Washout - Sand",     45,  120,   8, 18),
    # Mandrel Washout - Chemical: slower corrosion-driven
    ("Mandrel Washout - Chemical",  90, 270,  10, 22),
    # Valve Failure - Stuck Open: injection runs high
    ("Valve Failure - Stuck Open",  30, 120,   6, 16),
    # Valve Failure - Stuck Closed: injection drops to zero
    ("Valve Failure - Stuck Closed",30,  90,   8, 18),
]

# ── Pump models (ESP) ─────────────────────────────────────────────────────────
# Head-curve polynomials solved so head declines smoothly from shut-in head
# through BEP to max flow. The previous coefficients drove head onto the 5 ft
# clamp floor at and above BEP on all three models.
PUMP_MODELS = [
    {"model":"Centrilift GN5600","manufacturer":"Centrilift","bep_bpd":4000,
     "min_bpd":2000,"max_bpd":6000,"stages":120,
     "poly":{"a":-1.083333333e-06,"b":0.00083333333,"c":52.0}},
    {"model":"REDA D3000N","manufacturer":"REDA","bep_bpd":2800,
     "min_bpd":1200,"max_bpd":4000,"stages":100,
     "poly":{"a":-2.886904762e-06,"b":0.003797619,"c":46.0}},
    {"model":"Centrilift FC1200","manufacturer":"Centrilift","bep_bpd":1200,
     "min_bpd":800,"max_bpd":1600,"stages":150,
     "poly":{"a":-1.614583333e-05,"b":0.012708333,"c":34.0}},
]

# ── Vendors — Permian Basin specific ─────────────────────────────────────────
LOE_VENDORS = [
    "Red Bluff Internal",
    "Permian Chemical Supply",
    "Delaware Basin Compression",
    "Midland Hot Oil Service",
    "Reeves County Water Hauling",
    "West Texas Electrical Services",
    "Permian Basin Roustabout",
]
# Workover vendors are no longer a flat list. Each vendor has a discipline, and
# the scope table below names the vendors qualified to run that scope. A rod-and-
# pump contractor never gets billed for an ESP cable-and-motor change-out.
WO_VENDORS_BY_DISCIPLINE = {
    "rod_pump":     ["West Texas Rod & Pump","Permian Rod Service Co"],
    "rig_general":  ["Permian Well Services","Delaware Basin Workover"],
    "esp":          ["Midland Artificial Lift Services","Delaware Basin ESP Services"],
    "gas_lift":     ["Permian Gas Lift Specialists","Midland Artificial Lift Services"],
    "stimulation":  ["West Texas Pumping & Stimulation","Delaware Basin Workover"],
    "chemical":     ["Permian Chemical Supply","West Texas Pumping & Stimulation"],
    "surface":      ["Permian Basin Roustabout","West Texas Electrical Services"],
}
# Supporting-asset vendors are split by what is actually being billed, so the
# vendor matches the line item and not just the asset type.
SUPP_VENDORS = {
    ("Facility","routine"):     ["West Texas Electrical Services","Red Bluff Internal"],
    ("Facility","repair"):      ["West Texas Electrical Services","Permian Basin Roustabout"],
    ("Compressor","routine"):   ["Delaware Basin Compression","Red Bluff Internal"],
    ("Compressor","repair"):    ["Delaware Basin Compression","West Texas Electrical Services"],
    ("Tank Battery","routine"): ["Red Bluff Internal","Midland Hot Oil Service"],
    ("Tank Battery","repair"):  ["Permian Basin Roustabout","Midland Hot Oil Service"],
    ("SWD Well","routine"):     ["Reeves County Water Hauling","Red Bluff Internal"],
    ("SWD Well","repair"):      ["Permian Well Services","West Texas Pumping & Stimulation"],
}

# ── Cost model constants ──────────────────────────────────────────────────────
# UNIT RATES ARE ENGINEERING ESTIMATES, NOT SOURCED FROM AN AFE OR AN ACTUAL
# OPERATING BUDGET. They are shaped to plausible Permian magnitudes so the cost
# drivers exist and are learnable. Replace with real rates before any economic
# conclusion is drawn from this dataset.
#
# Well LOE — a built-up cost, not a function of oil volume alone:
#   fixed (lift-specific) + oil handling + produced-water handling
#   + gas gathering/compression + electricity (from actual amps/runtime)
#   + gas-lift injection compression + well-age escalation + idle-well standby
LOE_FIXED_BY_LIFT = {          # $/month fixed: pumper route, surface maintenance
    # Narrow spread on purpose: a wide random fixed base drowns the variable
    # drivers and makes cost look oil-only inside a fixed-oil band.
    "Pumping Unit": (3600, 4900),
    "ESP":          (4900, 6400),
    "Gas Lift":     (4300, 5700),
}
# These are WELLSITE-ONLY rates now. Previously the well was billed
# $3.20/bbl including a "trucking share" while the tank battery separately billed
# the haul, the facility separately billed the same oil, and the SWD separately
# billed the same water: every barrel was charged three or four times and the
# lifetime cost total was inflated accordingly. Downstream handling now lives
# exclusively on the supporting assets (SUPP_RATES below).
LOE_OIL_RATE    = 0.55    # $/bbl oil — wellsite treating chemical and gauging only
LOE_WATER_RATE  = 0.42    # $/bbl produced water — wellsite gathering only
LOE_GAS_RATE    = 0.07    # $/mcf — wellsite separation only
LOE_POWER_USD_KWH = 0.085 # $/kWh West Texas industrial
LOE_GL_INJ_RATE = 0.22    # $/mscf injected — compression fuel and rental
LOE_AGE_ESC     = 0.018   # +1.8%/yr on the fixed component (aging surface equip)
LOE_IDLE_PER_DAY= (110, 260)   # $/day a non-producing well still costs
LOE_MAX_CAP     = 450000  # sanity ceiling only — must not bind in practice

# ── Ordinary (non-failure) well downtime ─────────────────────────────────────
# Every well, including the healthy cohort, now takes routine outages:
# weather, power loss, offset frac hits, facility upsets, pumper error, scheduled
# surface work. Previously the 10 clean wells ran 100% uptime for ~17 straight
# years, so run_status alone identified the healthy cohort with perfect accuracy
# and all the ledger de-labeling work was moot.
# These outages produce ordinary SHUT-IN rows, are NOT recorded as failure events,
# and generate no workover invoice — exactly like real non-repair downtime.
# Most routine downtime is short, but a real minority runs for weeks: offset frac
# protection, pipeline curtailment, extended power loss, regulatory shut-in,
# waiting on a rig. That long tail is REQUIRED, not cosmetic. With short-only
# outages (the first cut of this fix) clean wells topped out at 4 days while every
# failure ran 6+, so "any outage >= 5 days" was still a perfect failure detector
# and "any well with a long outage" still identified the failing cohort exactly.
ROUTINE_OUTAGES_PER_YEAR = (1.0, 2.4)
ROUTINE_SHORT_DAYS       = (1, 4)
ROUTINE_LONG_DAYS        = (6, 26)   # spans the failure shut-in band on purpose
ROUTINE_LONG_FRAC        = 0.28
# Electrical model: 480 V 3-phase, 0.85 power factor
VOLTS=480.0; PF=0.85; SQRT3=1.7320508

# Supporting assets — volume/downtime driven, not a flat random draw.
SUPP_FIXED = {                 # $/month fixed base by asset type
    "Facility":     (3200, 5400),
    "Compressor":   (3800, 6200),
    "Tank Battery": (2200, 4000),
    "SWD Well":     (4200, 7000),
}
SUPP_RATES = {                 # variable rates, $ per unit of throughput
    # Oil/water/gas handling that used to be double-billed at the well now sits
    # here only. Rates raised to absorb what was taken out of well LOE.
    "Facility":     {"oil_bbl":1.05, "water_bbl":0.22, "gas_mcf":0.12},
    "Compressor":   {"gas_mcf":0.30, "fuel_mcf":1.20, "lost_runtime_hrs":45.0},
    "Tank Battery": {"oil_hauled_bbl":1.55, "water_hauled_bbl":0.60},
    "SWD Well":     {"water_disposed_bbl":0.42},
}
SUPP_EVENT_COST = {            # $ per excursion day rolled into the monthly bill
    "Facility":     900,       # BS&W out of spec
    "Compressor":   1400,      # runtime shortfall or engine overtemp
    "Tank Battery": 1200,      # high level alarm
    "SWD Well":     2200,      # injection pressure at the top of the range
}
SUPP_REPAIR_THRESHOLD = {      # excursion days in a month that trigger a repair
    "Facility":2, "Compressor":2, "Tank Battery":2, "SWD Well":2,
}
SUPP_REPAIR_RANGE = {
    "Facility":     (8000,  42000),
    "Compressor":   (12000, 85000),
    "Tank Battery": (6000,  30000),
    "SWD Well":     (15000,110000),
}
# A supporting asset carries for a WHOLE LEASE what a well LOE invoice carries for
# one well, so its monthly bill is legitimately an order of magnitude larger. The
# a 250k ceiling bound on nearly every invoice once oil/water/gas
# handling here, pinning cost flat and decoupling it from throughput. Sanity
# ceiling only — the build now fails if it binds.
SUPP_MAX_CAP = 2500000

# ── Synthetic price deck ──────────────────────────────────────────────────────
# SYNTHETIC but shaped to real Permian history so the regime changes an AI needs
# to see are present: the 2008 run-up and collapse, the H2-2014 crash, the 2018
# Midland differential blowout, April 2020, the 2022 spike, and the recurring
# Waha gas basis collapse (Waha genuinely printed negative monthly averages).
# Not actual market data — do not use for accounting or reserve reporting.

# WTI Cushing monthly ($/bbl). Dramatic years are shaped explicitly; calm years
# are generated as a normalized random walk around the annual anchor.
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

# Henry Hub annual anchors ($/mcf)
HH_ANNUAL = {2008:8.86,2009:3.94,2010:4.37,2011:4.00,2012:2.75,2013:3.73,2014:4.37,
             2015:2.62,2016:2.52,2017:2.99,2018:3.15,2019:2.56,2020:2.03,2021:3.89,
             2022:6.42,2023:2.54,2024:2.19,2025:3.40,2026:3.55}

# WTI Midland differential vs Cushing ($/bbl). Negative = Midland discount.
# 2018 blowout (takeaway constrained) then inversion to a premium once pipe came on.
MIDLAND_DIFF = {2008:-1.00,2009:-1.10,2010:-1.00,2011:-1.60,2012:-3.50,2013:-4.50,
                2014:-6.20,2015:-3.90,2016:-1.10,2017:-1.00,2018:-11.80,2019:-1.60,
                2020: 0.30,2021: 0.20,2022: 0.55,2023: 1.00,2024: 1.10,2025: 1.00,2026: 0.95}

# Waha basis vs Henry Hub ($/mcf). Deeply negative in the constrained years.
WAHA_BASIS = {2008:-0.35,2009:-0.40,2010:-0.35,2011:-0.45,2012:-0.50,2013:-0.55,
              2014:-0.60,2015:-0.45,2016:-0.40,2017:-0.75,2018:-1.65,2019:-3.05,
              2020:-1.25,2021:-0.55,2022:-1.55,2023:-1.85,2024:-2.30,2025:-1.05,2026:-0.95}

# Months where Waha realized a NEGATIVE monthly average (producers paid to flow gas).
WAHA_NEGATIVE_MONTHS = {
    "2019-03","2019-04","2019-09","2020-04","2020-05",
    "2023-04","2023-05","2024-03","2024-04","2024-05","2024-06",
}

# Texas fiscal terms
SEV_TAX_OIL   = 0.046    # 4.6% of market value
SEV_TAX_GAS   = 0.075    # 7.5% of market value
AD_VALOREM    = 0.012    # ~1.2% of value, county assessed
OIL_TRANSPORT = 2.35     # $/bbl gathering + trucking
GAS_GPT       = 0.62     # $/mcf gathering, processing, treating
NGL_YIELD_MIN = 55.0     # bbl NGL per MMcf wet gas
NGL_YIELD_MAX = 95.0
NGL_PCT_WTI   = 0.36     # NGL barrel realizes ~36% of WTI

def _monthly_from_annual(anchor, rng, vol=0.075):
    """12 monthly values as a random walk normalized to the annual mean."""
    w=[]; x=1.0
    for _ in range(12):
        x*= (1.0+rng.gauss(0,vol)); w.append(x)
    m=sum(w)/12.0
    return [anchor*v/m for v in w]

def build_price_deck():
    """Monthly realized-price deck keyed 'YYYY-MM'. Deterministic under the seed."""
    rng=random.Random(20260708)
    deck={}
    for yr in range(2008,2027):
        wti = WTI_MONTHLY.get(yr) or _monthly_from_annual(WTI_ANNUAL[yr], rng, 0.085)
        hh  = _monthly_from_annual(HH_ANNUAL[yr], rng, 0.110)
        for i in range(12):
            key=f"{yr}-{i+1:02d}"
            # Midland realized oil: Cushing + differential, with monthly wobble.
            md = MIDLAND_DIFF[yr]*(1.0+rng.gauss(0,0.18))
            oil = max(1.00, wti[i] + md)
            # Waha realized gas: Henry Hub + basis. Forced negative in known windows.
            wb = WAHA_BASIS[yr]*(1.0+rng.gauss(0,0.22))
            gas = hh[i] + wb
            if key in WAHA_NEGATIVE_MONTHS:
                gas = -abs(rng.uniform(0.05,0.95))
            else:
                gas = max(0.04, gas)
            deck[key]={"wti_cushing_usd_bbl":round(wti[i],2),
                       "midland_differential_usd_bbl":round(md,2),
                       "realized_oil_usd_bbl":round(oil,2),
                       "henry_hub_usd_mcf":round(hh[i],3),
                       "waha_basis_usd_mcf":round(wb,3),
                       "realized_gas_usd_mcf":round(gas,3),
                       "realized_ngl_usd_bbl":round(wti[i]*NGL_PCT_WTI,2)}
    return deck

PRICE_DECK = build_price_deck()
# ── Helpers ───────────────────────────────────────────────────────────────────
def cl(v,lo,hi):
    # PRECONDITION: lo <= hi. Called ~15M times per build, so it is deliberately
    # branch-free rather than validating. If a retuned constant ever inverts the
    # bounds this returns lo and silently discards the ordering instead of
    # raising — check any new clamp bounds by hand.
    return max(lo,min(hi,v))
def nx(v,p=0.025): return v*(1+random.gauss(0,p))
def rd(s,e): return s+timedelta(days=random.randint(0,(e-s).days))
def hyp(qi,di,b,t):
    if b==0: return qi*np.exp(-di*t)
    return qi/((1+b*di*t)**(1.0/b))

def gor_mult(t):
    """GOR climbs as reservoir pressure depletes."""
    return 1.0 + 1.6*(t/6200.0)

def drift(t, rate):
    """Slow reservoir-depletion drift on a healthy baseline. rate = frac change over 17yr."""
    return 1.0 + rate*(t/6200.0)

def late_mult(t, cut, floor, ramp):
    """Smooth late-life decline. Replaces the old overnight step change."""
    if t<=cut: return 1.0
    return max(floor, 1.0-(1.0-floor)*min(1.0,(t-cut)/ramp))

def sensor_next(lift, current, sensor_state, age_yrs):
    """Choose next failure sub-type based on current sensor state and well age."""
    if lift == "Pumping Unit":
        cas  = sensor_state.get('cas', 100)
        fill = sensor_state.get('fill', 80)
        spm  = sensor_state.get('spm', 8)
        # Corrosion signature: elevated casing pressure
        if cas > 200 and age_yrs > 6:
            return random.choice([t for t in PU_TYPES if "Corrosion" in t[0]])
        # Low fillage: worn pump or stuck
        elif fill < 50 and age_yrs > 4:
            return random.choice([t for t in PU_TYPES if "Worn" in t[0] or "Stuck" in t[0]])
        # Vibrating spm: rod vibration
        elif spm > 12:
            return random.choice([t for t in PU_TYPES if "Vibration" in t[0]])
        else:
            opts=[t for t in PU_TYPES if t[0]!=current]
            return random.choice(opts)
    elif lift == "ESP":
        temp = sensor_state.get('temp', 200)
        pip  = sensor_state.get('pip', 400)
        if temp > 230 and age_yrs > 5:
            return random.choice([t for t in ESP_TYPES if "Overtemp" in t[0]])
        elif pip < 300:
            return random.choice([t for t in ESP_TYPES if "Cavitation" in t[0]])
        else:
            opts=[t for t in ESP_TYPES if t[0]!=current]
            return random.choice(opts)
    else:  # Gas Lift
        inj  = sensor_state.get('inj', 500)
        cas  = sensor_state.get('cas', 1000)
        if inj > 900:
            return random.choice([t for t in GL_TYPES if "Stuck Open" in t[0] or "Washout" in t[0]])
        elif inj < 100:
            return random.choice([t for t in GL_TYPES if "Stuck Closed" in t[0]])
        else:
            opts=[t for t in GL_TYPES if t[0]!=current]
            return random.choice(opts)

def routine_outages(sd, ed):
    """
    Ordinary non-failure downtime. {start_date: duration_days}.
    Weather, power loss, offset frac hits, facility upsets, pumper error,
    scheduled surface work. Applied to EVERY well including the clean cohort, so
    100% uptime no longer identifies the healthy wells. These are not failure
    events and never generate a workover invoice.
    """
    out={}; total=(ed-sd).days
    if total<=0: return out
    n=max(1,int(round(total/365.25*random.uniform(*ROUTINE_OUTAGES_PER_YEAR))))
    # The long tail is allocated by CONSTRUCTION, not by an independent coin flip
    # per outage. Left to chance, a well can draw zero long outages over its whole
    # life — and since every failing well has long outages by definition, a well
    # with none is identifiably clean. Every well gets at least two.
    n_long=min(n,max(2,int(round(n*ROUTINE_LONG_FRAC))))
    kinds=[ROUTINE_LONG_DAYS]*n_long+[ROUTINE_SHORT_DAYS]*(n-n_long)
    random.shuffle(kinds)
    for rng in kinds:
        st=sd+timedelta(days=random.randint(0,max(0,total-1)))
        out[st.strftime('%Y-%m-%d')]=random.randint(*rng)
    return out

def _pearson(xs,ys):
    """Pearson r. Module level so every verification block can reach it."""
    n=len(xs)
    if n<3: return 0.0
    mx=sum(xs)/n; my=sum(ys)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    sxx=sum((a-mx)**2 for a in xs); syy=sum((b-my)**2 for b in ys)
    return sxy/((sxx*syy)**0.5) if sxx>0 and syy>0 else 0.0

def stamp_rows(history):
    """Row-level organization_id/timestamp, matching Facility/Compressor/SWD."""
    for r in history:
        r["organization_id"]=ORGANIZATION_ID
        r["timestamp"]=r["date"]+" 00:00:00"
    return history

def make_pump_curve(avg_oil):
    m = PUMP_MODELS[0] if avg_oil>1200 else PUMP_MODELS[1] if avg_oil>600 else PUMP_MODELS[2]
    a=m["poly"]["a"]; b=m["poly"]["b"]; c=m["poly"]["c"]
    curve=[]
    for q in [m["min_bpd"],m["bep_bpd"]*0.7,m["bep_bpd"],m["bep_bpd"]*1.2,m["max_bpd"]]:
        h=max(5.0,a*q**2+b*q+c)
        eff=cl(85*(1-((q-m["bep_bpd"])/(m["bep_bpd"]*0.8))**2),20,90)
        bhp=round((q*42/1440*h)/(3960*(eff/100)),3)
        curve.append({"flow_rate_bpd":round(q,0),"head_ft_per_stage":round(h,2),
                      "efficiency_percent":round(eff,1),
                      "brake_horsepower_per_stage":bhp,
                      "brake_horsepower_total":round(bhp*m["stages"],1)})
    return {"manufacturer":m["manufacturer"],"model":m["model"],"stages":m["stages"],
            "baseline_frequency_hz":60,"min_operating_flow_bpd":m["min_bpd"],
            "max_operating_flow_bpd":m["max_bpd"],"best_efficiency_point_bpd":m["bep_bpd"],
            "polynomial_coefficients":{"head_curve":m["poly"]},"curve_data_60hz":curve}

# ── PU history ────────────────────────────────────────────────────────────────
def gen_pu(aid, spud, is_clean, lateral_ft=0):
    """
    Permian rod pump well — Pumping Unit history.
    Sub-types drive sensor physics; not stored in output.
    PU fields: 20 total including new Permian-specific fields.
    """
    # Trimmed at the top end: a rod pump lifting much beyond ~450 bopd of oil is
    # outside normal beam-pump capacity even at high fluid volumes.
    qi=random.uniform(15,215); di=random.uniform(0.001,0.006); b=random.uniform(0.7,1.4)
    # Lateral length drives rate and decline; geometry is no longer cosmetic.
    _qm,_dm,_bb=geom_params(lateral_ft); qi*=_qm; di*=_dm; b=min(1.9,b+_bb)
    wc_=random.uniform(0.45,0.90); gor=random.uniform(100,600)
    # Water on its own curve: starts as a ratio of initial oil, grows, declines slowly
    wqi=qi*(wc_/(1-wc_))*random.uniform(0.55,0.85); wgrow=random.uniform(1.2,3.0)
    spm0=random.uniform(4,14); stroke0=random.uniform(48,120)
    fill0=random.uniform(60,92); pip0=random.uniform(80,900)
    cas0=random.uniform(30,200); tub0=random.uniform(20,120)
    amps0=random.uniform(15,65); temp0=random.uniform(110,185)
    rt0=random.uniform(18,23); cb_eff0=random.uniform(85,99)
    gas_rate0=random.uniform(0.5,8.0)  # casing annulus gas rate — H2S/CO2 indicator
    lc_cut=random.randint(3600,5100); lc_ramp=random.randint(900,1800)  # per-well late-life onset
    # Reservoir-depletion drift on healthy baselines — applies to clean and failing alike
    dr_fill=random.uniform(-0.30,-0.10); dr_cas=random.uniform(0.15,0.55)
    dr_pip=random.uniform(-0.35,-0.12)

    sc=random.choice(PU_TYPES)
    scenario=sc[0]; _,det_min,det_max,si_min,si_max=sc
    fwin=random.randint(det_min,det_max)
    next_fail=random.randint(200,500); detr=0; done=False

    history=[]; events=[]; current=spud; t=0; status="PRODUCING"; downtime=0
    pending_si=0; si_is_fail=False; si_len=0; routine=routine_outages(spud,END_DATE)
    fb=fill0; pb=pip0; cb=cas0; rb=rt0
    fill_window=[]; sensor_state={'cas':cas0,'fill':fill0,'spm':spm0}

    while current<=END_DATE:
        _ds=current.strftime("%Y-%m-%d")
        # A failure that completed yesterday shuts the well in TODAY, so the
        # first shut-in row is a properly zeroed row. Previously status flipped to
        # SHUT-IN part-way through building the row, and the failure day was emitted
        # with run_status SHUT-IN alongside live amps, pressures and oil volume — an
        # unmissable marker for the exact failure date.
        if status=="PRODUCING" and pending_si>0:
            status="SHUT-IN"; downtime=pending_si; si_len=pending_si
            pending_si=0; si_is_fail=True
        # Ordinary non-failure downtime. No failure event, no workover invoice.
        elif status=="PRODUCING" and _ds in routine:
            status="SHUT-IN"; downtime=routine[_ds]; si_len=routine[_ds]; si_is_fail=False
        oil=hyp(qi,di,b,t)*late_mult(t,lc_cut,0.40,lc_ramp)
        water=wqi*(1.0+wgrow*min(t/6200.0,1.0))*(1.0/(1.0+0.15*di*t))
        gas=oil*gor*gor_mult(t)/1000
        spm=nx(spm0); stroke=nx(stroke0); fill=nx(fb*drift(t,dr_fill))
        pip=nx(pb*drift(t,dr_pip)); cas=nx(cb*drift(t,dr_cas)); tub=nx(tub0)
        amps=nx(amps0); temp=nx(temp0); rt=nx(rb,0.02)
        cb_eff=nx(cb_eff0,0.01)
        ann_gas=nx(gas_rate0,0.04)

        if status=="PRODUCING" and not is_clean:
            in_detr=not done and t>=next_fail
            if in_detr:
                detr+=1; frac=min(1.0,detr/fwin)

                # Sub-type physics — sensor deterioration only, no label stored
                if "Parted Rods - Mechanical" in scenario:
                    # Sudden — minimal pre-signal, stroke crashes near failure
                    if frac>0.85:
                        stroke=nx(stroke0*(1-frac*0.6)); stroke=cl(stroke,0,180)
                        spm=nx(spm0*(1-frac*0.5)); spm=cl(spm,0,20)
                        amps=cl(nx(amps0+frac*15),0,80)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Parted Rods - Corrosion" in scenario:
                    # H2S/CO2 signature — casing pressure rises first, fillage declines
                    cas=cl(nx(cb+frac*random.uniform(40,150)),0,1800)
                    fill=cl(nx(fb-frac*25),0,100)
                    ann_gas=cl(nx(gas_rate0*(1+frac*3.0)),0,50)  # H2S gas rising
                    amps=cl(nx(amps0+frac*8),0,80)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Worn Pump - Sand" in scenario:
                    # Abrasive wear — slow fillage decline, SPM steady
                    fill=cl(nx(fb-frac*40),0,100)
                    # Pump noise — amps slightly elevated, irregular
                    amps=cl(nx(amps0+frac*random.gauss(5,2)),0,80)
                    oil*=(1-frac*0.30)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Worn Pump - Scale" in scenario:
                    # Calcium carbonate — fillage drops but differently from sand
                    fill=cl(nx(fb-frac*35),0,100)
                    pip=cl(nx(pb+frac*200),0,5000)  # intake restricted
                    cb_eff=cl(nx(cb_eff0-frac*15),0,100)
                    oil*=(1-frac*0.25)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Stuck Pump - Sand" in scenario:
                    # Fast — SPM drops quickly, fillage zeroes
                    spm=cl(nx(spm0*(1-frac*0.9)),0,20)
                    fill=cl(nx(fb*(1-frac*0.95)),0,100)
                    amps=cl(nx(amps0+frac*20),0,80)  # motor straining
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Rod String Vibration" in scenario:
                    # Current and stroke oscillate — resonance
                    osc=np.sin(detr*0.3)*frac
                    amps=cl(nx(amps0+osc*15),0,80)
                    stroke=cl(nx(stroke0+osc*8),0,180)
                    spm=cl(nx(spm0+osc*2),0,20)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                sensor_state={'cas':cas,'fill':fill,'spm':spm}

            # Derived features computed AFTER failure physics so they carry the signal
            fill_window.append(fill)
            if len(fill_window)>7: fill_window.pop(0)
            fill_trend=round((fill_window[-1]-fill_window[0])/7,3) if len(fill_window)==7 else 0.0
            rod_load=round(cl(amps/max(amps0,1),0,3),3)

            row={
                "date":                          current.strftime("%Y-%m-%d"),
                "run_status":                    status,
                "allocated_oil_bbl":             round(max(0,nx(oil)),2),
                "test_oil_bopd":                 round(max(0,nx(oil,0.06)),2),
                "gross_stroke_length_in":        round(cl(stroke,0,180),1),
                "net_stroke_length_in":          round(cl(stroke*fill/100*nx(1,0.02),0,180),1),
                "strokes_per_min":               round(cl(spm,0,20),2),
                "pump_fillage_pct":              round(cl(fill,0,100),1),
                "pump_fillage_trend_7day":       fill_trend,
                "casing_pressure_psi":           round(cl(cas,0,1800),1),
                "tubing_pressure_psi":           round(cl(tub,0,800),1),
                "motor_current_amps":            round(cl(amps,0,80),2),
                "freq_hz":                       round(cl(nx(60,0.01),30,75),1),
                "pump_intake_pressure_psi":      round(cl(pip,0,5000),1),
                "motor_temp_f":                  round(cl(temp,60,280),1),
                "daily_runtime_hrs":             round(cl(rt,0,24),1),
                "rod_load_ratio":                rod_load,
                "casing_annulus_gas_rate_mscfd": round(cl(ann_gas,0,50),3),
                "counterbalance_efficiency_pct": round(cl(cb_eff,0,100),1),
                "allocated_water_bbl":           round(max(0,nx(water)),2),
                "allocated_gas_mcf":             round(max(0,nx(gas)),3),
            }

        elif status=="SHUT-IN":
            downtime-=1
            row={k:0.0 for k in ["allocated_oil_bbl","test_oil_bopd","gross_stroke_length_in",
                "net_stroke_length_in","strokes_per_min","pump_fillage_pct","pump_fillage_trend_7day",
                "motor_current_amps","freq_hz","pump_intake_pressure_psi","daily_runtime_hrs",
                "rod_load_ratio","allocated_water_bbl","allocated_gas_mcf"]}
            row.update({
                "date":current.strftime("%Y-%m-%d"),"run_status":"SHUT-IN",
                "casing_pressure_psi":round(cl(nx(cas,0.05),0,1800),1),
                "tubing_pressure_psi":round(cl(nx(tub0,0.05),0,800),1),
                "motor_temp_f":round(cl(nx(90,0.03),60,200),1),
                "casing_annulus_gas_rate_mscfd":round(cl(nx(gas_rate0,0.05),0,50),3),
                "counterbalance_efficiency_pct":0.0,
            })
            if downtime<=0:
                status="PRODUCING"; fill_window=[]
                if si_is_fail:
                    detr=0; done=False
                    # Workover restores toward the original baseline, not a fresh random one
                    fb=cl(fb*random.uniform(0.90,1.10),55,92); pb=cl(pb*random.uniform(0.85,1.15),80,900)
                    cb=cl(cb*random.uniform(0.85,1.15),30,220); rb=cl(rb*random.uniform(0.97,1.03),18,23)
                    next_sc=sensor_next("Pumping Unit",scenario,sensor_state,t/365.0)
                    scenario=next_sc[0]; _,det_min,det_max,si_min,si_max=next_sc
                    fwin=random.randint(det_min,det_max); next_fail=t+random.randint(150,420)
                    sensor_state={'cas':cb,'fill':fb,'spm':spm0}
        else:
            # Clean well — healthy production
            fill_c=cl(nx(fill0*drift(t,dr_fill)),0,100); amps_c=cl(nx(amps0),0,80)
            fill_window.append(fill_c)
            if len(fill_window)>7: fill_window.pop(0)
            fill_trend=round((fill_window[-1]-fill_window[0])/7,3) if len(fill_window)==7 else 0.0
            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":"PRODUCING",
                "allocated_oil_bbl":round(max(0,nx(oil)),2),
                "test_oil_bopd":round(max(0,nx(oil,0.06)),2),
                "gross_stroke_length_in":round(cl(nx(stroke0),0,180),1),
                "net_stroke_length_in":round(cl(nx(stroke0)*fill_c/100,0,180),1),
                "strokes_per_min":round(cl(nx(spm0),0,20),2),
                "pump_fillage_pct":round(fill_c,1),
                "pump_fillage_trend_7day":fill_trend,
                "casing_pressure_psi":round(cl(nx(cas0*drift(t,dr_cas)),0,1800),1),
                "tubing_pressure_psi":round(cl(nx(tub0),0,800),1),
                "motor_current_amps":round(amps_c,2),
                "freq_hz":round(cl(nx(60,0.01),30,75),1),
                "pump_intake_pressure_psi":round(cl(nx(pip0*drift(t,dr_pip)),0,5000),1),
                "motor_temp_f":round(cl(nx(temp0),60,280),1),
                "daily_runtime_hrs":round(cl(nx(rt0,0.02),0,24),1),
                "rod_load_ratio":round(cl(amps_c/max(amps0,1),0,3),3),
                "casing_annulus_gas_rate_mscfd":round(cl(nx(gas_rate0),0,50),3),
                "counterbalance_efficiency_pct":round(cl(nx(cb_eff0,0.01),0,100),1),
                "allocated_water_bbl":round(max(0,nx(water)),2),
                "allocated_gas_mcf":round(max(0,nx(gas)),3),
            }

        # Record the shut-in event with the sub-type that actually caused it.
        # This list drives repair scope selection in the invoice builder. It is
        # NEVER written to the output JSON — see the label-scan check at the end.
        if (row['run_status']=='SHUT-IN' and si_is_fail
                and (not history or history[-1]['run_status']!='SHUT-IN')):
            events.append({'date':row['date'],'dur':si_len,'scenario':scenario})

        history.append(row); current+=timedelta(days=1); t+=1
    return history, events

# ── ESP history ───────────────────────────────────────────────────────────────
def gen_esp(aid, spud, is_clean, lateral_ft=0):
    """
    Permian Basin ESP well history.
    Sub-types: Motor Overtemp-Gradual/Rapid, Pump Cavitation-Gas/Scale
    New field: intake_temp_f (fluid temp at pump intake, distinct from motor temp)
    """
    # Vintage-calibrated. These wells spud 2008-2010: short laterals, small fracs.
    # A 2008-2010 Permian horizontal made 200-600 bopd IP30, not the 2,000-4,000
    # of a 2019+ Delaware well. ESP is selected here for total FLUID volume at
    # high water cut, not for oil rate.
    qi=random.uniform(85,430); di=random.uniform(0.0015,0.007); b=random.uniform(0.6,1.5)
    # Lateral length drives rate and decline; geometry is no longer cosmetic.
    _qm,_dm,_bb=geom_params(lateral_ft); qi*=_qm; di*=_dm; b=min(1.9,b+_bb)
    wc_=random.uniform(0.20,0.80); gor=random.uniform(300,3500)
    wqi=qi*(wc_/(1-wc_))*random.uniform(0.50,0.80); wgrow=random.uniform(1.5,3.5)
    pip0=random.uniform(150,800); disc0=random.uniform(1500,7000)
    amps0=random.uniform(40,200); temp0=random.uniform(160,220)
    intake_temp0=random.uniform(130,190)  # fluid temp at intake — lower than motor temp
    hz0=random.uniform(48,63); vib0=random.uniform(0.05,0.5)
    ftp0=random.uniform(150,600); acp0=random.uniform(50,400)
    lc_cut=random.randint(3000,4400); lc_ramp=random.randint(900,1800)
    dr_pip=random.uniform(-0.40,-0.15); dr_temp=random.uniform(0.05,0.20)
    dr_disc=random.uniform(-0.30,-0.10)

    sc=random.choice(ESP_TYPES)
    scenario=sc[0]; _,det_min,det_max,si_min,si_max=sc
    fwin=random.randint(det_min,det_max)
    next_fail=random.randint(200,500); detr=0; done=False

    history=[]; events=[]; current=spud; t=0; status="PRODUCING"; downtime=0
    pending_si=0; si_is_fail=False; si_len=0; routine=routine_outages(spud,END_DATE)
    pb=pip0; db=disc0; ab=amps0; tb=temp0; itb=intake_temp0
    sensor_state={'temp':temp0,'pip':pip0}

    while current<=END_DATE:
        _ds=current.strftime("%Y-%m-%d")
        # A failure that completed yesterday shuts the well in TODAY, so the
        # first shut-in row is a properly zeroed row. Previously status flipped to
        # SHUT-IN part-way through building the row, and the failure day was emitted
        # with run_status SHUT-IN alongside live amps, pressures and oil volume — an
        # unmissable marker for the exact failure date.
        if status=="PRODUCING" and pending_si>0:
            status="SHUT-IN"; downtime=pending_si; si_len=pending_si
            pending_si=0; si_is_fail=True
        # Ordinary non-failure downtime. No failure event, no workover invoice.
        elif status=="PRODUCING" and _ds in routine:
            status="SHUT-IN"; downtime=routine[_ds]; si_len=routine[_ds]; si_is_fail=False
        oil=hyp(qi,di,b,t)*late_mult(t,lc_cut,0.50,lc_ramp)
        water=wqi*(1.0+wgrow*min(t/6200.0,1.0))*(1.0/(1.0+0.15*di*t))
        gas=oil*gor*gor_mult(t)/1000
        pip=nx(pb*drift(t,dr_pip)); disc=nx(db*drift(t,dr_disc)); amps=nx(ab); temp=nx(tb*drift(t,dr_temp))
        intake_temp=nx(itb,0.02); hz=nx(hz0,0.01); vib=nx(vib0)
        ftp=nx(ftp0); acp=nx(acp0)

        if status=="PRODUCING" and not is_clean:
            in_detr=not done and t>=next_fail
            if in_detr:
                detr+=1; frac=min(1.0,detr/fwin)

                if "Motor Overtemp - Gradual" in scenario:
                    # Insulation wear — temp climbs steadily, amps erratic
                    temp=cl(tb+frac*random.uniform(50,110),100,350)
                    amps=cl(nx(ab+random.gauss(0,frac*20)),5,280)
                    intake_temp=cl(itb+frac*20,80,300)  # intake warms too
                    disc=cl(nx(db-frac*500),100,10000)
                    oil*=(1-frac*0.20)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Motor Overtemp - Rapid" in scenario:
                    # Fast temp spike — amps spike then crash
                    temp=cl(tb+frac*random.uniform(80,160),100,350)
                    amps=cl(nx(ab+frac*40),5,280)
                    vib=cl(nx(vib0+frac*1.5),0,5)
                    oil*=(1-frac*0.10)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Pump Cavitation - Gas" in scenario:
                    # Gas slugs — PIP drops, amps oscillate, vibration high
                    pip=cl(nx(pb-frac*random.uniform(100,350)),10,8000)
                    amps=cl(nx(ab+random.gauss(0,frac*12)),5,280)
                    vib=cl(nx(vib0+frac*2.0),0,5)
                    temp=cl(nx(tb+frac*25),100,350)
                    # intake temp drops with gas slugs
                    intake_temp=cl(itb-frac*20,80,300)
                    oil*=(1-frac*0.15)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Pump Cavitation - Scale" in scenario:
                    # Scale blocks intake — PIP drops slowly, discharge drops
                    pip=cl(nx(pb-frac*random.uniform(80,250)),10,8000)
                    disc=cl(nx(db-frac*random.uniform(600,1800)),100,10000)
                    amps=cl(nx(ab+frac*10),5,280)
                    intake_temp=cl(itb+frac*15,80,300)  # backs up, warms
                    oil*=(1-frac*0.22)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                sensor_state={'temp':temp,'pip':pip}

            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":status,
                "allocated_oil_bbl":round(max(0,nx(oil)),2),
                "test_oil_bopd":round(max(0,nx(oil,0.06)),2),
                "avg_tubing_pressure_psi":round(cl(ftp,30,2000),1),
                "avg_casing_pressure_psi":round(cl(acp,20,1500),1),
                "motor_temp_f":round(cl(temp,100,380),1),
                "intake_temp_f":round(cl(intake_temp,80,320),1),
                "motor_current_amps":round(cl(amps,5,280),2),
                "freq_hz":round(cl(hz,30,75),1),
                "pump_intake_pressure_psi":round(cl(pip,10,8000),1),
                "pump_discharge_pressure_psi":round(cl(disc,100,9500),1),
                "vibration_hz":round(cl(vib,0,5),3),
                "allocated_water_bbl":round(max(0,nx(water)),2),
                "allocated_gas_mcf":round(max(0,nx(gas)),3),
            }

        elif status=="SHUT-IN":
            downtime-=1
            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":"SHUT-IN",
                "allocated_oil_bbl":0.0,"test_oil_bopd":0.0,
                "avg_tubing_pressure_psi":0.0,
                "avg_casing_pressure_psi":round(cl(nx(acp0,0.05),20,800),1),
                "motor_temp_f":round(nx(90,0.03),1),"intake_temp_f":round(nx(85,0.03),1),
                "motor_current_amps":0.0,"freq_hz":0.0,
                "pump_intake_pressure_psi":0.0,"pump_discharge_pressure_psi":0.0,
                "vibration_hz":0.0,"allocated_water_bbl":0.0,"allocated_gas_mcf":0.0,
            }
            if downtime<=0:
                status="PRODUCING"
                if si_is_fail:
                    detr=0; done=False
                    pb=cl(pb*random.uniform(0.85,1.15),150,800); db=cl(db*random.uniform(0.88,1.12),1500,7000)
                    ab=cl(ab*random.uniform(0.90,1.10),40,200); tb=cl(tb*random.uniform(0.95,1.05),160,220)
                    itb=cl(itb*random.uniform(0.95,1.05),130,190)
                    next_sc=sensor_next("ESP",scenario,sensor_state,t/365.0)
                    scenario=next_sc[0]; _,det_min,det_max,si_min,si_max=next_sc
                    fwin=random.randint(det_min,det_max); next_fail=t+random.randint(150,420)
                    sensor_state={'temp':tb,'pip':pb}
        else:
            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":"PRODUCING",
                "allocated_oil_bbl":round(max(0,nx(oil)),2),
                "test_oil_bopd":round(max(0,nx(oil,0.06)),2),
                "avg_tubing_pressure_psi":round(cl(nx(ftp0),30,2000),1),
                "avg_casing_pressure_psi":round(cl(nx(acp0),20,1500),1),
                "motor_temp_f":round(cl(nx(temp0*drift(t,dr_temp)),100,380),1),
                "intake_temp_f":round(cl(nx(intake_temp0),80,320),1),
                "motor_current_amps":round(cl(nx(amps0),5,280),2),
                "freq_hz":round(cl(nx(hz0,0.01),30,75),1),
                "pump_intake_pressure_psi":round(cl(nx(pip0*drift(t,dr_pip)),10,8000),1),
                "pump_discharge_pressure_psi":round(cl(nx(disc0*drift(t,dr_disc)),100,9500),1),
                "vibration_hz":round(cl(nx(vib0),0,5),3),
                "allocated_water_bbl":round(max(0,nx(water)),2),
                "allocated_gas_mcf":round(max(0,nx(gas)),3),
            }

        # Record the shut-in event with the sub-type that actually caused it.
        # This list drives repair scope selection in the invoice builder. It is
        # NEVER written to the output JSON — see the label-scan check at the end.
        if (row['run_status']=='SHUT-IN' and si_is_fail
                and (not history or history[-1]['run_status']!='SHUT-IN')):
            events.append({'date':row['date'],'dur':si_len,'scenario':scenario})

        history.append(row); current+=timedelta(days=1); t+=1
    return history, events

# ── GL history ────────────────────────────────────────────────────────────────
def gen_gl(aid, spud, is_clean, lateral_ft=0):
    """
    Permian Basin Gas Lift well history.
    Sub-types: Mandrel Washout-Sand/Chemical, Valve Failure-Stuck Open/Closed
    New fields: casing_tubing_gap_psi, gor_scf_bbl
    """
    # Vintage-calibrated for 2008-2010 spuds — see the note in gen_esp.
    qi=random.uniform(40,290); di=random.uniform(0.001,0.005); b=random.uniform(0.7,1.4)
    # Lateral length drives rate and decline; geometry is no longer cosmetic.
    _qm,_dm,_bb=geom_params(lateral_ft); qi*=_qm; di*=_dm; b=min(1.9,b+_bb)
    wc_=random.uniform(0.25,0.75); gor=random.uniform(500,4000)
    wqi=qi*(wc_/(1-wc_))*random.uniform(0.50,0.80); wgrow=random.uniform(1.3,3.0)
    cas0=random.uniform(600,2000); tub0=random.uniform(100,500)
    inj0=random.uniform(100,900); injp0=random.uniform(800,2000)
    wht0=random.uniform(100,180); chk0=random.uniform(30,75)
    lc_cut=random.randint(3000,4400); lc_ramp=random.randint(900,1800)
    dr_cas=random.uniform(-0.30,-0.10); dr_inj=random.uniform(0.10,0.40)

    sc=random.choice(GL_TYPES)
    scenario=sc[0]; _,det_min,det_max,si_min,si_max=sc
    fwin=random.randint(det_min,det_max)
    next_fail=random.randint(200,500); detr=0; done=False

    history=[]; events=[]; current=spud; t=0; status="PRODUCING"; downtime=0
    pending_si=0; si_is_fail=False; si_len=0; routine=routine_outages(spud,END_DATE)
    cb=cas0; ib=inj0; ipb=injp0
    sensor_state={'inj':inj0,'cas':cas0}

    while current<=END_DATE:
        _ds=current.strftime("%Y-%m-%d")
        # A failure that completed yesterday shuts the well in TODAY, so the
        # first shut-in row is a properly zeroed row. Previously status flipped to
        # SHUT-IN part-way through building the row, and the failure day was emitted
        # with run_status SHUT-IN alongside live amps, pressures and oil volume — an
        # unmissable marker for the exact failure date.
        if status=="PRODUCING" and pending_si>0:
            status="SHUT-IN"; downtime=pending_si; si_len=pending_si
            pending_si=0; si_is_fail=True
        # Ordinary non-failure downtime. No failure event, no workover invoice.
        elif status=="PRODUCING" and _ds in routine:
            status="SHUT-IN"; downtime=routine[_ds]; si_len=routine[_ds]; si_is_fail=False
        oil=hyp(qi,di,b,t)*late_mult(t,lc_cut,0.45,lc_ramp)
        water=wqi*(1.0+wgrow*min(t/6200.0,1.0))*(1.0/(1.0+0.15*di*t))
        gas=oil*gor*gor_mult(t)/1000
        cas=nx(cb*drift(t,dr_cas)); tub=nx(tub0); inj=nx(ib*drift(t,dr_inj)); injp=nx(ipb)
        wht=nx(wht0); chk=nx(chk0,0.02)
        gap=cl(cas-tub,0,3000); gor_val=round(gas*1000/max(oil,0.1),0)

        if status=="PRODUCING" and not is_clean:
            in_detr=not done and t>=next_fail
            if in_detr:
                detr+=1; frac=min(1.0,detr/fwin)

                if "Mandrel Washout - Sand" in scenario:
                    # Faster erosion — injection rate creeps up, pressure drops
                    inj=cl(nx(ib+frac*random.uniform(80,350)),0,15000)
                    injp=cl(nx(ipb-frac*random.uniform(60,220)),0,3000)
                    oil*=(1-frac*0.18)
                    gap=cl(cas-tub,0,3000)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Mandrel Washout - Chemical" in scenario:
                    # Slower corrosion — same pattern but gentler over longer window
                    inj=cl(nx(ib+frac*random.uniform(40,180)),0,15000)
                    injp=cl(nx(ipb-frac*random.uniform(30,120)),0,3000)
                    # Casing pressure slightly elevated
                    cas=cl(nx(cb+frac*30),0,3000)
                    oil*=(1-frac*0.12)
                    gap=cl(cas-tub,0,3000)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Valve Failure - Stuck Open" in scenario:
                    # Injection stays high even as pressure drops — valve can't close
                    inj=cl(nx(ib*(1+frac*0.6)),0,15000)
                    injp=cl(nx(ipb-frac*150),0,3000)
                    # GOR climbs as over-injected gas passes
                    gor_val=round(gor_val*(1+frac*0.5),0)
                    oil*=(1-frac*0.10)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                elif "Valve Failure - Stuck Closed" in scenario:
                    # Injection drops toward zero — valve can't open
                    inj=cl(nx(ib*(1-frac*0.95)),0,15000)
                    injp=cl(nx(ipb+frac*80),0,3000)  # upstream pressure builds
                    # Production drops without lift
                    oil*=(1-frac*0.50)
                    wht=cl(nx(wht0-frac*20),30,300)
                    if detr>=fwin:
                        pending_si=random.randint(si_min,si_max); done=True

                gap=cl(cas-tub,0,3000); gor_val=round(gas*1000/max(oil,0.1),0)
                sensor_state={'inj':inj,'cas':cas}

            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":status,
                "allocated_oil_bbl":round(max(0,nx(oil)),2),
                "test_oil_bopd":round(max(0,nx(oil,0.06)),2),
                "avg_casing_pressure_psi":round(cl(cas,0,3000),1),
                "casing_injection_pressure_psi":round(cl(injp,0,3000),1),
                "tubing_pressure_psi":round(cl(tub,0,1500),1),
                "injection_rate_mscfd":round(cl(inj,0,15000),1),
                "wellhead_temp_f":round(cl(wht,30,300),1),
                "choke_position_pct":round(cl(chk,0,100),1),
                "casing_tubing_gap_psi":round(gap,1),
                "gor_scf_bbl":gor_val,
                "allocated_water_bbl":round(max(0,nx(water)),2),
                "allocated_gas_mcf":round(max(0,nx(gas)),3),
            }

        elif status=="SHUT-IN":
            downtime-=1
            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":"SHUT-IN",
                "allocated_oil_bbl":0.0,"test_oil_bopd":0.0,
                "avg_casing_pressure_psi":round(cl(nx(cas0*0.3),0,3000),1),
                "casing_injection_pressure_psi":0.0,
                "tubing_pressure_psi":round(cl(nx(tub0*0.6),0,1500),1),
                "injection_rate_mscfd":0.0,
                "wellhead_temp_f":round(cl(nx(70,0.05),30,200),1),
                "choke_position_pct":0.0,
                "casing_tubing_gap_psi":0.0,"gor_scf_bbl":0.0,
                "allocated_water_bbl":0.0,"allocated_gas_mcf":0.0,
            }
            if downtime<=0:
                status="PRODUCING"
                if si_is_fail:
                    detr=0; done=False
                    cb=cl(cb*random.uniform(0.90,1.10),600,2000); ib=cl(ib*random.uniform(0.85,1.15),100,900)
                    ipb=cl(ipb*random.uniform(0.92,1.08),800,2000)
                    next_sc=sensor_next("Gas Lift",scenario,sensor_state,t/365.0)
                    scenario=next_sc[0]; _,det_min,det_max,si_min,si_max=next_sc
                    fwin=random.randint(det_min,det_max); next_fail=t+random.randint(150,420)
                    sensor_state={'inj':ib,'cas':cb}
        else:
            row={
                "date":current.strftime("%Y-%m-%d"),"run_status":"PRODUCING",
                "allocated_oil_bbl":round(max(0,nx(oil)),2),
                "test_oil_bopd":round(max(0,nx(oil,0.06)),2),
                "avg_casing_pressure_psi":round(cl(nx(cas0*drift(t,dr_cas)),0,3000),1),
                "casing_injection_pressure_psi":round(cl(nx(injp0),0,3000),1),
                "tubing_pressure_psi":round(cl(nx(tub0),0,1500),1),
                "injection_rate_mscfd":round(cl(nx(inj0*drift(t,dr_inj)),0,15000),1),
                "wellhead_temp_f":round(cl(nx(wht0),30,300),1),
                "choke_position_pct":round(cl(nx(chk0,0.02),0,100),1),
                "casing_tubing_gap_psi":round(cl(cas-tub,0,3000),1),
                "gor_scf_bbl":round(gor_val*(1+random.gauss(0,0.05)),0),
                "allocated_water_bbl":round(max(0,nx(water)),2),
                "allocated_gas_mcf":round(max(0,nx(gas)),3),
            }

        # Record the shut-in event with the sub-type that actually caused it.
        # This list drives repair scope selection in the invoice builder. It is
        # NEVER written to the output JSON — see the label-scan check at the end.
        if (row['run_status']=='SHUT-IN' and si_is_fail
                and (not history or history[-1]['run_status']!='SHUT-IN')):
            events.append({'date':row['date'],'dur':si_len,'scenario':scenario})

        history.append(row); current+=timedelta(days=1); t+=1
    return history, events

# ── Supporting assets ─────────────────────────────────────────────────────────
def upset_days(sd, ed, per_year, dur_lo, dur_hi):
    """
    Schedule unplanned operational upsets across an asset's life and return
    {date_string: fraction_through_the_episode}. Supporting assets previously ran
    on tight noise with no excursions at all, so throughput was the only thing a
    cost model could respond to — there were no downtime or maintenance events in
    the data to bill against. These episodes are unlabeled: they exist only as
    sensor behaviour, exactly like the well failure modes.
    """
    out={}; total=(ed-sd).days
    n=max(1,int(round(total/365.25*per_year)))
    for _ in range(n):
        start=sd+timedelta(days=random.randint(0,max(0,total-1)))
        dur=random.randint(dur_lo,dur_hi)
        for k in range(dur):
            d=start+timedelta(days=k)
            if d>ed: break
            out[d.strftime('%Y-%m-%d')]=(k+1)/dur
    return out

def lease_daily(all_assets, lease):
    by={}
    for k,v in all_assets.items():
        if not k.startswith(lease) or v['type']!='Well': continue
        for d in v['history']:
            dt=d['date']
            if dt not in by: by[dt]={'oil':0,'water':0,'gas':0}
            by[dt]['oil']  +=d.get('allocated_oil_bbl',0)
            by[dt]['water']+=d.get('allocated_water_bbl',0)
            by[dt]['gas']  +=d.get('allocated_gas_mcf',0)
    return by

def make_facility(all_assets, lease, info, sd, ed):
    prod=lease_daily(all_assets,lease); h=[]; c=sd; wb=random.uniform(78,92); po=pw=pg=0.0
    # Emulsion / carryover upsets: BS&W out of spec, water transfer rate spikes.
    ups=upset_days(sd,ed,random.uniform(2.5,4.5),2,5)
    while c<=ed:
        dt=c.strftime('%Y-%m-%d'); dp=prod.get(dt,{'oil':0,'water':0,'gas':0})
        oil=dp['oil']; water=dp['water']; gas=dp['gas']
        wh=cl(nx(wb),74,96); p2=cl(nx(wh-random.uniform(5,10)),55,wh-5)
        p3=cl(nx(p2-random.uniform(5,10)),38,p2-5); gs=cl(nx(p3-random.uniform(5,12)),18,p3-5)
        t1=cl(nx(oil*0.15,0.08),0,6000); t2=cl(nx(oil*0.12,0.08),0,6000); t3=cl(nx(oil*0.10,0.08),0,6000)
        h.append({
            "date":dt,"wellhead_pressure_psi":round(wh,1),
            "two_phase_separator_pressure_psi":round(p2,1),
            "three_phase_separator_pressure_psi":round(p3,1),
            "gas_sales_line_pressure_psi":round(gs,1),
            "tester_1_pressure_psi":round(cl(nx(p2*0.9),30,p2),1),"tester_1_temp_f":round(cl(nx(142,0.04),90,220),1),
            "tester_1_oil_rate_bpd":round(t1,1),"tester_1_water_rate_bpd":round(cl(nx(t1*0.9,0.10),0,10000),1),
            "tester_1_gas_rate_mcfd":round(cl(nx(t1*0.35,0.10),0,5000),1),
            "tester_2_pressure_psi":round(cl(nx(p2*0.88),30,p2),1),"tester_2_temp_f":round(cl(nx(138,0.04),90,220),1),
            "tester_2_oil_rate_bpd":round(t2,1),"tester_2_water_rate_bpd":round(cl(nx(t2*0.9,0.10),0,10000),1),
            "tester_2_gas_rate_mcfd":round(cl(nx(t2*0.35,0.10),0,5000),1),
            "tester_3_pressure_psi":round(cl(nx(p2*0.85),30,p2),1),"tester_3_temp_f":round(cl(nx(135,0.04),90,220),1),
            "tester_3_oil_rate_bpd":round(t3,1),"tester_3_water_rate_bpd":round(cl(nx(t3*0.9,0.10),0,10000),1),
            "tester_3_gas_rate_mcfd":round(cl(nx(t3*0.35,0.10),0,5000),1),
            "prior_day_oil_production_bbl":round(po,2),"prior_day_water_production_bbl":round(pw,2),
            "prior_day_gas_production_mcf":round(pg,3),
            "sales_gas_rate_mscfd":round(cl(nx(gas*0.85,0.04),0,60000),2),
            "sales_gas_temp_f":round(cl(nx(90,0.03),55,125),1),
            "sales_gas_pressure_psi":round(cl(nx(gs*1.1,0.02),gs,900),1),
            "custody_transfer_rate_bblhr":round(cl(nx(oil/24,0.05),0,600),2),
            "custody_transfer_temp_f":round(cl(nx(108,0.03),70,145),1),
            "crude_gravity_api":round(cl(nx(38,0.02),28,52),1),
            "basic_sediment_water_pct":round(cl(nx(0.30,0.08)+(1.15+0.9*ups[dt] if dt in ups else 0.0),0.0,2.5),3),
            "produced_water_transfer_rate_bpd":round(cl(nx(water*0.9,0.04)*(1.35 if dt in ups else 1.0),0,30000),1),
            "water_transfer_discharge_psi":round(cl(nx(200,0.05),100,450),1),
            "produced_water_temp_f":round(cl(nx(115,0.03),80,155),1),
            "facility_oil_tank_level_ft":round(cl(nx(11,0.06),2,22),1),
            "facility_water_tank_level_ft":round(cl(nx(10,0.06),2,22),1),
            "run_status":"PRODUCING","organization_id":ORGANIZATION_ID,"timestamp":dt+" 00:00:00",
        })
        po=oil; pw=water; pg=gas; c+=timedelta(days=1)
    return h

def make_compressor(all_assets, lease, sd, ed):
    prod=lease_daily(all_assets,lease); h=[]; c=sd
    su=random.uniform(25,55); di=random.uniform(175,360); rp=random.uniform(850,1250); lu=random.uniform(40,60)
    # Overheat episodes: engine temp climbs and lube pressure sags for several days,
    # then the unit trips and is down for 1-4 days. Unlabeled — sensor behaviour only.
    heat=upset_days(sd,ed,random.uniform(3.0,5.0),4,9)
    down=upset_days(sd,ed,random.uniform(2.5,4.0),1,4)
    while c<=ed:
        dt=c.strftime('%Y-%m-%d'); gas=prod.get(dt,{}).get('gas',0)
        hf=heat.get(dt,0.0); is_down=dt in down
        if is_down:
            h.append({"date":dt,"compressor_suction_pressure_psi":round(cl(nx(su*1.6),12,80),1),
                      "compressor_suction_temp_f":round(cl(nx(85,0.04),55,130),1),
                      "compressor_discharge_pressure_psi":round(cl(nx(di*0.25),120,550),1),
                      "compressor_discharge_temp_f":round(cl(nx(150,0.04),150,310),1),
                      "gas_throughput_mscfd":0.0,
                      "compressor_engine_rpm":0.0,
                      "fuel_gas_rate_mcfd":0.0,
                      "compressor_engine_temp_f":round(cl(nx(145,0.04),140,270),1),
                      "compressor_lube_oil_pressure_psi":0.0,
                      "daily_runtime_hrs":0.0,
                      "run_status":"DOWN","organization_id":ORGANIZATION_ID,"timestamp":dt+" 00:00:00"})
        else:
            rt=cl(nx(22.5,0.02)-(hf*random.uniform(1.5,4.0)),0,24)
            h.append({"date":dt,"compressor_suction_pressure_psi":round(cl(nx(su),12,80),1),
                      "compressor_suction_temp_f":round(cl(nx(90,0.04),55,130),1),
                      "compressor_discharge_pressure_psi":round(cl(nx(di),120,550),1),
                      "compressor_discharge_temp_f":round(cl(nx(220,0.04)+hf*35,150,310),1),
                      "gas_throughput_mscfd":round(cl(nx(gas*0.9,0.05)*(1-0.15*hf),0,60000),1),
                      "compressor_engine_rpm":round(cl(nx(rp),700,1450),0),
                      "fuel_gas_rate_mcfd":round(cl(nx(gas*0.04,0.06)*(1+0.20*hf),0,2500),2),
                      "compressor_engine_temp_f":round(cl(nx(200,0.04)+hf*random.uniform(45,70),140,270),1),
                      "compressor_lube_oil_pressure_psi":round(cl(nx(lu)*(1-0.22*hf),28,78),1),
                      "daily_runtime_hrs":round(rt,1),
                      "run_status":"PRODUCING","organization_id":ORGANIZATION_ID,"timestamp":dt+" 00:00:00"})
        c+=timedelta(days=1)
    return h

def make_tank(all_assets, lease, sd, ed):
    """
    Tank battery driven by ACTUAL lease production (previously a random walk
    decoupled from the wells, which made run tickets ~30% of lease oil).

    Oil:   lease oil fills 3 stock tanks; a truck hauls a tank down to its heel
           when it reaches the haul trigger, and the hauled volume IS the run
           ticket. Run tickets therefore total ~100% of lease oil over life.
    Water: lease water enters the water tank, drains to the SWD by pipe, and the
           excess is trucked out as a disposal ticket.
    Alarm: high_level_alarm is a real setpoint (19 ft) below the 21 ft working
           maximum. Trucks are not always available, so levels genuinely back up
           and the alarm genuinely fires.
    """
    prod=lease_daily(all_assets,lease)
    peak_oil=max([v['oil'] for v in prod.values()] or [500.0])
    peak_wtr=max([v['water'] for v in prod.values()] or [500.0])

    SHELL_FT=24.0; USABLE_FT=21.0; ALARM_FT=19.0; TRIGGER_FT=17.0; HEEL_FT=2.0
    # Size the battery so 3 oil tanks hold ~2.5 days of peak production.
    oil_cap=max(500.0,peak_oil*2.5/3.0); wtr_cap=max(500.0,peak_wtr*1.6)
    obpf=oil_cap/USABLE_FT; wbpf=wtr_cap/USABLE_FT
    swd_pipe_frac=random.uniform(0.60,0.80)   # share of water piped to SWD

    oi=[HEEL_FT*obpf*random.uniform(1.0,2.5) for _ in range(3)]
    wi_=HEEL_FT*wbpf*random.uniform(1.0,2.5)
    gb=random.uniform(6,9)
    h=[]; c=sd
    while c<=ed:
        dt=c.strftime('%Y-%m-%d'); dp=prod.get(dt,{'oil':0,'water':0,'gas':0})
        oil=dp['oil']; water=dp['water']

        # Oil in, round-robin across the three stock tanks
        for i in range(3): oi[i]+=oil/3.0
        tix=[0.0,0.0,0.0]
        truck=random.random()>0.09          # truck unavailable ~9% of days
        for i in range(3):
            lvl=oi[i]/obpf
            if lvl>=TRIGGER_FT and truck:
                heel=HEEL_FT*obpf*random.uniform(0.9,1.2)
                tix[i]=max(0.0,oi[i]-heel); oi[i]=heel
            # Inventory is NOT clamped. Previously oi[i]=min(oi[i],USABLE_FT*obpf)
            # silently deleted barrels on overfill, so oil vanished from the mass
            # balance and the 90-110% run-ticket check still read green.
            # Volume now backs up in the tank and is hauled on a later day; only the
            # REPORTED level is capped at the shell height.

        # Water in; piped continuously to SWD, trucked when it backs up
        piped=water*swd_pipe_frac      # goes to the SWD by pipe
        wi_+=water-piped                # remainder lands in the water tank
        wtix=0.0
        wlvl=wi_/wbpf
        if wlvl>=TRIGGER_FT and truck:
            heel=HEEL_FT*wbpf*random.uniform(0.9,1.2)
            wtix=max(0.0,wi_-heel); wi_=heel
        # Same for water: inventory conserved, reported level capped.

        lv=[min(oi[0]/obpf,SHELL_FT),min(oi[1]/obpf,SHELL_FT),
            min(oi[2]/obpf,SHELL_FT),min(wi_/wbpf,SHELL_FT)]
        gb=cl(gb+random.gauss(0,0.25)+(0.03 if oil>0 else -0.05),1.5,18.0)

        h.append({"date":dt,
                  "ot_1_level_ft":round(lv[0],1),"ot_1_temperature_f":round(cl(nx(108,0.03),75,145),1),
                  "ot_1_run_ticket_bbl":round(tix[0],1),
                  "ot_2_level_ft":round(lv[1],1),"ot_2_temperature_f":round(cl(nx(105,0.03),75,145),1),
                  "ot_2_run_ticket_bbl":round(tix[1],1),
                  "ot_3_level_ft":round(lv[2],1),"ot_3_temperature_f":round(cl(nx(103,0.03),75,145),1),
                  "ot_3_run_ticket_bbl":round(tix[2],1),
                  "wt_1_level_ft":round(lv[3],1),"wt_1_temperature_f":round(cl(nx(98,0.03),65,135),1),
                  "wt_1_disposal_ticket_bbl":round(wtix,1),
                  # Published so the SWD disposes exactly the water that
                  # actually reaches it. Previously the SWD took ~95% of ALL lease
                  # water while the tank battery separately trucked the 20-40% that
                  # supposedly never went to the SWD, so total disposal exceeded
                  # total production.
                  "water_piped_to_swd_bbl":round(piped,1),
                  "gb_1_level_ft":round(gb,1),"gb_1_temperature_f":round(cl(nx(102,0.03),70,135),1),
                  "high_level_alarm":1 if any(v>ALARM_FT for v in lv) else 0,
                  "run_status":"PRODUCING","organization_id":ORGANIZATION_ID,
                  "timestamp":dt+" 00:00:00"})
        c+=timedelta(days=1)
    return h

def make_swd(all_assets, lease, info, sd, ed, tank_hist):
    """
    Feed is the water the tank battery actually sends: piped volume plus
    trucked disposal tickets. Capacity is sized to the lease so the well can keep
    up, and anything it cannot take on a plugged day carries as backlog rather
    than disappearing, so disposal totals reconcile to produced water over life.
    """
    feed={r['date']:(r.get('water_piped_to_swd_bbl',0.0)+r.get('wt_1_disposal_ticket_bbl',0.0))
          for r in tank_hist}
    peak=max(list(feed.values()) or [500.0])
    h=[]; c=sd; cum=0.0; backlog=0.0
    ib=max(500.0,peak*random.uniform(1.08,1.35))
    # Near-wellbore plugging episodes: injection pressure climbs toward the licensed
    # maximum and disposal rate falls off until the well is acidized. Unlabeled.
    plug=upset_days(sd,ed,random.uniform(2.0,3.5),6,16)
    while c<=ed:
        dt=c.strftime('%Y-%m-%d')
        pf=plug.get(dt,0.0)
        backlog+=feed.get(dt,0.0)
        d=min(backlog,cl(nx(ib,0.04)*(1-0.30*pf),0,60000))
        backlog-=d; cum+=d
        h.append({"date":dt,"disposal_rate_bpd":round(d,1),
                  "surface_injection_pressure_psi":round(cl(nx(1800,0.04)+pf*random.uniform(1150,1500),1000,3500),1),
                  "swdwellhead_pressure_psi":round(cl(nx(1750,0.04)+pf*random.uniform(1100,1450),950,3400),1),
                  "tubing_pressure_psi":round(cl(nx(1780,0.04)+pf*random.uniform(1100,1450),980,3450),1),
                  "casing_pressure_psi":round(cl(nx(130,0.05),50,500),1),
                  "injection_temp_f":round(cl(nx(98,0.03),70,135),1),
                  "cumulative_disposal_volume_bbl":round(cum,1),
                  "pump_speed_spm":round(cl(nx(45,0.03),20,75),1),"pump_rpm":round(cl(nx(400,0.03),200,650),0),
                  "injection_pump_suction_psi":round(cl(nx(50,0.04),20,110),1),
                  "injection_pump_discharge_psi":round(cl(nx(1850,0.04)+pf*random.uniform(1100,1500),1000,3600),1),
                  "lube_oil_pressure_psi":round(cl(nx(50,0.04),30,80),1),
                  "lube_oil_temp_f":round(cl(nx(180,0.03)+pf*8,130,230),1),
                  "vibration_high_freq_hz":round(cl(nx(0.9,0.08)*(1+0.35*pf),0,5),3),
                  "packing_leak_rate_gpm":round(cl(nx(0.02,0.10),0,0.5),3),
                  "lube_oil_volume_gal":round(cl(nx(13,0.04),5,22),1),
                  "run_status":"PRODUCING","organization_id":ORGANIZATION_ID,"timestamp":dt+" 00:00:00"})
        c+=timedelta(days=1)
    return h

# ── Audit hooks ───────────────────────────────────────────────────────────────
# In-process only, never serialized. WO_AUDIT pairs each failure-driven workover
# with the sub-type that caused it so the verification block can measure how much
# of the sub-type is recoverable from the billed scope.
WO_AUDIT=[]      # (scenario, scope_key, vendor)
PREV_AUDIT=[]    # (lift, scope_key, vendor)

# ── Invoice counter ───────────────────────────────────────────────────────────
inv_counter=defaultdict(int)
def next_inv(year):
    inv_counter[year]+=1
    return f"RBNR-{year}-{inv_counter[year]:05d}"

# ── Workover scopes ──────────────────────────────────────────────────────────
# Each scope carries: billed description, cost range, and the vendor DISCIPLINE
# qualified to perform it. Vendor is drawn from that discipline's roster, so the
# vendor on the invoice can always do the job on the invoice.
# Cost ranges still overlap heavily on purpose — see the de-labeling note below.
WO_SCOPES = {
    "rod_partial": ("Rod string inspection and partial replacement",     28000,  95000, "rod_pump"),
    "rod_full":    ("Rod string replacement and pump change-out",        45000, 140000, "rod_pump"),
    "pump_change": ("Pump change-out — tubing pulled",                   35000, 130000, "rig_general"),
    "tubing_leak": ("Tubing leak repair and re-run",                     22000,  88000, "rig_general"),
    "esp_change":  ("ESP change-out — cable and motor",                  70000, 180000, "esp"),
    "esp_pump":    ("ESP pump and intake replacement",                   55000, 150000, "esp"),
    "gl_valve":    ("Gas lift valve and mandrel replacement",            18000,  72000, "gas_lift"),
    "gl_survey":   ("Gas lift design review and valve resetting",        12000,  48000, "gas_lift"),
    "cleanout":    ("Wellbore cleanout and scale treatment",             15000,  65000, "stimulation"),
    "chem_treat":  ("Chemical squeeze and corrosion inhibitor treatment", 9000,  38000, "chemical"),
    "surface":     ("Surface equipment repair and recommission",         12000,  48000, "surface"),
}

# Failure sub-type → plausible repair scopes, weighted.
# The mapping is deliberately MANY-TO-MANY: every sub-type has at least three
# candidate scopes, and every scope is reachable from at least two sub-types, so
# the billed scope is now operationally consistent with the failure without
# becoming a one-to-one decode of it. Verification measures the residual
# recoverability and fails the build if any scope pins a single sub-type.
MODE_SCOPES = {
    # Pumping unit
    "Parted Rods - Mechanical":   [("rod_full",5),("rod_partial",3),("pump_change",2),("tubing_leak",1)],
    "Parted Rods - Corrosion":    [("rod_partial",4),("rod_full",3),("chem_treat",2),("tubing_leak",2),("cleanout",1)],
    "Worn Pump - Sand":           [("pump_change",5),("cleanout",3),("rod_partial",2),("tubing_leak",1)],
    "Worn Pump - Scale":          [("cleanout",4),("pump_change",4),("chem_treat",3),("rod_partial",1)],
    "Stuck Pump - Sand":          [("pump_change",5),("cleanout",3),("rod_full",2),("rod_partial",1)],
    "Rod String Vibration":       [("rod_partial",4),("surface",3),("rod_full",2),("pump_change",2)],
    # ESP
    "Motor Overtemp - Gradual":   [("esp_change",5),("esp_pump",3),("surface",2),("tubing_leak",1)],
    "Motor Overtemp - Rapid":     [("esp_change",5),("esp_pump",2),("surface",2),("tubing_leak",1)],
    "Pump Cavitation - Gas":      [("esp_pump",4),("esp_change",3),("surface",2),("tubing_leak",1)],
    "Pump Cavitation - Scale":    [("cleanout",4),("esp_pump",4),("esp_change",2),("chem_treat",2)],
    # Gas lift
    "Mandrel Washout - Sand":     [("gl_valve",5),("cleanout",2),("gl_survey",2),("tubing_leak",1)],
    "Mandrel Washout - Chemical": [("gl_valve",4),("chem_treat",3),("gl_survey",2),("tubing_leak",1)],
    "Valve Failure - Stuck Open": [("gl_valve",5),("gl_survey",3),("surface",1),("tubing_leak",1)],
    "Valve Failure - Stuck Closed":[("gl_valve",5),("gl_survey",3),("cleanout",1),("surface",1)],
}

# Preventive service is lift-appropriate but failure-independent. Drawn from the
# lighter scopes only, and priced at the bottom of each range so preventive spend
# overlaps the cheap end of failure-driven spend.
PREVENTIVE_SCOPES = {
    "Pumping Unit": ["rod_partial","pump_change","cleanout","chem_treat","surface"],
    "ESP":          ["esp_pump","cleanout","chem_treat","surface","tubing_leak"],
    "Gas Lift":     ["gl_survey","gl_valve","chem_treat","cleanout","surface"],
}

def _scope_vendor(key):
    return random.choice(WO_VENDORS_BY_DISCIPLINE[WO_SCOPES[key][3]])

def workover_scope(scenario, duration_days):
    """
    Pick the repair scope from the sub-type that ACTUALLY failed, then draw the
    vendor from the discipline qualified for that scope.

    Cost still comes from the scope's own range, and the ranges overlap across
    scopes, so cost alone does not recover the scope, the outage length, or the
    sub-type. Duration nudges the draw toward heavier or lighter work — a real
    effect, since the job determines how long the well is down — but does not
    determine it.
    """
    cands = None
    for mode, scopes in MODE_SCOPES.items():
        if mode in scenario:
            cands = scopes; break
    if cands is None:               # unmapped sub-type — fall back to light work
        cands = [("surface",1),("tubing_leak",1),("cleanout",1)]
    keys = [k for k,_ in cands]
    w = []
    for k,base in cands:
        lo = WO_SCOPES[k][1]
        heavy = lo >= 35000
        if duration_days > 16: w.append(base * (1.6 if heavy else 1.0))
        else:                  w.append(base * (1.0 if heavy else 1.4))
    key = random.choices(keys, weights=w, k=1)[0]
    desc, lo, hi, _ = WO_SCOPES[key]
    return desc, round(random.uniform(lo, hi), 2), _scope_vendor(key), key

def preventive_scope(lift):
    """Scheduled preventive service — clean wells get these too, so 'zero
    workovers' is not itself a label identifying the healthy cohort."""
    key = random.choice(PREVENTIVE_SCOPES[lift])
    desc, lo, hi, _ = WO_SCOPES[key]
    return desc, round(random.uniform(lo*0.45, lo*1.30), 2), _scope_vendor(key), key

def _kwh_day(row, lift):
    """Daily electrical consumption from the sensor record itself."""
    amps = row.get('motor_current_amps', 0.0) or 0.0
    if amps <= 0: return 0.0
    kw = amps * SQRT3 * VOLTS * PF / 1000.0
    if lift == "ESP":
        hrs = 24.0                                   # ESP runs continuously
    else:
        hrs = row.get('daily_runtime_hrs', 0.0) or 0.0
    return kw * hrs

def loe_cost(lift, agg, age_years):
    """
    Built-up monthly well LOE. Driven by oil, produced water, gas, metered
    electrical load, gas-lift injection volume, well age, and idle days — not by
    oil volume alone.
    """
    flo, fhi = LOE_FIXED_BY_LIFT[lift]
    fixed = random.uniform(flo, fhi) * (1.0 + LOE_AGE_ESC * age_years)
    var  = agg['oil']   * LOE_OIL_RATE   * random.uniform(0.88, 1.12)
    var += agg['water'] * LOE_WATER_RATE * random.uniform(0.85, 1.15)
    var += agg['gas']   * LOE_GAS_RATE   * random.uniform(0.90, 1.10)
    var += agg['kwh']   * LOE_POWER_USD_KWH * random.uniform(0.95, 1.05)
    if lift == "Gas Lift":
        var += agg['inj'] * LOE_GL_INJ_RATE * random.uniform(0.90, 1.10)
    idle = agg['idle_days'] * random.uniform(*LOE_IDLE_PER_DAY)
    return round(min(fixed + var + idle, LOE_MAX_CAP), 2)

def make_well_invoices(aid, history, wi, lift, events):
    """
    events is the internal failure-event list from the generator: date, outage
    length, and the sub-type that caused it. It is used to choose a repair scope
    that matches the failure, and is NOT emitted.
    """
    ledger=[]
    # Monthly aggregation of every LOE driver, not just oil.
    agg=defaultdict(lambda: {'oil':0.0,'water':0.0,'gas':0.0,'kwh':0.0,'inj':0.0,
                             'prod_days':0,'idle_days':0,'first':None})
    for d in history:
        mo=d['date'][:7]; a=agg[mo]
        if a['first'] is None: a['first']=d['date']
        if d['run_status']=='PRODUCING':
            a['prod_days']+=1
            a['oil']  +=d.get('allocated_oil_bbl',0.0)
            a['water']+=d.get('allocated_water_bbl',0.0)
            a['gas']  +=d.get('allocated_gas_mcf',0.0)
            a['kwh']  +=_kwh_day(d,lift)
            a['inj']  +=d.get('injection_rate_mscfd',0.0) or 0.0
        else:
            a['idle_days']+=1
    spud=datetime.strptime(history[0]['date'],'%Y-%m-%d') if history else None

    for mo in sorted(agg):
        a=agg[mo]
        # A fully idle month used to be skipped entirely. The MISSING
        # invoice was itself an unambiguous "this well was down all month" label.
        # Idle months now bill fixed cost plus standby, which is what actually
        # happens: the pumper still drives the route and the lease still costs money.
        inv_dt=a['first']; yr=int(inv_dt[:4])
        age=(datetime.strptime(inv_dt,'%Y-%m-%d')-spud).days/365.25
        g=loe_cost(lift,a,age)
        ledger.append({"invoice_id":next_inv(yr),"invoice_date":inv_dt,"category":"Monthly LOE",
                       "vendor":random.choice(LOE_VENDORS),
                       "service_description":f"Monthly lease operating expenses — {aid}",
                       "gross_cost_usd":g,"working_interest":wi,"total_usd":round(g*wi,2),
                       "production_bbl_month":round(a['oil'],1),
                       "water_bbl_month":round(a['water'],1),
                       "gas_mcf_month":round(a['gas'],1),
                       "electricity_kwh_month":round(a['kwh'],1)})

    # Workover invoices. Scope now follows the actual failure mode; vendor follows
    # the scope. Vendor billing still lags the job by 5-40 days, matching real AP
    # timing, so invoice_date is not an exact failure timestamp, and
    # shutin_duration_days is still deliberately NOT emitted.
    for ev in events:
        bill=(datetime.strptime(ev['date'],'%Y-%m-%d')+timedelta(days=random.randint(5,40)))
        if bill>END_DATE: bill=END_DATE
        bd=bill.strftime('%Y-%m-%d'); yr=int(bd[:4])
        desc,g,vendor,_key=workover_scope(ev['scenario'],ev['dur'])
        WO_AUDIT.append((ev['scenario'],_key,vendor))
        ledger.append({"invoice_id":next_inv(yr),"invoice_date":bd,"category":"Workover",
                       "vendor":vendor,"service_description":f"{desc} — {aid}",
                       "gross_cost_usd":g,"working_interest":wi,"total_usd":round(g*wi,2)})

    # Scheduled preventive service, independent of failures. Every well gets these,
    # so a well with no failure-driven workovers still shows workover-category spend.
    if history:
        sdt=datetime.strptime(history[0]['date'],'%Y-%m-%d')
        edt=datetime.strptime(history[-1]['date'],'%Y-%m-%d')
        cur=sdt+timedelta(days=random.randint(200,700))
        while cur<=edt:
            desc,g,vendor,_key=preventive_scope(lift); yr=cur.year
            PREV_AUDIT.append((lift,_key,vendor))
            ledger.append({"invoice_id":next_inv(yr),"invoice_date":cur.strftime('%Y-%m-%d'),
                           "category":"Workover","vendor":vendor,
                           "service_description":f"{desc} — {aid}",
                           "gross_cost_usd":g,"working_interest":wi,"total_usd":round(g*wi,2)})
            cur+=timedelta(days=random.randint(500,1100))
    ledger.sort(key=lambda r: r['invoice_date'])
    return ledger

def _supp_drivers(atype, row):
    """Per-day throughput and excursion flags read off the asset's own history."""
    if atype=="Facility":
        return ({"oil_bbl":  (row.get('custody_transfer_rate_bblhr',0.0) or 0.0)*24.0,
                 "water_bbl": row.get('produced_water_transfer_rate_bpd',0.0) or 0.0,
                 "gas_mcf":   row.get('sales_gas_rate_mscfd',0.0) or 0.0},
                1 if (row.get('basic_sediment_water_pct',0.0) or 0.0)>1.0 else 0)
    if atype=="Compressor":
        gas=row.get('gas_throughput_mscfd',0.0) or 0.0
        rt =row.get('daily_runtime_hrs',24.0) or 0.0
        exc=1 if (row.get('run_status')=='DOWN' or rt<21.0
                  or (row.get('compressor_engine_temp_f',0.0) or 0.0)>250.0) else 0
        return ({"gas_mcf":gas,"fuel_mcf":row.get('fuel_gas_rate_mcfd',0.0) or 0.0,
                 "lost_runtime_hrs":max(0.0,24.0-rt)}, exc)
    if atype=="Tank Battery":
        oil=sum(row.get(f'ot_{i}_run_ticket_bbl',0.0) or 0.0 for i in (1,2,3))
        return ({"oil_hauled_bbl":oil,
                 "water_hauled_bbl":row.get('wt_1_disposal_ticket_bbl',0.0) or 0.0},
                int(row.get('high_level_alarm',0) or 0))
    if atype=="SWD Well":
        return ({"water_disposed_bbl":row.get('disposal_rate_bpd',0.0) or 0.0},
                1 if (row.get('surface_injection_pressure_psi',0.0) or 0.0)>3000.0 else 0)
    return ({}, 0)

def make_supp_invoices(aid, atype, history, wi):
    """
    Supporting-asset spend is now driven by what the asset actually moved and how
    often it ran outside its normal band: throughput, hauled and disposed volume,
    runtime shortfall, and alarm days. Heavy excursion months also generate a
    separate repair invoice from a repair-qualified vendor.
    """
    ledger=[]
    fixed_lo,fixed_hi=SUPP_FIXED.get(atype,(3000,6000))
    rates=SUPP_RATES.get(atype,{})
    ev_cost=SUPP_EVENT_COST.get(atype,0)
    routine_v=SUPP_VENDORS.get((atype,"routine"),["Red Bluff Internal"])
    repair_v =SUPP_VENDORS.get((atype,"repair"), ["Red Bluff Internal"])

    agg=defaultdict(lambda: {"units":defaultdict(float),"exc":0,"first":None})
    for d in history:
        mo=d['date'][:7]; a=agg[mo]
        if a['first'] is None: a['first']=d['date']
        units,exc=_supp_drivers(atype,d)
        for k,v in units.items(): a['units'][k]+=v
        a['exc']+=exc

    for mo in sorted(agg):
        a=agg[mo]; inv_dt=a['first']; yr=int(inv_dt[:4])
        g=random.uniform(fixed_lo,fixed_hi)
        for k,rate in rates.items():
            g+=a['units'].get(k,0.0)*rate*random.uniform(0.90,1.10)
        g+=a['exc']*ev_cost*random.uniform(0.85,1.15)
        g=round(min(g,SUPP_MAX_CAP),2)
        rec={"invoice_id":next_inv(yr),"invoice_date":inv_dt,"category":"Monthly LOE",
             "vendor":random.choice(routine_v),
             "service_description":f"Monthly operating expenses — {aid}",
             "gross_cost_usd":g,"working_interest":wi,"total_usd":round(g*wi,2)}
        # Emit the throughput the bill was built from so the driver is auditable.
        for k,v in a['units'].items(): rec[f"{k}_month"]=round(v,1)
        ledger.append(rec)

        if a['exc']>=SUPP_REPAIR_THRESHOLD.get(atype,3):
            rg=round(random.uniform(*SUPP_REPAIR_RANGE.get(atype,(8000,55000)))
                     *(1.0+0.04*a['exc']),2)
            rd=(datetime.strptime(inv_dt,'%Y-%m-%d')+timedelta(days=random.randint(8,34)))
            if rd>END_DATE: rd=END_DATE
            ledger.append({"invoice_id":next_inv(rd.year),"invoice_date":rd.strftime('%Y-%m-%d'),
                           "category":"Maintenance","vendor":random.choice(repair_v),
                           "service_description":f"Corrective maintenance and recommission — {aid}",
                           "gross_cost_usd":rg,"working_interest":wi,"total_usd":round(rg*wi,2)})
    ledger.sort(key=lambda r: r['invoice_date'])
    return ledger

rev_counter=defaultdict(int)
def next_rev(year):
    rev_counter[year]+=1
    return f"RBNR-REV-{year}-{rev_counter[year]:05d}"

def make_revenue(aid, history, nri, ngl_yield):
    """
    Monthly revenue records. Gross volumes are valued at the realized Midland/Waha
    price, the operator's share is NRI (royalty burden already netted), then Texas
    severance, ad valorem, and midstream deducts come off to give net revenue.

    RECONCILING THE LEDGER:
      oil_volume_bbl   x realized_oil_usd_bbl = gross_oil_revenue_usd
      residue_gas_mcf  x realized_gas_usd_mcf = gross_gas_revenue_usd
      ngl_volume_bbl   x realized_ngl_usd_bbl = gross_ngl_revenue_usd
    Use residue_gas_mcf, NOT gas_volume_mcf, for the gas line. gas_volume_mcf is
    wet wellhead gas and is ~11-14% larger because the NGLs have not been
    extracted from it yet.

    The formula permits negative net revenue when Waha gas prices go negative, but
    in practice oil and NGL revenue dominate on these wells, so it does not occur
    in this build — the verification block reports the actual count. Gas revenue
    alone IS negative in the 11 negative-Waha months, which is the realistic
    behaviour worth learning from.
    """
    ledger=[]
    mo_oil=defaultdict(float); mo_gas=defaultdict(float); mo_wtr=defaultdict(float)
    for d in history:
        if d['run_status']!='PRODUCING': continue
        m=d['date'][:7]
        mo_oil[m]+=d.get('allocated_oil_bbl',0.0)
        mo_gas[m]+=d.get('allocated_gas_mcf',0.0)
        mo_wtr[m]+=d.get('allocated_water_bbl',0.0)
    # Every month the well existed gets a record, including fully idle
    # months (all zeros). A missing month used to flag a full-month outage.
    all_months=sorted({d['date'][:7] for d in history})
    for m in all_months:
        p=PRICE_DECK.get(m)
        if not p:
            # Silently skipping would drop the month from the revenue ledger, and a
            # missing month is exactly the full-outage label this build works to
            # remove. Fail loudly instead. Fires only if END_DATE is moved beyond
            # the deck's coverage (currently through 2026-12).
            raise SystemExit(f"price deck has no entry for {m} — extend build_price_deck()")
        yr=int(m[:4])
        o=mo_oil.get(m,0.0); g=mo_gas.get(m,0.0); w=mo_wtr.get(m,0.0)
        ngl_bbl=g/1000.0*ngl_yield              # bbl NGL from wet gas
        g_res=max(0.0,g-ngl_bbl*1.15)           # residue gas after NGL shrink

        # Value the volumes AS PUBLISHED. Pricing the full-precision floats and
        # then rounding the volumes separately leaves the ledger un-tie-out-able
        # on small records — a 0.16 bbl NGL line rounded from 0.1556 misprices by
        # ~3%. Rounding first makes volume x price == revenue exactly, at every
        # magnitude, which is what an analyst auditing the ledger expects.
        o=round(o,2); g_res=round(g_res,3); ngl_bbl=round(ngl_bbl,2)

        gr_oil=o*p["realized_oil_usd_bbl"]
        gr_gas=g_res*p["realized_gas_usd_mcf"]
        gr_ngl=ngl_bbl*p["realized_ngl_usd_bbl"]
        gross=gr_oil+gr_gas+gr_ngl

        n_oil=gr_oil*nri; n_gas=gr_gas*nri; n_ngl=gr_ngl*nri
        n_gross=n_oil+n_gas+n_ngl

        sev=n_oil*SEV_TAX_OIL+(n_gas+n_ngl)*SEV_TAX_GAS
        adval=max(0.0,n_gross)*AD_VALOREM
        gath=o*nri*OIL_TRANSPORT+g_res*nri*GAS_GPT
        net=n_gross-sev-adval-gath

        ledger.append({
            "revenue_id":next_rev(yr),"production_month":m,
            "oil_volume_bbl":round(o,2),"gas_volume_mcf":round(g,3),
            # Wellhead gas is WET. NGLs are extracted at the plant and the shrunk
            # residue stream is what gets sold at the gas price, so
            # gas_volume_mcf x realized_gas_usd_mcf does NOT equal gross gas
            # revenue. Residue volume is published so the ledger reconciles:
            #   residue_gas_mcf x realized_gas_usd_mcf = gross_gas_revenue_usd
            "residue_gas_mcf":round(g_res,3),
            "ngl_volume_bbl":round(ngl_bbl,2),"water_volume_bbl":round(w,2),
            "realized_oil_usd_bbl":p["realized_oil_usd_bbl"],
            "realized_gas_usd_mcf":p["realized_gas_usd_mcf"],
            "realized_ngl_usd_bbl":p["realized_ngl_usd_bbl"],
            "gross_oil_revenue_usd":round(gr_oil,2),
            "gross_gas_revenue_usd":round(gr_gas,2),
            "gross_ngl_revenue_usd":round(gr_ngl,2),
            "gross_revenue_usd":round(gross,2),
            "net_revenue_interest":nri,
            "net_revenue_before_deducts_usd":round(n_gross,2),
            "severance_tax_usd":round(sev,2),
            "ad_valorem_tax_usd":round(adval,2),
            "gathering_processing_usd":round(gath,2),
            "net_revenue_usd":round(net,2),
        })
    return ledger

# ═══════════════════════════════════════════════════════════════════════════════
# ── Build ─────────────────────────────────────────────────────────────────────
# Guarded so that importing this module gives you the configuration and the
# generator functions without running a ~70s build or writing a ~650 MB file.
# An `if` block creates no new scope, so every name below remains module-level
# exactly as before.
if __name__ == "__main__":
    print("="*60); print(f"Red Bluff Resources V2 — Build [{SCALE}]"); print("="*60)

    all_assets={}; well_num=1
    # In-process only. Never serialized — the label scan at the end enforces that.
    FAILURE_EVENTS={}
    _per=N_WELLS//len(LEASES)
    lease_base={lz:i*_per for i,lz in enumerate(LEASES)}

    for lease,info in LEASES.items():
        pool=[]
        for lift,count in info["dist"].items():
            pool.extend([lift]*count)
        random.shuffle(pool)

        # Wellbore geometry is a property of the well, not of the lift method.
        # Assign H/V per lease at the target horizontal fraction, weighting by lift
        # (horizontals skew ESP/GL; verticals skew rod pump) so the lift counts above
        # stay exact. Previously the suffix was derived FROM the lift, which made every
        # rod pump vertical and every ESP/GL horizontal — no horizontal rod pumps at all.
        n_horiz=round(len(pool)*HORIZ_FRAC[info["sub_basin"]])
        # Weighted sampling without replacement (Efraimidis-Spirakis): affinity biases the
        # draw but does not determine it, so every lift method gets both profiles.
        order=sorted(range(len(pool)),
                     key=lambda j: -(random.random()**(1.0/H_AFFINITY[pool[j]])))
        horiz_idx=set(order[:n_horiz])

        print(f"\n{lease} ({info['sub_basin']}): {n_horiz} horizontal / {len(pool)-n_horiz} vertical")
        for i,lift in enumerate(pool):
            base=lease_base[lease]
            well_no=base+i+1
            is_horiz=i in horiz_idx
            suffix="-H" if is_horiz else "-V"
            aid=f"{lease}-RB-{well_no:03d}{suffix}"
            lateral_ft=int(round(random.uniform(4500,10500),-2)) if is_horiz else 0
            spud=rd(START_MIN,START_MAX)
            is_clean=well_no in ALL_CLEAN_NUMS

            print(f"  [{well_num:>3}/100] {aid} ({lift}{'|CLEAN' if is_clean else ''})")

            if lift=="Pumping Unit":
                hist,fev=gen_pu(aid,spud,is_clean,lateral_ft)
            elif lift=="ESP":
                hist,fev=gen_esp(aid,spud,is_clean,lateral_ft)
            else:
                hist,fev=gen_gl(aid,spud,is_clean,lateral_ft)
            stamp_rows(hist)

            wi=info["wi"]; nri=info["nri"]
            ledger=make_well_invoices(aid,hist,wi,lift,fev)
            # Retained in-process only, for the scope/vendor consistency audit below.
            FAILURE_EVENTS[aid]=fev

            asset={
                "type":"Well","lift":lift,"organization_id":ORGANIZATION_ID,
                "wellbore_profile":"Horizontal" if is_horiz else "Vertical",
                "lateral_length_ft":lateral_ft,
                "area":info["sub_basin"],"working_interest":wi,"lease_type":info["lease_type"],
                "net_revenue_interest":nri,
                "location":{"basin":info["basin"],"sub_basin":info["sub_basin"],
                            "state":info["state"],"county":info["county"],"region":info["region"]},
                "assigned_personnel":{
                    "well_perf_engineer":{"name":info["engineer"],"role":"Well Performance Engineer",
                                           "email":f"{info['engineer'].lower().replace(' ','.')}@redbluffresources.com"},
                    "prod_technician":   {"name":f"{lease} Production Tech","role":"Production Technician",
                                           "email":f"prodtech.{lease.lower()}@redbluffresources.com"},
                },
                "financial_ledger":ledger,
                "revenue_ledger":make_revenue(aid,hist,nri,random.uniform(NGL_YIELD_MIN,NGL_YIELD_MAX)),
                "history":hist,
            }
            if lift=="ESP":
                prod=[d for d in hist if d['run_status']=='PRODUCING']
                gross=sorted(d.get('allocated_oil_bbl',0)+d.get('allocated_water_bbl',0) for d in prod)
                # Size on median lifetime gross liquid, not the first 30 days of peak
                med=gross[len(gross)//2] if gross else 0
                asset["pump_curve"]=make_pump_curve(med)

            all_assets[aid]=asset
            well_num+=1

    # Build supporting assets
    wells_only={k:v for k,v in all_assets.items() if v['type']=='Well'}
    all_dates=sorted(set(d['date'] for v in wells_only.values() for d in v['history']))
    sd=datetime.strptime(all_dates[0],'%Y-%m-%d')
    ed=datetime.strptime(all_dates[-1],'%Y-%m-%d')
    print(f"\nBuilding {4*len(LEASES)} supporting assets ({all_dates[0]} → {all_dates[-1]})...")

    SWD_SPECS={"Triplex":{"plunger_diameter_in":4.5,"stroke_length_in":12.0,"gear_ratio":5.0,"max_pressure_psi":3500},
               "HPS":    {"plunger_diameter_in":5.0,"stroke_length_in":14.0,"gear_ratio":4.5,"max_pressure_psi":4000}}

    for lease,info in LEASES.items():
        print(f"  {lease}...")
        wi=info["wi"]; lt=info["lease_type"]
        loc={"basin":info["basin"],"sub_basin":info["sub_basin"],"state":info["state"],
             "county":info["county"],"region":info["region"]}
        pers={"well_perf_engineer":{"name":info["engineer"],"role":"Well Performance Engineer",
                                     "email":f"{info['engineer'].lower().replace(' ','.')}@redbluffresources.com"},
              "prod_technician":   {"name":f"{lease} Production Tech","role":"Production Technician",
                                     "email":f"prodtech.{lease.lower()}@redbluffresources.com"}}
        base={"lift":None,"organization_id":ORGANIZATION_ID,"area":info["sub_basin"],
              "revenue_ledger":[],
              "working_interest":wi,"lease_type":lt,"net_revenue_interest":None,
              "location":loc,"assigned_personnel":pers}

        # Per-lease start. Using the global earliest date gave every lease's
        # facility, compressor, tank battery and SWD rows dated before that lease
        # had a single well — zero-throughput rows that read as downtime.
        _lw=[v for k,v in all_assets.items() if v['type']=='Well' and k.startswith(lease)]
        lsd=datetime.strptime(min(v['history'][0]['date'] for v in _lw),'%Y-%m-%d')
        fh=make_facility(all_assets,lease,info,lsd,ed)
        all_assets[f"{lease}-FAC-01"]={**base,"type":"Facility","financial_ledger":make_supp_invoices(f"{lease}-FAC-01","Facility",fh,wi),"history":fh}

        ch=make_compressor(all_assets,lease,lsd,ed)
        all_assets[f"{lease}-COMP-01"]={**base,"type":"Compressor","financial_ledger":make_supp_invoices(f"{lease}-COMP-01","Compressor",ch,wi),"history":ch}

        th=make_tank(all_assets,lease,lsd,ed)
        all_assets[f"{lease}-TANK-01"]={**base,"type":"Tank Battery","financial_ledger":make_supp_invoices(f"{lease}-TANK-01","Tank Battery",th,wi),"history":th}

        pt=info["swd_pump"]
        sh=make_swd(all_assets,lease,info,lsd,ed,th)
        all_assets[f"{lease}-SWD-01"]={**base,"type":"SWD Well","net_revenue_interest":info["nri"],
                                        "pump_type":pt,"pump_static_specs":SWD_SPECS[pt],
                                        "financial_ledger":make_supp_invoices(f"{lease}-SWD-01","SWD Well",sh,wi),"history":sh}

    # ── Final verification ────────────────────────────────────────────────────────
    total_h=sum(len(a['history']) for a in all_assets.values())
    wells_f={k:v for k,v in all_assets.items() if v['type']=='Well'}
    supp_f ={k:v for k,v in all_assets.items() if v['type']!='Well'}
    avg_days=sum(len(v['history']) for v in wells_f.values())/len(wells_f)
    max_date=max(d['date'] for v in all_assets.values() for d in v['history'])
    min_date=min(d['date'] for v in all_assets.values() for d in v['history'])

    # Failure counts come from the in-process event list. run_status
    # alone can no longer be used: it also carries ordinary non-failure downtime,
    # which is the entire point of routine downtime.
    fails=0; recent=0; last_dates=[]
    for k in wells_f:
        evs=FAILURE_EVENTS.get(k,[])
        fails+=len(evs)
        if evs:
            last=max(e['date'] for e in evs)
            last_dates.append(last)
            if last>='2024-01-01': recent+=1

    # The healthy cohort is defined by the build, not inferred from uptime.
    clean_keys={k for k in wells_f if int(k.split('-RB-')[1][:3]) in ALL_CLEAN_NUMS}
    clean_ct=len(clean_keys)
    clean_had_failure=sum(1 for k in clean_keys if FAILURE_EVENTS.get(k))
    # Clean wells must NOT be separable by uptime. Every well takes routine
    # outages, so both cohorts must show downtime days.
    _dn=lambda k: sum(1 for d in wells_f[k]['history'] if d['run_status']=='SHUT-IN')
    clean_zero_dt=sum(1 for k in clean_keys if _dn(k)==0)
    clean_dt_rate=(sum(_dn(k) for k in clean_keys)/
                   sum(len(wells_f[k]['history']) for k in clean_keys)) if clean_keys else 0.0
    total_inv=sum(len(a['financial_ledger']) for a in all_assets.values())
    all_ids=[i['invoice_id'] for a in all_assets.values() for i in a['financial_ledger']]
    dups=len(all_ids)-len(set(all_ids))
    wi_err=sum(1 for a in all_assets.values() for i in a['financial_ledger']
               if abs(i['total_usd']-round(i['gross_cost_usd']*i['working_interest'],2))>0.01)
    pump_ct=sum(1 for v in wells_f.values() if v.get('pump_curve'))
    esp_ct =sum(1 for v in wells_f.values() if v['lift']=='ESP')
    pres_ok=all(all(d['wellhead_pressure_psi']>d['two_phase_separator_pressure_psi']>
                    d['three_phase_separator_pressure_psi']>d['gas_sales_line_pressure_psi']
                    for d in all_assets[k]['history']) for k in all_assets if 'FAC' in k)
    LABEL_FIELDS=['failure_cause','failure_scenario','failure_type','failure_subtype',
                  'optimization_flag','shutin_duration_days','days_to_failure','is_clean',
                  'scenario','scope_key','failure_mode','repair_scope_key','idle_days',
                  'downtime_days']
    # Scan history AND financial_ledger — the ledger is where labels previously leaked.
    bad=[f for f in LABEL_FIELDS
         if any(f in d for v in all_assets.values() for d in v['history'])
         or any(f in i for v in all_assets.values() for i in v['financial_ledger'])]

    # Workover invoices must NOT coincide with shut-in start dates.
    wo_exact=0; wo_tot=0
    for _k,v in wells_f.items():
        starts={e['date'] for e in FAILURE_EVENTS.get(_k,[])}
        for i in v['financial_ledger']:
            if i['category']=='Workover':
                wo_tot+=1
                if i['invoice_date'] in starts: wo_exact+=1
    wo_leak_pct=(wo_exact/wo_tot*100) if wo_tot else 0.0

    # Clean wells must not be identifiable by absence of workover spend.
    clean_no_wo=sum(1 for k in clean_keys
                    if not any(i['category']=='Workover' for i in wells_f[k]['financial_ledger']))

    # A SHUT-IN row must be a genuinely dead row. This is the check that
    # would have caught the earlier bug where the failure day was emitted as SHUT-IN with
    # live amps, pressures and oil volume.
    LIVE_ON_SHUTIN=['allocated_oil_bbl','test_oil_bopd','motor_current_amps',
                    'strokes_per_min','injection_rate_mscfd','freq_hz']
    shutin_live=sum(1 for v in wells_f.values() for d in v['history']
                    if d['run_status']=='SHUT-IN'
                    and any((d.get(f,0.0) or 0.0)>0 for f in LIVE_ON_SHUTIN))

    # Every month a well existed must carry an LOE invoice and a revenue
    # record. A missing month used to label a full-month outage.
    months_no_loe=0; months_no_rev=0
    for v in wells_f.values():
        hm={d['date'][:7] for d in v['history']}
        lm={i['invoice_date'][:7] for i in v['financial_ledger'] if i['category']=='Monthly LOE'}
        rm={r['production_month'] for r in v['revenue_ledger']}
        months_no_loe+=len(hm-lm); months_no_rev+=len(hm-rm)

    # Geometry sanity: both profiles must appear on every lift method.
    geom=Counter((v['lift'],v['wellbore_profile']) for v in wells_f.values())
    geom_ok=all(geom.get((lf,p),0)>0 for lf in ['ESP','Gas Lift','Pumping Unit'] for p in ['Horizontal','Vertical'])
    horiz_ct=sum(1 for v in wells_f.values() if v['wellbore_profile']=='Horizontal')
    lat_ok=all((v['lateral_length_ft']>0)==(v['wellbore_profile']=='Horizontal') for v in wells_f.values())

    # Geometry must be visible in the production, not just stored as a field.
    def _peak(v):
        p=[d.get('allocated_oil_bbl',0.0) for d in v['history'][:365] if d['run_status']=='PRODUCING']
        return sum(sorted(p)[-30:])/30.0 if len(p)>=30 else 0.0
    # Per-lift H/V ratios are reported but NOT gated on: with 17 gas lift wells the
    # vertical cell is ~5 wells and the ratio is mostly noise (it has printed
    # anywhere from 1.6x to 5.7x across seeds on identical physics).
    # The gate is on two pooled statistics that actually have the sample size to
    # mean something, and that test the geometry claim directly:
    #   1. peak oil normalised by the lift's own vertical median, horizontals vs 1.0
    #   2. correlation between lateral_length_ft and that normalised peak
    geom_ratio={}; geom_cells={}; _vmed={}
    for lf in ['ESP','Gas Lift','Pumping Unit']:
        hz=[_peak(v) for v in wells_f.values() if v['lift']==lf and v['wellbore_profile']=='Horizontal']
        vt=sorted(_peak(v) for v in wells_f.values() if v['lift']==lf and v['wellbore_profile']=='Vertical')
        geom_cells[lf]=(len(hz),len(vt))
        _vmed[lf]=vt[len(vt)//2] if vt else 0.0
        geom_ratio[lf]=(sum(hz)/len(hz))/(sum(vt)/len(vt)) if hz and vt and sum(vt)>0 else 0.0
    _nx=[];_ny=[]
    for v in wells_f.values():
        if v['wellbore_profile']!='Horizontal': continue
        m=_vmed.get(v['lift'],0.0)
        if m<=0: continue
        _nx.append(v['lateral_length_ft']); _ny.append(_peak(v)/m)
    geom_pooled=(sum(_ny)/len(_ny)) if _ny else 0.0
    geom_lat_corr=_pearson(_nx,_ny) if len(_nx)>=10 else 0.0
    # Scale-aware: the 16-well smoke build cannot put 3 wells on both profiles of
    # every lift, so a hardcoded 3 failed the test scale outright.
    MIN_CELL=3 if SCALE=="full" else 1
    geom_cells_ok=all(min(c)>=MIN_CELL for c in geom_cells.values())
    geom_live=(geom_pooled>1.15 and geom_lat_corr>0.20 and geom_cells_ok)

    # Derived-feature liveness: rod_load_ratio must move with motor_current_amps.
    import statistics as _st
    _rl=[]
    for v in wells_f.values():
        if v['lift']!='Pumping Unit': continue
        p=[r for r in v['history'] if r['run_status']=='PRODUCING']
        if len(p)<500: continue
        hi=sorted(p,key=lambda r:-r['motor_current_amps'])[:50]
        _rl.append(_st.mean(r['rod_load_ratio'] for r in hi))
    rl_live=(_st.mean(_rl) if _rl else 0)
    # Fillage trend must be negative when fillage is collapsing.
    _ft=[]
    for v in wells_f.values():
        if v['lift']!='Pumping Unit': continue
        low=[r for r in v['history'] if r['run_status']=='PRODUCING' and r['pump_fillage_pct']<25]
        if len(low)>=20: _ft.append(_st.mean(r['pump_fillage_trend_7day'] for r in low))
    ft_neg=(sum(1 for x in _ft if x<0)/len(_ft)*100) if _ft else -1.0  # -1 = no sample

    # Tank battery must be coupled to production: run tickets should account for
    # essentially all lease oil (previously ~30%, a decoupled random walk).
    tank_ratio={}; alarm_tot=0; tank_rows=0
    for _lease in LEASES:
        _oil=sum(d.get('allocated_oil_bbl',0) for k,v in wells_f.items()
                 if k.startswith(_lease) for d in v['history'])
        _t=all_assets[f"{_lease}-TANK-01"]['history']
        _tix=sum(r['ot_1_run_ticket_bbl']+r['ot_2_run_ticket_bbl']+r['ot_3_run_ticket_bbl'] for r in _t)
        tank_ratio[_lease]=(_tix/_oil*100) if _oil else 0.0
        alarm_tot+=sum(r['high_level_alarm'] for r in _t); tank_rows+=len(_t)
    tank_lo=min(tank_ratio.values()); tank_hi=max(tank_ratio.values())
    tank_ok=90<=tank_lo and tank_hi<=110
    alarm_ok=0<alarm_tot<tank_rows*0.25

    # Oil must be conserved: run tickets must account for essentially all lease
    # oil. The old 90-110% band let 10% of it vanish unnoticed.
    # Water disposed must not exceed water produced.
    mb_oil={}; mb_wtr={}
    for _lease in LEASES:
        _o=sum(d.get('allocated_oil_bbl',0) for k,v in wells_f.items()
               if k.startswith(_lease) for d in v['history'])
        _w=sum(d.get('allocated_water_bbl',0) for k,v in wells_f.items()
               if k.startswith(_lease) for d in v['history'])
        _t=all_assets[f"{_lease}-TANK-01"]['history']
        _s=all_assets[f"{_lease}-SWD-01"]['history']
        _tix=sum(r['ot_1_run_ticket_bbl']+r['ot_2_run_ticket_bbl']+r['ot_3_run_ticket_bbl'] for r in _t)
        mb_oil[_lease]=(_tix/_o*100) if _o else 0.0
        _disp=sum(r['disposal_rate_bpd'] for r in _s)
        mb_wtr[_lease]=(_disp/_w*100) if _w else 0.0
    mb_oil_lo=min(mb_oil.values()); mb_oil_hi=max(mb_oil.values())
    mb_wtr_lo=min(mb_wtr.values()); mb_wtr_hi=max(mb_wtr.values())
    mb_oil_ok=98.0<=mb_oil_lo and mb_oil_hi<=102.0
    mb_wtr_ok=mb_wtr_hi<=100.5

    # Every row on every asset must carry organization_id and timestamp.
    rows_missing=sum(1 for v in all_assets.values() for d in v['history']
                     if 'organization_id' not in d or 'timestamp' not in d)

    # Revenue ledger integrity
    rev_rows=sum(len(v.get('revenue_ledger',[])) for v in all_assets.values())
    # The gas line must tie out against RESIDUE volume, not wet wellhead volume.
    # Before residue_gas_mcf was published, every gas-bearing record failed this
    # by 11-14% and the ledger could not be audited.
    _STREAMS=[('oil_volume_bbl','realized_oil_usd_bbl','gross_oil_revenue_usd'),
              ('residue_gas_mcf','realized_gas_usd_mcf','gross_gas_revenue_usd'),
              ('ngl_volume_bbl','realized_ngl_usd_bbl','gross_ngl_revenue_usd')]
    rev_recon_bad=0; rev_gas_rows=0
    for v in all_assets.values():
        for r in v.get('revenue_ledger',[]):
            if r.get('residue_gas_mcf',0)<=0: continue
            rev_gas_rows+=1
            for _vol,_pr,_rv in _STREAMS:
                if abs(r[_vol]*r[_pr]-r[_rv])>0.011: rev_recon_bad+=1; break
    rev_dups=0; rev_math=0; rev_nri=0; rev_months_missing=0
    _seen=set()
    for k,v in wells_f.items():
        prod_months={d['date'][:7] for d in v['history'] if d['run_status']=='PRODUCING'}
        rl=v['revenue_ledger']
        if len({r['production_month'] for r in rl})!=len(rl): rev_dups+=1
        rev_months_missing+=len(prod_months-{r['production_month'] for r in rl})
        for r in rl:
            if r['revenue_id'] in _seen: rev_dups+=1
            _seen.add(r['revenue_id'])
            exp=r['gross_oil_revenue_usd']+r['gross_gas_revenue_usd']+r['gross_ngl_revenue_usd']
            if abs(exp-r['gross_revenue_usd'])>0.05: rev_math+=1
            net=(r['net_revenue_before_deducts_usd']-r['severance_tax_usd']
                 -r['ad_valorem_tax_usd']-r['gathering_processing_usd'])
            if abs(net-r['net_revenue_usd'])>0.05: rev_math+=1
            if abs(r['net_revenue_before_deducts_usd']
                   -r['gross_revenue_usd']*r['net_revenue_interest'])>0.05: rev_nri+=1
    neg_gas_months=sum(1 for m,p in PRICE_DECK.items() if p['realized_gas_usd_mcf']<0)
    tot_net=sum(r['net_revenue_usd'] for v in wells_f.values() for r in v['revenue_ledger'])
    tot_cost=sum(i['total_usd'] for v in all_assets.values() for i in v['financial_ledger'])
    deck_ok=len(PRICE_DECK)==228 and neg_gas_months>0

    # ── Scope / vendor consistency (change 1, 2, 3) ───────────────────────────────
    DESC_TO_KEY={v[0]:k for k,v in WO_SCOPES.items()}
    vendor_mismatch=0; scope_unknown=0; wo_scope_ct=Counter()
    for v in all_assets.values():
        for i in v['financial_ledger']:
            if i['category']!='Workover': continue
            desc=i['service_description'].rsplit(' — ',1)[0]
            key=DESC_TO_KEY.get(desc)
            if key is None: scope_unknown+=1; continue
            wo_scope_ct[key]+=1
            if i['vendor'] not in WO_VENDORS_BY_DISCIPLINE[WO_SCOPES[key][3]]:
                vendor_mismatch+=1
    # Preventive scopes must be lift-appropriate.
    prev_lift_bad=sum(1 for lf,k,_ in PREV_AUDIT if k not in PREVENTIVE_SCOPES[lf])
    prev_vendor_bad=sum(1 for _,k,vd in PREV_AUDIT
                        if vd not in WO_VENDORS_BY_DISCIPLINE[WO_SCOPES[k][3]])

    # Scope must be operationally consistent with the failure without decoding it.
    _by_scope=defaultdict(Counter); _by_mode=defaultdict(set)
    for scen,key,_vd in WO_AUDIT:
        mode=next((m for m in MODE_SCOPES if m in scen), scen)
        _by_scope[key][mode]+=1; _by_mode[mode].add(key)
    _prev_by_scope=Counter(k for _,k,_ in PREV_AUDIT)
    scope_leak=0.0; scope_leak_key=None
    for key,c in _by_scope.items():
        share=max(c.values())/(sum(c.values())+_prev_by_scope.get(key,0))
        if share>scope_leak: scope_leak, scope_leak_key = share, key
    scopes_single_mode=[k for k,c in _by_scope.items() if len(c)<2]
    modes_single_scope=[m for m,ks in _by_mode.items() if len(ks)<3]

    # ── Supporting-asset spend must track its own throughput (change 4) ───────────
    # Correlate against total throughput moved, since more than one volume drives
    # the bill (a tank battery hauls both oil and water).
    SUPP_DRIVER_KEY={"Facility":["oil_bbl_month","water_bbl_month","gas_mcf_month"],
                     "Compressor":["gas_mcf_month","fuel_mcf_month"],
                     "Tank Battery":["oil_hauled_bbl_month","water_hauled_bbl_month"],
                     "SWD Well":["water_disposed_bbl_month"]}
    supp_corr={}
    for atype,dks in SUPP_DRIVER_KEY.items():
        xs=[];ys=[]
        for v in supp_f.values():
            if v['type']!=atype: continue
            for i in v['financial_ledger']:
                if i['category']=='Monthly LOE' and all(k in i for k in dks):
                    # rate-weighted throughput: a tank hauling cheap water and dear oil
                    # is not "decoupled" just because the two volumes differ in size
                    xs.append(sum(i[k]*SUPP_RATES[atype].get(k[:-6],1.0) for k in dks))
                    ys.append(i['gross_cost_usd'])
        supp_corr[atype]=_pearson(xs,ys)
    supp_corr_ok=all(c>0.5 for c in supp_corr.values())
    supp_flat=sum(1 for v in supp_f.values() for i in v['financial_ledger']
                  if i['category']=='Monthly LOE' and not any(k.endswith('_month') for k in i))
    maint_ct=sum(1 for v in supp_f.values() for i in v['financial_ledger']
                 if i['category']=='Maintenance')
    _supp_loe=[i for v in supp_f.values() for i in v['financial_ledger'] if i['category']=='Monthly LOE']
    supp_cap_hits=sum(1 for i in _supp_loe if abs(i['gross_cost_usd']-SUPP_MAX_CAP)<0.01)

    # Supporting-asset upsets must be present and bounded, and must produce repairs.
    supp_exc={}; supp_rows={}
    for atype in SUPP_DRIVER_KEY:
        e=0; r=0
        for v in supp_f.values():
            if v['type']!=atype: continue
            for d in v['history']:
                r+=1; _u,_x=_supp_drivers(atype,d); e+=_x
        supp_exc[atype]=e; supp_rows[atype]=r
    supp_exc_ok=all(0<supp_exc[a]<supp_rows[a]*0.20 for a in supp_exc)
    comp_down=sum(1 for v in supp_f.values() if v['type']=='Compressor'
                  for d in v['history'] if d['run_status']=='DOWN')

    # ── Well LOE must be more than a function of oil (change 5) ───────────────────
    _loe=[i for v in wells_f.values() for i in v['financial_ledger'] if i['category']=='Monthly LOE']
    loe_corr_oil  =_pearson([i['production_bbl_month'] for i in _loe],[i['gross_cost_usd'] for i in _loe])
    loe_corr_water=_pearson([i['water_bbl_month'] for i in _loe],[i['gross_cost_usd'] for i in _loe])
    loe_corr_kwh  =_pearson([i['electricity_kwh_month'] for i in _loe],[i['gross_cost_usd'] for i in _loe])
    # Partial signal test: within a narrow oil band, cost must still vary materially
    # with water and power. If oil were the only driver this spread would collapse.
    _band=[i for i in _loe if 900<=i['production_bbl_month']<=1100]
    if len(_band)>=30:
        _b=sorted(_band,key=lambda i:i['water_bbl_month']+i['electricity_kwh_month']*0.05)
        _q=max(1,len(_b)//4)
        _lo=sum(i['gross_cost_usd'] for i in _b[:_q])/_q
        _hi=sum(i['gross_cost_usd'] for i in _b[-_q:])/_q
        loe_band_ratio=_hi/_lo if _lo>0 else 0.0
    else:
        loe_band_ratio=0.0
    loe_ok=(loe_corr_water>0.25 and loe_corr_kwh>0.25 and loe_band_ratio>1.25)
    loe_cap_hits=sum(1 for i in _loe if abs(i['gross_cost_usd']-LOE_MAX_CAP)<0.01)
    loe_med=sorted(i['gross_cost_usd'] for i in _loe)[len(_loe)//2] if _loe else 0

    # ── Outage length must NOT decode failure vs routine downtime ────────────────
    # An earlier cut gave clean wells downtime but only SHORT downtime, so
    # every outage >= 5 days was a failure and the cohort was still perfectly
    # separable. This measures the actual separation instead of assuming it away.
    _len_fail=Counter(); _len_rout=Counter(); _clean_long=0
    for _k,_v in wells_f.items():
        _fd={e['date'] for e in FAILURE_EVENTS.get(_k,[])}
        _run=0; _start=None; _has_long=False
        for _d in _v['history']:
            if _d['run_status']=='SHUT-IN':
                if _run==0: _start=_d['date']
                _run+=1
            elif _run:
                (_len_fail if _start in _fd else _len_rout)[_run]+=1
                if _start not in _fd and _run>=6: _has_long=True
                _run=0
        if _run:
            (_len_fail if _start in _fd else _len_rout)[_run]+=1
            if _start not in _fd and _run>=6: _has_long=True
        if _k in clean_keys and _has_long: _clean_long+=1
    _nf=sum(_len_fail.values()); _nr=sum(_len_rout.values())
    # Kolmogorov-Smirnov style: the best single length threshold an attacker could use
    outage_ks=0.0; outage_ks_t=0
    if _nf and _nr:
        for _T in range(1,41):
            _a=sum(v for k,v in _len_fail.items() if k>=_T)/_nf
            _b=sum(v for k,v in _len_rout.items() if k>=_T)/_nr
            if abs(_a-_b)>outage_ks: outage_ks, outage_ks_t = abs(_a-_b), _T
    routine_long_share=(sum(v for k,v in _len_rout.items() if k>=6)/_nr) if _nr else 0.0
    clean_long_ok=(_clean_long==len(clean_keys))

    fpw=fails/len(wells_f)
    neg=sum(1 for v in wells_f.values() for d in v['history']
            for k,val in d.items() if k in ['allocated_oil_bbl','motor_current_amps','pump_intake_pressure_psi',
            'pump_fillage_pct','strokes_per_min'] and isinstance(val,(int,float)) and val<0)

    lift_ct=Counter(v['lift'] for v in wells_f.values())

    print(f"\n{'='*60}")
    print(f"FINAL VERIFICATION")
    print(f"{'='*60}")
    MIN_INV=20000 if SCALE=="full" else 100
    print(f"  Total assets:       {len(all_assets)}/{N_ASSETS} {'✓' if len(all_assets)==N_ASSETS else '⚠'}")
    print(f"  Wells:              {len(wells_f)}/{N_WELLS} {'✓' if len(wells_f)==N_WELLS else '⚠'}")
    print(f"  Supporting:         {len(supp_f)}/{4*len(LEASES)} {'✓' if len(supp_f)==4*len(LEASES) else '⚠'}")
    print(f"  Lift mix:           {dict(lift_ct)}")
    print(f"  Total daily rows:   {total_h:,}")
    print(f"  Avg years/well:     {avg_days/365:.1f}")
    print(f"  Date span:          {min_date} → {max_date} {'✓' if max_date==END_DATE.strftime('%Y-%m-%d') else '⚠'}")
    print(f"  Failure events:     {fails} ({fails/len(wells_f):.1f}/well)")
    _den=len(wells_f)-clean_ct
    print(f"  Post-2024 failures: {recent}/{_den} {'✓' if recent>=int(_den*0.85) else '⚠'}")
    print(f"  Latest failure:     {max(last_dates) if last_dates else None}")
    print(f"  Clean wells:        {clean_ct}/{N_CLEAN} {'✓' if clean_ct==N_CLEAN else '⚠'}")
    print(f"  Clean w/ failures:  {clean_had_failure} {'✓' if clean_had_failure==0 else '⚠ physics leaked'}")
    print(f"  Clean w/ 0 downtime:{clean_zero_dt} {'✓' if clean_zero_dt==0 else '⚠ UPTIME LABELS COHORT'}")
    print(f"  Clean downtime rate:{clean_dt_rate*100:.2f}% of days")
    print(f"  Outage len separability: KS {outage_ks:.2f} at >={outage_ks_t}d {'✓' if outage_ks<0.80 else '⚠ LENGTH DECODES FAILURE'}")
    print(f"  Routine outages >=6d:    {routine_long_share*100:.0f}% of {_nr:,} {'✓' if routine_long_share>0.15 else '⚠ no long tail'}")
    print(f"  Clean wells w/ long o/a: {_clean_long}/{len(clean_keys)} {'✓' if clean_long_ok else '⚠ clean cohort short-only'}")
    print(f"  Horizontal/Vertical:{horiz_ct}/{len(wells_f)-horiz_ct} {'✓' if geom_ok and lat_ok else '⚠'}")
    print(f"  Geometry x lift:    {'all 6 combos present ✓' if geom_ok else '⚠ missing combo'}")
    print(f"  Failures/well:      {fpw:.1f} {'✓' if 8<=fpw<=13 else '⚠ (target 8-13)'}")
    print(f"  Total invoices:     {total_inv:,} {'✓' if total_inv>MIN_INV else '⚠'}")
    print(f"  Duplicate inv IDs:  {dups} {'✓' if dups==0 else '⚠'}")
    print(f"  WI math errors:     {wi_err} {'✓' if wi_err==0 else '⚠'}")
    print(f"  ESP pump curves:    {pump_ct}/{esp_ct} {'✓' if pump_ct==esp_ct else '⚠'}")
    print(f"  Pressure step-down: {'✓' if pres_ok else '⚠'}")
    print(f"  Bad/label fields:   {bad if bad else 'none ✓'}")
    print(f"  Negatives:          {neg} {'✓' if neg==0 else '⚠'}")
    print(f"  WO invoices on event date: {wo_exact}/{wo_tot} ({wo_leak_pct:.1f}%) {'✓' if wo_leak_pct<10 else '⚠ LABEL LEAK'}")
    print(f"  Clean wells w/o WO spend:  {clean_no_wo}/{len(clean_keys)} {'✓' if clean_no_wo==0 else '⚠ LABEL LEAK'}")
    print(f"  Live values on SHUT-IN row:{shutin_live} {'✓' if shutin_live==0 else '⚠ FAILURE DAY EXPOSED'}")
    print(f"  Months w/o LOE invoice:    {months_no_loe} {'✓' if months_no_loe==0 else '⚠ IDLE MONTH LABELED'}")
    print(f"  Months w/o revenue rec:    {months_no_rev} {'✓' if months_no_rev==0 else '⚠ IDLE MONTH LABELED'}")
    print(f"  Oil mass balance:          {mb_oil_lo:.1f}-{mb_oil_hi:.1f}% {'✓' if mb_oil_ok else '⚠ OIL LOST'}")
    print(f"  Water disposed vs made:    {mb_wtr_lo:.1f}-{mb_wtr_hi:.1f}% {'✓' if mb_wtr_ok else '⚠ DOUBLE DISPOSAL'}")
    print(f"  Horiz/vert peak-oil ratio: "+", ".join(f"{k} {v:.2f}x" for k,v in geom_ratio.items())+f" {'✓' if geom_live else '⚠ geometry inert'}")
    print(f"  Horiz/vert well counts:    "+", ".join(f"{k} {h}H/{vv}V" for k,(h,vv) in geom_cells.items())+f" {'✓' if geom_cells_ok else '⚠ profile cell too thin'}")
    print(f"  Pooled H uplift (norm):    {geom_pooled:.2f}x {'✓' if geom_pooled>1.15 else '⚠ geometry inert'}")
    print(f"  Lateral length vs peak:    r={geom_lat_corr:.2f} {'✓' if geom_lat_corr>0.20 else '⚠ length does not drive rate'}")
    print(f"  High level alarms:         {alarm_tot:,} of {tank_rows:,} rows {'✓' if alarm_ok else '⚠ alarm dead'}")
    print(f"  Rows missing org_id/ts:    {rows_missing:,} {'✓' if rows_missing==0 else '⚠'}")
    print(f"  Revenue records:           {rev_rows:,} {'✓' if rev_rows>MIN_INV else '⚠'}")
    print(f"  Ledger ties out (o/g/ngl): {rev_gas_rows-rev_recon_bad:,}/{rev_gas_rows:,} {'✓' if rev_recon_bad==0 else '⚠ LEDGER WILL NOT TIE OUT'}")
    print(f"  Revenue math errors:       {rev_math} {'✓' if rev_math==0 else '⚠'}")
    print(f"  NRI application errors:    {rev_nri} {'✓' if rev_nri==0 else '⚠'}")
    print(f"  Producing months w/o rev:  {rev_months_missing} {'✓' if rev_months_missing==0 else '⚠'}")
    print(f"  Price deck months:         {len(PRICE_DECK)} ({neg_gas_months} negative-gas) {'✓' if deck_ok else '⚠'}")
    print(f"  Lifetime net revenue:      ${tot_net/1e6:,.0f}M  vs cost ${tot_cost/1e6:,.0f}M")
    print(f"  rod_load on high-amp days: {rl_live:.3f} {'✓' if rl_live>1.05 else '⚠ feature dead'}")
    print(f"  fillage trend neg on collapse: "+(f"n/a ({len(_ft)} wells sampled)" if ft_neg<0
          else f"{ft_neg:.0f}% {'✓' if ft_neg>=80 else '⚠ feature dead'}"))
    print(f"\n  ── Scope / vendor coupling ──")
    print(f"  Unrecognized WO scopes:    {scope_unknown} {'✓' if scope_unknown==0 else '⚠'}")
    print(f"  Vendor not qualified:      {vendor_mismatch}/{sum(wo_scope_ct.values()):,} {'✓' if vendor_mismatch==0 else '⚠ VENDOR MISMATCH'}")
    print(f"  Preventive scope off-lift: {prev_lift_bad} {'✓' if prev_lift_bad==0 else '⚠'}")
    print(f"  Preventive vendor bad:     {prev_vendor_bad} {'✓' if prev_vendor_bad==0 else '⚠'}")
    print(f"  Sub-types per scope:       min {min(len(c) for c in _by_scope.values())} {'✓' if not scopes_single_mode else '⚠ '+str(scopes_single_mode)}")
    print(f"  Scopes per sub-type:       min {min(len(k) for k in _by_mode.values())} {'✓' if not modes_single_scope else '⚠ '+str(modes_single_scope)}")
    print(f"  Max P(sub-type | scope):   {scope_leak:.2f} on '{scope_leak_key}' {'✓' if scope_leak<0.75 else '⚠ SCOPE DECODES MODE'}")
    print(f"  Scope mix:                 {dict(wo_scope_ct.most_common())}")
    print(f"\n  ── Cost drivers ──")
    print(f"  Supp invoices w/o driver:  {supp_flat} {'✓' if supp_flat==0 else '⚠'}")
    for _a,_c in supp_corr.items():
        print(f"  Supp corr {_a:<13}{_c:.2f} {'✓' if _c>0.5 else '⚠ decoupled'}")
    print(f"  Supp maintenance invoices: {maint_ct:,} {'✓' if maint_ct>0 else '⚠ repair path dead'}")
    print(f"  Supp cap hits:             {supp_cap_hits}/{len(_supp_loe):,} {'✓' if supp_cap_hits==0 else '⚠ CAP BINDING'}")
    print(f"  Supp excursion days:       "+", ".join(f"{a} {supp_exc[a]:,}/{supp_rows[a]:,}" for a in supp_exc)+f" {'✓' if supp_exc_ok else '⚠'}")
    print(f"  Compressor down days:      {comp_down:,} {'✓' if comp_down>0 else '⚠'}")
    print(f"  LOE corr oil/water/kwh:    {loe_corr_oil:.2f} / {loe_corr_water:.2f} / {loe_corr_kwh:.2f} {'✓' if loe_ok else '⚠'}")
    print(f"  LOE spread at fixed oil:   {loe_band_ratio:.2f}x {'✓' if loe_band_ratio>1.25 else '⚠ oil-only'}")
    print(f"  LOE median / cap hits:     ${loe_med:,.0f} / {loe_cap_hits} {'✓' if loe_cap_hits<len(_loe)*0.01 else '⚠ cap binding'}")

    issues=[x for x in[
        f"Assets: {len(all_assets)}" if len(all_assets)!=N_ASSETS else None,
        f"Wells: {len(wells_f)}" if len(wells_f)!=N_WELLS else None,
        f"Clean: {clean_ct}" if clean_ct!=N_CLEAN else None,
        f"Clean wells with failure events: {clean_had_failure}" if clean_had_failure else None,
        f"Clean wells with zero downtime (uptime labels cohort): {clean_zero_dt}" if clean_zero_dt else None,
        f"Outage length decodes failure: KS={outage_ks:.2f} at >={outage_ks_t}d" if outage_ks>=0.80 else None,
        f"Routine outages have no long tail: {routine_long_share*100:.0f}% >=6d" if routine_long_share<=0.15 else None,
        f"Clean wells with only short outages: {len(clean_keys)-_clean_long}" if not clean_long_ok else None,
        f"SHUT-IN rows carrying live sensor values: {shutin_live}" if shutin_live else None,
        f"Well-months with no LOE invoice: {months_no_loe}" if months_no_loe else None,
        f"Well-months with no revenue record: {months_no_rev}" if months_no_rev else None,
        f"Oil mass balance {mb_oil_lo:.1f}-{mb_oil_hi:.1f}% of lease oil" if not mb_oil_ok else None,
        f"Water disposed exceeds produced: {mb_wtr_hi:.1f}%" if not mb_wtr_ok else None,
        f"Geometry inert — pooled H uplift {geom_pooled:.2f}x" if geom_pooled<=1.15 else None,
        f"Lateral length does not drive rate — r={geom_lat_corr:.2f}" if geom_lat_corr<=0.20 else None,
        f"Too few wells on one profile to compare: {geom_cells}" if not geom_cells_ok else None,
        f"Post-2024: {recent}" if SCALE=="full" and recent<int((len(wells_f)-clean_ct)*0.85) else None,
        f"Failures/well: {fpw:.1f}" if SCALE=="full" and not (8<=fpw<=13) else None,
        f"WO label leak: {wo_leak_pct:.1f}% on exact event date" if wo_leak_pct>=10 else None,
        f"Clean wells w/o workover spend: {clean_no_wo}" if clean_no_wo>0 else None,
        "Geometry: missing lift x profile combo" if not geom_ok else None,
        "Lateral length inconsistent with profile" if not lat_ok else None,
        f"Tank run tickets {tank_lo:.0f}-{tank_hi:.0f}% of lease oil" if not tank_ok else None,
        f"high_level_alarm fired {alarm_tot} times" if not alarm_ok else None,
        f"Rows missing organization_id/timestamp: {rows_missing:,}" if rows_missing>0 else None,
        f"Revenue records: {rev_rows}" if rev_rows<MIN_INV else None,
        f"Revenue does not reconcile on {rev_recon_bad} records" if rev_recon_bad else None,
        f"Revenue math errors: {rev_math}" if rev_math>0 else None,
        f"NRI application errors: {rev_nri}" if rev_nri>0 else None,
        f"Duplicate revenue ids/months: {rev_dups}" if rev_dups>0 else None,
        f"Producing months without revenue: {rev_months_missing}" if rev_months_missing>0 else None,
        "Price deck incomplete" if not deck_ok and SCALE=="full" else None,
        f"rod_load_ratio dead ({rl_live:.3f})" if rl_live<=1.05 else None,
        f"fillage trend dead ({ft_neg:.0f}% negative)" if 0<=ft_neg<80 else None,
        f"Unrecognized workover scopes: {scope_unknown}" if scope_unknown>0 else None,
        f"Vendor not qualified for scope: {vendor_mismatch}" if vendor_mismatch>0 else None,
        f"Preventive scope not valid for lift: {prev_lift_bad}" if prev_lift_bad>0 else None,
        f"Preventive vendor not qualified: {prev_vendor_bad}" if prev_vendor_bad>0 else None,
        f"Scope decodes sub-type: max P={scope_leak:.2f} on {scope_leak_key}" if scope_leak>=0.75 else None,
        (f"Scopes reachable from one sub-type only: {scopes_single_mode}"
            if scopes_single_mode and SCALE=="full" else None),
        (f"Sub-types with <3 candidate scopes: {modes_single_scope}"
            if modes_single_scope and SCALE=="full" else None),
        f"Supporting-asset spend decoupled from throughput: {supp_corr}" if not supp_corr_ok else None,
        f"Supporting-asset invoices without driver volumes: {supp_flat}" if supp_flat>0 else None,
        f"Well LOE not multi-driver (water {loe_corr_water:.2f}, kwh {loe_corr_kwh:.2f}, band {loe_band_ratio:.2f}x)" if not loe_ok else None,
        f"LOE cap binding on {loe_cap_hits} invoices" if loe_cap_hits>=len(_loe)*0.01 else None,
        "Supporting-asset maintenance path never fired" if maint_ct==0 else None,
        f"SUPP_MAX_CAP binding on {supp_cap_hits} invoices" if supp_cap_hits>0 else None,
        f"Supporting-asset excursion days out of range: {supp_exc}" if not supp_exc_ok else None,
        "Compressor has no downtime" if comp_down==0 else None,
        f"Invoices: {total_inv}" if total_inv<MIN_INV else None,
        f"Dups: {dups}" if dups>0 else None,
        f"WI errors: {wi_err}" if wi_err>0 else None,
        f"Pump curves: {pump_ct}/{esp_ct}" if pump_ct!=esp_ct else None,
        "Pressure step-down" if not pres_ok else None,
        f"Bad fields: {bad}" if bad else None,
        f"Negatives: {neg}" if neg>0 else None,
        f"End date: {max_date}" if max_date!=END_DATE.strftime('%Y-%m-%d') else None,
    ] if x]

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for iss in issues: print(f"    ⚠ {iss}")
        print("\n  BUILD FAILED — output not written.")
        sys.exit(1)
    else:
        print(f"\n  ALL CLEAR")

    output={
        "company":COMPANY,"organization_id":ORGANIZATION_ID,"build_version":"V2",
        # Defaults to the dataset as-of date, NOT wall-clock time, so repeated builds of
        # the same script are byte-identical and the artifact can be checksummed.
        # Override with RBNR_BUILD_DATE=YYYY-MM-DD to stamp a real build date.
        "generated":os.environ.get("RBNR_BUILD_DATE", END_DATE.strftime("%Y-%m-%d")),
        "description":f"Red Bluff Resources V2 — {N_WELLS} wells, Permian Basin, sensor-driven failures, no labels",
        "price_deck":PRICE_DECK,
        "price_deck_note":("SYNTHETIC monthly realized prices shaped to real Permian history "
                          "(2008 spike/collapse, H2-2014 crash, 2018 Midland diff blowout, "
                          "Apr-2020, 2022 spike, negative Waha months). Not market data."),
        "fiscal_terms":{"severance_tax_oil":SEV_TAX_OIL,"severance_tax_gas":SEV_TAX_GAS,
                        "ad_valorem":AD_VALOREM,"oil_transport_usd_bbl":OIL_TRANSPORT,
                        "gas_gpt_usd_mcf":GAS_GPT},
        "assets":all_assets,
    }
    print(f"\nSaving {OUTPUT_FILE}...")
    with open(OUTPUT_FILE,"w") as f: json.dump(output,f,indent=2)
    sz=os.path.getsize(OUTPUT_FILE)
    print(f"Done — {sz/1024/1024:.0f} MB saved as {OUTPUT_FILE}")
