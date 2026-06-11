================================================================================
RED BLUFF RESOURCES SYNTHETIC DATASET
Digital Twin Pipeline
================================================================================

OVERVIEW
--------
This pipeline generates a fully realistic, physics-driven synthetic dataset for
60 upstream oil and gas wells across 4 leases in the Permian Basin, plus 16
supporting assets (facilities, compressors, SWD wells, tank batteries).

The dataset is designed for training AI/ML models in:
  - Predictive Maintenance (Time-To-Failure, early warning detection)
  - Financial Forecasting (LOE regression, workover cost prediction)
  - Anomaly Detection (AP audit, sensor anomaly classification)

COMPANY
-------
  Name:    Red Bluff Resources
  Basin:   Permian Basin (Delaware Basin + Midland Basin)
  State:   TX  (Texas)
  Leases:  Chaparral (Reeves County, Delaware Basin)
           Mesquite  (Loving County,  Delaware Basin)
           Caliche   (Midland County,  Midland Basin)
           Llano     (Ector County,    Midland Basin)

PIPELINE — RUN IN ORDER
-----------------------
  1. python3 generate_digital_twin.py
     Builds 60 wells with lift-specific sensors, hyperbolic Arps decline
     curves, and gradual random-walk pre-failure deterioration.
     Output: red_bluff_engineering.json

  2. python3 generate_invoices.py
     Generates financial ledger (Monthly LOE, Ad-Hoc LOE, Workover invoices)
     for all 60 wells. Updates red_bluff_engineering.json in-place.
     Output: red_bluff_engineering.json (updated)

  3. python3 build_master.py
     Adds 16 supporting assets, assigns personnel and location to all 76
     assets, selects 12 clean wells (no failures) for healthy baseline
     training data.
     Output: Red_Bluff_Resources_Master.json

  4. python3 export_to_csv.py  (optional)
     Flattens master JSON into 4 relational CSV files joined by asset_id.
     Output: assets.csv, personnel.csv, production_history.csv, invoices.csv

DATASET CONTENTS
----------------
  Total assets:        76
    Wells:             60  (producing wells)
    Facilities:         4  (one per lease — central battery/gathering)
    Compressors:        4  (one per lease — gas lift compressor station)
    SWD Wells:          4  (one per lease — salt water disposal injection well)
    Tank Batteries:     4  (one per lease — oil and water storage)

  Date range:          February 2012 → May 7, 2026
  Daily rows:          ~304,763
  Total invoices:      ~62,454

WELL CONFIGURATION
------------------
  Total wells:         60
  Lift mix:            27 Pumping Unit  (45%)  — Vertical wells  (-V suffix)
                       21 ESP           (35%)  — Horizontal wells (-H suffix)
                       12 Gas Lift      (20%)  — Horizontal wells (-H suffix)

  Wells per lease:     15 per lease (Chaparral, Mesquite, Caliche, Llano)

  Well naming:         {Lease}-12-{Number}-{Suffix}
                       -V = Vertical   (Pumping Unit)
                       -H = Horizontal (ESP, Gas Lift)
                       Example: Chaparral-12-3-V, Mesquite-12-16-H

  Failure split:       48 wells experience at least one failure (80%)
                       12 wells run completely clean, no failures (20%)
                         3 clean wells per lease, one of each lift type

  Clean wells:
    Chaparral: Chaparral-12-9-H  (ESP)
               Chaparral-12-10-V (Pumping Unit)
               Chaparral-12-15-H (Gas Lift)
    Mesquite:  Mesquite-12-19-H  (ESP)
               Mesquite-12-21-V  (Pumping Unit)
               Mesquite-12-22-H  (Gas Lift)
    Caliche:   Caliche-12-32-V   (Pumping Unit)
               Caliche-12-35-H   (ESP)
               Caliche-12-37-H   (Gas Lift)
    Llano:     Llano-12-48-H     (ESP)
               Llano-12-49-H     (Gas Lift)
               Llano-12-54-V     (Pumping Unit)

SENSOR FIELDS PER LIFT TYPE
----------------------------
  Pumping Unit — 17 fields:
    date, allocated_oil, test_oil, gross_stroke_len, net_stroke_len,
    strokes_per_min, pump_fillage_pct, casing_pressure_psi,
    tubing_pressure_psi, motor_current_amps, freq_hz,
    pump_intake_pressure, motor_temp_f, run_status, failure_cause,
    allocated_water, allocated_gas

  ESP — 15 fields:
    date, allocated_oil, test_oil, avg_tubing_pressure,
    avg_casing_pressure, motor_temp_f, motor_current_amps, freq_hz,
    pump_intake_pressure_psi, pump_discharge_pressure_psi, vibration_hz,
    run_status, failure_cause, allocated_water, allocated_gas

  Gas Lift — 14 fields:
    date, allocated_oil, test_oil, avg_casing_pressure,
    casing_injection_pressure_psi, tubing_pressure_psi,
    injection_rate_mscfd, wellhead_temp_f, choke_position_pct,
    motor_temp_f, run_status, failure_cause, allocated_water, allocated_gas

SUPPORTING ASSET FIELDS
------------------------
  Facility     — 22 fields: tester pressure/temp/rates, LACT, gas sales,
                             water transfer, tank levels, run_status
  Compressor   — 12 fields: suction/discharge pressure & temp, throughput,
                             engine RPM, fuel consumption, lube oil, runtime
  SWD Well     — 12 fields: injection rate/pressure, wellhead pressure,
                             tubing/casing pressure, daily & cumulative volume
  Tank Battery —  9 fields: oil/water tank levels & temps, haul volumes,
                             high tank alarm

PRODUCTION DATA
---------------
  Decline curve:       Hyperbolic (Arps)  q(t) = q_i / (1 + b*Di*t)^(1/b)
                       Wells decline steeply early then flatten to a stable
                       long-tail rate. No well ever reaches zero oil on a
                       producing day.

  Production streams:  allocated_oil    (bbl/day)
                       allocated_water  (bbl/day) — rises over time as
                                        water cut increases with depletion
                       allocated_gas    (mcf/day)  — declines with oil

  test_oil:            Monthly well test value — present once per month,
                       null on all other days. Reflects ±5-12% measurement
                       variance vs allocated_oil.

  Motor temperatures (healthy operating ranges):
                       Pumping Unit: 100–160°F  (avg ~130°F)
                       ESP:          150–250°F  (avg ~200°F)
                       Gas Lift:      80–130°F  (avg ~105°F)

RUN STATUS & FAILURE CAUSES
----------------------------
  run_status values:   PRODUCING  — well is online and producing
                       SHUT-IN    — well is down for workover/repair
                       ONLINE     — supporting assets (facility, compressor,
                                    SWD well, tank battery) are operational

  Downtime duration:   5–13 days per event (avg ~9 days)

  Failure causes (100% coverage on all SHUT-IN events):
    Pumping Unit:      Parted Rod String     (14–28 day pre-failure window)
                       Worn Pump / Tag       (60–120 day pre-failure window)
    ESP:               Motor Overtemp Shutdown (45–75 day window)
                       Motor/Cable Failure   (20–40 day window)
                       Pump Cavitation       (10–21 day window)
    Gas Lift:          Gas Lift Valve Failure (60–90 day window)
                       Mandrel Washout       (45–75 day window)

  Pre-failure signal:  Sensor values drift gradually toward failure threshold
                       on a random walk with partial recoveries — signal is
                       buried in noise early and only becomes clear in the
                       final days before failure. This is the critical training
                       signal for predictive maintenance model learning.

PERSONNEL
---------
  One Well Performance Engineer per lease (consistent across all assets):
    Chaparral: Derek Tatum      | wellperf@redbluffresources.com
    Mesquite:  Kristen Albright | wellperf@redbluffresources.com
    Caliche:   Derek Tatum      | wellperf@redbluffresources.com
    Llano:     Kristen Albright | wellperf@redbluffresources.com

  One Production Technician per lease:
    Chaparral: Chaparral Production Tech | prodtech.chaparral@redbluffresources.com
    Mesquite:  Mesquite Production Tech  | prodtech.mesquite@redbluffresources.com
    Caliche:   Caliche Production Tech   | prodtech.caliche@redbluffresources.com
    Llano:     Llano Production Tech     | prodtech.llano@redbluffresources.com

FINANCIAL DATA
--------------
  Invoice prefix:      RBNR-YYYY-NNNNN
  GL code format:      XXX-XXX  (e.g. 605-110, 710-520, 822-110)
  Invoice categories:  Monthly LOE, Ad-Hoc LOE, Workover

  Vendors (10 total):
    Trans-Pecos Electric          — electrical power
    Red Bluff Internal            — labor and G&A
    Permian Chemical Supply       — chemicals
    Basin Telemetry & Controls    — SCADA / RTU
    Reeves County Water Hauling   — water hauling and SWD
    Midland Hot Oil Service       — hot oil / paraffin treatment
    Pecos Valley Roustabout       — location maintenance
    Red Bluff Rig Services        — workovers and SWD wellbore
    Delaware Basin Compression    — compressor maintenance
    West Texas Electrical Services— VFD service and electrical

KNOWN SIMPLIFICATIONS
---------------------
  - All wells share Section 12 within their lease (no geographic spread
    across sections — a simplification of real land survey structure)
  - vibration_hz (ESP) and choke_position_pct (Gas Lift) are realistic
    sensors but not universally deployed on all real wells
  - Sensor ranges sourced from data_types_and_ranges.xlsx represent the
    full sensor capability range; healthy day-to-day values are a subset
  - Each rebuild of the pipeline will produce different random values
    (spud dates, production rates, sensor readings) while maintaining
    the same structural design and field schema

================================================================================
INTERNAL USE ONLY — SYNTHETIC DATA — RED BLUFF RESOURCES
================================================================================
