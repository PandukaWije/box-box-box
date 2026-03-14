#!/usr/bin/env python3
"""
race_simulator.py — Box Box Box F1 Race Simulator

Reads race config + strategies from stdin (JSON),
outputs finishing positions to stdout (JSON).

Usage:
  cat data/test_cases/inputs/test_001.json | python solution/race_simulator.py
"""

import json
import sys
import os

PARAMS_FILE = 'solution/fitted_params.json'
TEMP_REF    = 30.0


# ─── Hardcoded fallback params ────────────────────────────────────────────────
# These are reasonable starting defaults. Replace by running fit_params.py.
FALLBACK_PARAMS = {
    "soft_offset":  -1.5,
    "med_offset":   -0.7,
    "soft_deg":      0.08,
    "med_deg":       0.04,
    "hard_deg":      0.02,
    "soft_cliff":    5.0,
    "med_cliff":    12.0,
    "hard_cliff":   20.0,
    "temp_scale":    0.015,
}


def load_params():
    """Load fitted params from cache, or fall back to defaults."""
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            return json.load(f)
    # Fallback — run solution/fit_params.py first for best accuracy
    return FALLBACK_PARAMS


def calc_lap_time(base, compound, tire_age, track_temp, p):
    """
    lap_time = base
             + compound_offset[c]
             + deg_rate[c] * max(0, tire_age - cliff[c]) * (1 + temp_scale * (temp - 30))
    """
    offsets  = {'SOFT': p['soft_offset'], 'MEDIUM': p['med_offset'], 'HARD': 0.0}
    deg_rate = {'SOFT': p['soft_deg'],    'MEDIUM': p['med_deg'],    'HARD': p['hard_deg']}
    cliff    = {'SOFT': p['soft_cliff'],  'MEDIUM': p['med_cliff'],  'HARD': p['hard_cliff']}

    eff_age   = max(0.0, tire_age - cliff[compound])
    temp_mult = 1.0 + p['temp_scale'] * (track_temp - TEMP_REF)
    return base + offsets[compound] + deg_rate[compound] * eff_age * temp_mult


def simulate_race(race_config, strategies, params):
    """Simulate a race. Returns list of driver IDs in finishing order (1st → last)."""
    base   = race_config['base_lap_time']
    laps   = race_config['total_laps']
    pit_t  = race_config['pit_lane_time']
    temp   = race_config['track_temp']

    driver_times = {}

    for strat in strategies.values():
        did  = strat['driver_id']
        tire = strat['starting_tire']
        pits = {ps['lap']: ps['to_tire'] for ps in strat.get('pit_stops', [])}

        total = 0.0
        age   = 0

        for lap in range(1, laps + 1):
            age   += 1
            total += calc_lap_time(base, tire, age, temp, params)
            if lap in pits:
                total += pit_t
                tire   = pits[lap]
                age    = 0

        driver_times[did] = total

    # Sort by total race time — lowest time = 1st place
    return sorted(driver_times, key=lambda d: driver_times[d])


def main():
    params    = load_params()
    test_case = json.load(sys.stdin)

    finishing_positions = simulate_race(
        test_case['race_config'],
        test_case['strategies'],
        params,
    )

    result = {
        'race_id':             test_case['race_id'],
        'finishing_positions': finishing_positions,
    }

    print(json.dumps(result))


if __name__ == '__main__':
    main()
