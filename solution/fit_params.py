#!/usr/bin/env python3
"""
fit_params.py — Reverse-engineer the lap time formula from historical races.

The race simulator is deterministic. This script finds the exact parameters
of the lap time formula by minimising ranking loss across historical races.

Usage: python solution/fit_params.py
Output: solution/fitted_params.json
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, differential_evolution

DATA_DIR   = '../data/historical_races'
OUTPUT     = 'fitted_params.json'
TEMP_REF   = 30.0   # Normalise temperature around 30°C

# ─── Lap Time Formula ─────────────────────────────────────────────────────────
#
#  lap_time = base_lap_time
#           + compound_offset[c]                          ← compound speed delta
#           + deg_rate[c]                                 ← degradation per lap
#             * max(0, tire_age - cliff[c])               ← grace period
#             * (1 + temp_scale * (track_temp - TEMP_REF)) ← temperature effect
#
#  Parameters (9 total):
#   p[0] = soft_offset   (< 0, SOFT is faster than HARD)
#   p[1] = med_offset    (< 0, MEDIUM is faster than HARD)
#   p[2] = soft_deg_rate (> 0)
#   p[3] = med_deg_rate  (> 0)
#   p[4] = hard_deg_rate (> 0)
#   p[5] = soft_cliff    (≥ 0, grace laps before degradation starts)
#   p[6] = med_cliff     (≥ 0)
#   p[7] = hard_cliff    (≥ 0)
#   p[8] = temp_scale    (> 0, higher temp → more degradation)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES  = ['soft_offset','med_offset','soft_deg','med_deg','hard_deg',
                'soft_cliff','med_cliff','hard_cliff','temp_scale']
INITIAL      = [-1.5, -0.7, 0.08, 0.04, 0.02, 5.0, 12.0, 20.0, 0.015]
BOUNDS       = [(-5.0, -0.01), (-4.0, -0.01),
                (0.001, 0.6), (0.001, 0.4), (0.001, 0.3),
                (0.0, 30.0), (0.0, 40.0), (0.0, 60.0),
                (-0.05, 0.1)]


def lap_time_formula(base, compound, tire_age, track_temp, p):
    offsets  = {'SOFT': p[0], 'MEDIUM': p[1], 'HARD': 0.0}
    deg_rate = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    cliff    = {'SOFT': p[5], 'MEDIUM': p[6], 'HARD': p[7]}

    eff_age = max(0.0, tire_age - cliff[compound])
    temp_mult = 1.0 + p[8] * (track_temp - TEMP_REF)
    return base + offsets[compound] + deg_rate[compound] * eff_age * temp_mult


def simulate_race(cfg, strategies, p):
    """Return {driver_id: total_time}."""
    base   = cfg['base_lap_time']
    laps   = cfg['total_laps']
    pit_t  = cfg['pit_lane_time']
    temp   = cfg['track_temp']

    times = {}
    for strat in strategies.values():
        did  = strat['driver_id']
        tire = strat['starting_tire']
        pits = {ps['lap']: ps['to_tire'] for ps in strat.get('pit_stops', [])}

        t = 0.0; age = 0
        for lap in range(1, laps + 1):
            age += 1
            t += lap_time_formula(base, tire, age, temp, p)
            if lap in pits:
                t += pit_t
                tire = pits[lap]
                age = 0
        times[did] = t
    return times


# ─── Pre-computation ──────────────────────────────────────────────────────────

def precompute(race):
    """
    Extract the param-independent parts so the optimiser loop is fast.
    Returns (driver_laps_list, fixed_baseline, expected_order, temp)
      driver_laps_list: {did: [(compound, tire_age), ...]}
      fixed_baseline:   {did: base_total + pit_penalties}  (no formula terms)
    """
    cfg   = race['race_config']
    base  = cfg['base_lap_time']
    laps  = cfg['total_laps']
    pit_t = cfg['pit_lane_time']
    temp  = cfg['track_temp']

    driver_laps  = {}
    fixed_base   = {}

    for strat in race['strategies'].values():
        did  = strat['driver_id']
        tire = strat['starting_tire']
        pits = {ps['lap']: ps['to_tire'] for ps in strat.get('pit_stops', [])}

        lap_seq = []; age = 0; pit_cost = base * laps
        for lap in range(1, laps + 1):
            age += 1
            lap_seq.append((tire, age))
            if lap in pits:
                pit_cost += pit_t
                tire = pits[lap]
                age = 0

        driver_laps[did] = lap_seq
        fixed_base[did]  = pit_cost

    return (driver_laps, fixed_base, race['finishing_positions'], temp)


def fast_loss(p, precomputed_races):
    """
    Pairwise ranking loss (adjacent pairs only).
    For adjacent (winner, loser): penalise if winner_time ≥ loser_time.
    """
    offsets  = {'SOFT': p[0], 'MEDIUM': p[1], 'HARD': 0.0}
    deg_rate = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    cliff    = {'SOFT': p[5], 'MEDIUM': p[6], 'HARD': p[7]}
    temp_s   = p[8]

    total = 0.0
    for (driver_laps, fixed_base, expected, temp) in precomputed_races:
        temp_mult_base = 1.0 - temp_s * TEMP_REF  # factored out
        temp_mult      = 1.0 + temp_s * (temp - TEMP_REF)

        times = {}
        for did, lap_seq in driver_laps.items():
            t = fixed_base[did]
            for (c, age) in lap_seq:
                eff = max(0.0, age - cliff[c])
                t  += offsets[c] + deg_rate[c] * eff * temp_mult
            times[did] = t

        # Adjacent pair ranking loss
        margin = 0.001
        for i in range(len(expected) - 1):
            a, b = expected[i], expected[i+1]
            diff = times[a] - times[b]  # want diff < 0
            if diff >= -margin:
                total += (diff + margin) ** 2

    return total


def accuracy(p, races, n=None):
    """Fraction of races where predicted order exactly matches expected."""
    sample = races[:n] if n else races
    correct = 0
    for race in sample:
        cfg = race['race_config']
        times = simulate_race(cfg, race['strategies'], p)
        predicted = sorted(times, key=lambda d: times[d])
        if predicted == race['finishing_positions']:
            correct += 1
    return correct / len(sample)


# ─── Main Fitting ─────────────────────────────────────────────────────────────

def load_races(n=1000):
    races = []
    for rf in sorted(Path(DATA_DIR).glob('*.json')):
        with open(rf) as f:
            races.extend(json.load(f))
        if len(races) >= n:
            break
    return races[:n]


def fit(num_fit=500, num_val=200):
    print("="*60)
    print("Box Box Box — Parameter Fitter")
    print("="*60)

    print(f"\n[1/4] Loading {num_fit + num_val} historical races...")
    all_races = load_races(num_fit + num_val)
    fit_races = all_races[:num_fit]
    val_races = all_races[num_fit:num_fit + num_val]
    print(f"      Fit: {len(fit_races)}   Val: {len(val_races)}")

    print("\n[2/4] Pre-computing race data for fast optimisation...")
    precomputed = [precompute(r) for r in fit_races]
    print(f"      Done.")

    # ── Phase 1: Global search (Differential Evolution) ────────────────────
    print("\n[3/4] Phase 1 — Global search (Differential Evolution)...")
    t0 = time.time()
    de_result = differential_evolution(
        fast_loss,
        BOUNDS,
        args=(precomputed,),
        maxiter=300,
        tol=1e-12,
        seed=42,
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.7,
        workers=1,
        init='latinhypercube',
        disp=False,
    )
    print(f"      Loss: {de_result.fun:.8f}  ({time.time()-t0:.1f}s)")
    _print_params(de_result.x)

    # ── Phase 2: Local refinement (L-BFGS-B) ──────────────────────────────
    print("\n      Phase 2 — Local refinement (L-BFGS-B)...")
    t0 = time.time()
    local_result = minimize(
        fast_loss,
        de_result.x,
        args=(precomputed,),
        method='L-BFGS-B',
        bounds=BOUNDS,
        options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-12},
    )
    best = local_result.x
    print(f"      Loss: {local_result.fun:.8f}  ({time.time()-t0:.1f}s)")
    _print_params(best)

    # ── Validation ─────────────────────────────────────────────────────────
    print("\n[4/4] Validation...")
    fit_acc = accuracy(best, fit_races, n=200)
    val_acc = accuracy(best, val_races)
    print(f"      Training accuracy (200): {fit_acc*100:.1f}%")
    print(f"      Validation accuracy ({len(val_races)}): {val_acc*100:.1f}%")

    # Save
    param_dict = dict(zip(PARAM_NAMES, best.tolist()))
    with open(OUTPUT, 'w') as f:
        json.dump(param_dict, f, indent=2)
    print(f"\n✅  Saved to {OUTPUT}")

    if val_acc < 0.7:
        print("\n⚠️  Accuracy < 70%. Tips:")
        print("    - Try more races: increase num_fit")
        print("    - Check formula structure in explore_data.py first")
        print("    - The formula might have a different degradation shape")

    return param_dict


def _print_params(p):
    for name, val in zip(PARAM_NAMES, p):
        print(f"        {name:<15} = {val:+.6f}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-races', type=int, default=500,
                    help='Number of races to use for fitting (default 500)')
    ap.add_argument('--val-races', type=int, default=200,
                    help='Number of races for validation (default 200)')
    args = ap.parse_args()

    fit(num_fit=args.fit_races, num_val=args.val_races)
