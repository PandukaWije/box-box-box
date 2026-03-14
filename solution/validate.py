#!/usr/bin/env python3
"""
validate.py — Test your simulator accuracy against historical data.

Usage:
  python solution/validate.py              # test against 500 historical races
  python solution/validate.py --n 2000    # more races
  python solution/validate.py --debug     # show first 5 failures in detail
"""

import json
import sys
import argparse
from pathlib import Path

# Add solution dir to path so we can import race_simulator
sys.path.insert(0, '.')
from solution.race_simulator import load_params, simulate_race

DATA_DIR = 'data/historical_races'


def load_races(n):
    races = []
    for rf in sorted(Path(DATA_DIR).glob('*.json')):
        with open(rf) as f:
            races.extend(json.load(f))
        if len(races) >= n:
            break
    return races[:n]


def validate(n=500, debug=False):
    print(f"Loading {n} historical races...")
    races = load_races(n)
    params = load_params()

    correct = 0
    failures = []

    for race in races:
        cfg      = race['race_config']
        strats   = race['strategies']
        expected = race['finishing_positions']

        predicted = simulate_race(cfg, strats, params)

        if predicted == expected:
            correct += 1
        else:
            failures.append((race, predicted, expected))

    acc = correct / len(races) * 100
    print(f"\n{'='*50}")
    print(f"Results: {correct}/{len(races)} correct  ({acc:.1f}%)")
    print(f"{'='*50}")

    if acc >= 95:
        print("🏆 Excellent! Very likely to ace the test cases.")
    elif acc >= 80:
        print("✅ Good score. Minor formula tweaks may push higher.")
    elif acc >= 60:
        print("⚠️  Decent start. Formula might be missing a factor.")
    else:
        print("❌ Low accuracy. Check your formula structure.")

    if debug and failures:
        print(f"\n--- First {min(5, len(failures))} Failures ---")
        for race, pred, exp in failures[:5]:
            cfg = race['race_config']
            print(f"\nRace {race['race_id']}  Temp={cfg['track_temp']}  Laps={cfg['total_laps']}")
            print(f"  Expected:  {exp[:5]}...")
            print(f"  Predicted: {pred[:5]}...")
            # Show where they diverge
            for i, (e, p) in enumerate(zip(exp, pred)):
                if e != p:
                    print(f"  First diff at position {i+1}: expected {e}, got {p}")
                    break


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    validate(n=args.n, debug=args.debug)
