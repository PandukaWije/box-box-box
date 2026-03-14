#!/usr/bin/env python3
"""
explore_data.py — Run this FIRST to understand the data structure.
Usage: python solution/explore_data.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = '../data/historical_races'


def load_sample(n=200):
    races = []
    for rf in sorted(Path(DATA_DIR).glob('*.json')):
        with open(rf) as f:
            batch = json.load(f)
        races.extend(batch)
        if len(races) >= n:
            break
    return races[:n]


def print_race(race, idx=0):
    cfg = race['race_config']
    print(f"\n{'='*60}")
    print(f"Race: {race['race_id']}  |  Track: {cfg['track']}")
    print(f"Laps: {cfg['total_laps']}  |  Base: {cfg['base_lap_time']}s  |  "
          f"Pit: {cfg['pit_lane_time']}s  |  Temp: {cfg['track_temp']}°C")
    print(f"\n{'Driver':<8} {'Start':<8} {'Stops':<6} {'Strategy Summary':<40} {'Finish'}")
    print("-" * 70)

    finish = race['finishing_positions']
    finish_pos = {d: i+1 for i, d in enumerate(finish)}

    for pos in sorted(race['strategies'].keys(), key=lambda x: int(x[3:])):
        s = race['strategies'][pos]
        did = s['driver_id']
        stops = s['pit_stops']
        compound_seq = s['starting_tire']
        for ps in stops:
            compound_seq += f" →[L{ps['lap']}]→ {ps['to_tire']}"
        print(f"{did:<8} {s['starting_tire']:<8} {len(stops):<6} {compound_seq:<40} P{finish_pos[did]}")


def analyze_degradation_signal(races):
    """
    Key insight: compare two drivers in same race where one runs longer on same compound.
    We can infer relative degradation by watching when longer stints lose time.
    """
    print("\n\n" + "="*60)
    print("STINT LENGTH ANALYSIS")
    print("Looking for races with similar strategies but different stint lengths")
    print("="*60)

    compound_stint_data = defaultdict(list)  # compound -> [(stint_len, finish_pos, track_temp)]

    for race in races:
        cfg = race['race_config']
        finish = race['finishing_positions']
        finish_pos = {d: i+1 for i, d in enumerate(finish)}

        for s in race['strategies'].values():
            did = s['driver_id']
            current_tire = s['starting_tire']
            stops = sorted(s['pit_stops'], key=lambda x: x['lap'])

            stint_start = 1
            for ps in stops:
                stint_len = ps['lap'] - stint_start
                compound_stint_data[current_tire].append({
                    'stint_len': stint_len,
                    'finish_pos': finish_pos[did],
                    'temp': cfg['track_temp'],
                    'total_laps': cfg['total_laps'],
                })
                current_tire = ps['to_tire']
                stint_start = ps['lap'] + 1

            # Last stint
            stint_len = cfg['total_laps'] - stint_start + 1
            compound_stint_data[current_tire].append({
                'stint_len': stint_len,
                'finish_pos': finish_pos[did],
                'temp': cfg['track_temp'],
                'total_laps': cfg['total_laps'],
            })

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        data = compound_stint_data[compound]
        if not data:
            continue
        avg_len = sum(d['stint_len'] for d in data) / len(data)
        min_len = min(d['stint_len'] for d in data)
        max_len = max(d['stint_len'] for d in data)
        print(f"\n{compound}: n={len(data)}, stint length: min={min_len}, avg={avg_len:.1f}, max={max_len}")


def analyze_temperature_range(races):
    temps = sorted(set(r['race_config']['track_temp'] for r in races))
    laps = sorted(set(r['race_config']['total_laps'] for r in races))
    tracks = sorted(set(r['race_config']['track'] for r in races))
    print(f"\n\nTemperature range: {temps[:5]} ... {temps[-5:]}")
    print(f"Lap counts: {laps[:5]} ... {laps[-5:]}")
    print(f"Tracks ({len(tracks)}): {tracks[:10]}")


def find_simple_races(races, n=3):
    """Find races where strategies are simple (easy to analyze)."""
    print("\n\nSIMPLE RACES (1-stop strategies, diverse compounds)")
    print("="*60)
    shown = 0
    for race in races:
        strats = race['strategies']
        stop_counts = [len(s['pit_stops']) for s in strats.values()]
        if max(stop_counts) <= 1 and shown < n:
            print_race(race)
            shown += 1
            if shown >= n:
                break


def check_no_pit_drivers(races):
    """Drivers with no pit stops — pure compound comparison."""
    print("\n\nNO-PIT DRIVERS (pure compound effect)")
    print("="*60)

    compound_positions = defaultdict(list)  # compound -> [finishing_positions]

    for race in races:
        finish = race['finishing_positions']
        finish_pos = {d: i+1 for i, d in enumerate(finish)}
        n = len(finish)

        for s in race['strategies'].values():
            if len(s['pit_stops']) == 0:
                did = s['driver_id']
                c = s['starting_tire']
                # Normalise position to 0-1
                compound_positions[c].append(finish_pos[did] / n)

    for c in ['SOFT', 'MEDIUM', 'HARD']:
        pos_list = compound_positions[c]
        if pos_list:
            avg = sum(pos_list) / len(pos_list)
            print(f"{c:<8}: n={len(pos_list):>4}  avg_relative_pos={avg:.3f}  "
                  f"(lower is better, 0=1st, 1=last)")


if __name__ == '__main__':
    print("Loading sample races...")
    races = load_sample(n=500)
    print(f"Loaded {len(races)} races")

    # Print a few example races
    print("\n--- SAMPLE RACES ---")
    for i in [0, 1, 2]:
        print_race(races[i], i)

    # Analyze patterns
    analyze_temperature_range(races)
    find_simple_races(races, n=2)
    check_no_pit_drivers(races)
    analyze_degradation_signal(races)

    print("\n\nDone! Now run: python solution/fit_params.py")
