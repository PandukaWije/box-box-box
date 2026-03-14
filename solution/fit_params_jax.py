#!/usr/bin/env python3
"""
fit_params_jax.py — GPU-accelerated parameter fitter using JAX.

Designed for Google Colab with T4 GPU.
Expected time: ~20–40 seconds total (vs 26+ minutes on CPU).

Why JAX is fast here:
  - jit:  compiles the loss function to XLA once, runs native GPU code after
  - vmap: vectorises across ALL races simultaneously (no Python for-loop)
  - GPU:  evaluates 500 races in parallel instead of one-by-one

Run on Colab:
  Runtime → Change runtime type → T4 GPU
  Then run cells top to bottom.

Install:
  pip install jax[cuda12] optax tqdm   # Colab already has JAX, just needs optax
"""

# ═══════════════════════════════════════════════════════════
# CELL 1 — Imports & GPU check
# ═══════════════════════════════════════════════════════════

import json, time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import optax
from scipy.optimize import differential_evolution, minimize
from tqdm.auto import tqdm

# Confirm GPU
print("JAX devices:", jax.devices())
# Should show: [CudaDevice(id=0)]  on Colab T4
# If shows CpuDevice, go to Runtime → Change runtime type → T4 GPU

DATA_DIR = 'data/historical_races'
OUTPUT   = 'solution/fitted_params.json'
TEMP_REF = 30.0

PARAM_NAMES = ['soft_offset', 'med_offset',
               'soft_deg',    'med_deg',    'hard_deg',
               'soft_cliff',  'med_cliff',  'hard_cliff',
               'temp_scale']

BOUNDS = [(-5.0, -0.01), (-4.0, -0.01),
          (0.001, 0.6),  (0.001, 0.4),  (0.001, 0.3),
          (0.0,  30.0),  (0.0,  40.0),  (0.0,  60.0),
          (-0.05, 0.1)]

COMPOUND_IDX = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}


# ═══════════════════════════════════════════════════════════
# CELL 2 — Load & precompute races into GPU-ready arrays
# ═══════════════════════════════════════════════════════════

def load_races(n):
    races = []
    for rf in sorted(Path(DATA_DIR).glob('*.json')):
        with open(rf) as f:
            races.extend(json.load(f))
        if len(races) >= n:
            break
    return races[:n]


def races_to_arrays(races):
    """
    Convert races into fixed-size numpy arrays that JAX can load onto GPU.

    Returns:
      compound_seq : (R, D, L)  int32   — compound index per lap
      tire_age_seq : (R, D, L)  float32 — tire age per lap
      fixed_cost   : (R, D)     float32 — base*laps + pit penalties (param-free)
      track_temp   : (R,)       float32 — track temperature
      expected_rank: (R, D)     int32   — finishing rank of each driver (0=1st)
      driver_order : (R, D)     str     — driver IDs per race (for decoding)

    All sequences are padded to max_laps with zeros.
    """
    R = len(races)
    D = 20
    max_laps = max(r['race_config']['total_laps'] for r in races)

    compound_seq = np.zeros((R, D, max_laps), dtype=np.int32)
    tire_age_seq = np.zeros((R, D, max_laps), dtype=np.float32)
    fixed_cost   = np.zeros((R, D),           dtype=np.float32)
    track_temp   = np.zeros((R,),             dtype=np.float32)
    lap_count    = np.zeros((R,),             dtype=np.int32)
    expected_rank= np.zeros((R, D),           dtype=np.int32)

    for ri, race in enumerate(races):
        cfg   = race['race_config']
        base  = cfg['base_lap_time']
        laps  = cfg['total_laps']
        pit_t = cfg['pit_lane_time']
        temp  = cfg['track_temp']

        track_temp[ri] = temp
        lap_count[ri]  = laps

        finish  = race['finishing_positions']
        rank_of = {did: pos for pos, did in enumerate(finish)}

        strat_list = [race['strategies'][f'pos{i+1}'] for i in range(D)]

        for di, s in enumerate(strat_list):
            did  = s['driver_id']
            tire = s['starting_tire']
            pits = {ps['lap']: ps['to_tire'] for ps in s.get('pit_stops', [])}

            cost = base * laps
            age  = 0

            for lap in range(1, laps + 1):
                age += 1
                li   = lap - 1
                compound_seq[ri, di, li] = COMPOUND_IDX[tire]
                tire_age_seq[ri, di, li] = age

                if lap in pits:
                    cost += pit_t
                    tire  = pits[lap]
                    age   = 0

            fixed_cost[ri, di]    = cost - base * laps  # pit penalties only
            expected_rank[ri, di] = rank_of[did]

        # Store base*laps in fixed_cost properly
        fixed_cost[ri] += base * laps

    return {
        'compound_seq':  jnp.array(compound_seq),
        'tire_age_seq':  jnp.array(tire_age_seq),
        'fixed_cost':    jnp.array(fixed_cost),
        'track_temp':    jnp.array(track_temp),
        'lap_count':     jnp.array(lap_count),
        'expected_rank': jnp.array(expected_rank),
    }


print("Loading 2000 races...")
t0 = time.time()
all_races = load_races(2000)
print(f"Loaded {len(all_races)} races in {time.time()-t0:.1f}s")

print("Converting to GPU arrays...")
t0 = time.time()
arrays = races_to_arrays(all_races)
print(f"Converted in {time.time()-t0:.1f}s")
print(f"compound_seq shape: {arrays['compound_seq'].shape}")
print(f"  = {arrays['compound_seq'].shape[0]} races × "
      f"{arrays['compound_seq'].shape[1]} drivers × "
      f"{arrays['compound_seq'].shape[2]} laps")


# ═══════════════════════════════════════════════════════════
# CELL 3 — JAX loss function (JIT + vmap)
# ═══════════════════════════════════════════════════════════

@jit
def loss_single_race(params, compound_seq, tire_age_seq,
                     fixed_cost, track_temp, lap_count, expected_rank):
    """
    Loss for ONE race. JAX-compiled, runs in microseconds.

    params shape: (9,)
      [0] soft_offset  [1] med_offset
      [2] soft_deg     [3] med_deg    [4] hard_deg
      [5] soft_cliff   [6] med_cliff  [7] hard_cliff
      [8] temp_scale
    """
    offsets   = jnp.array([params[0], params[1], 0.0])
    deg_rates = jnp.array([params[2], params[3], params[4]])
    cliffs    = jnp.array([params[5], params[6], params[7]])
    temp_mult = 1.0 + params[8] * (track_temp - TEMP_REF)

    # compound_seq: (D, L), tire_age_seq: (D, L)
    # Gather per-compound values for every lap of every driver
    c_offset  = offsets  [compound_seq]   # (D, L)
    c_deg     = deg_rates[compound_seq]   # (D, L)
    c_cliff   = cliffs   [compound_seq]   # (D, L)

    # Degradation per lap (0 before cliff, linear after)
    eff_age   = jnp.maximum(0.0, tire_age_seq - c_cliff)   # (D, L)
    deg_cost  = c_deg * eff_age * temp_mult                 # (D, L)

    # Lap cost = offset + degradation  (base is already in fixed_cost)
    lap_cost  = c_offset + deg_cost                         # (D, L)

    # Zero out padding laps (lap index >= lap_count)
    lap_mask  = jnp.arange(compound_seq.shape[1]) < lap_count  # (L,)
    lap_cost  = lap_cost * lap_mask[None, :]                    # (D, L)

    # Total time per driver
    total_time = fixed_cost + lap_cost.sum(axis=1)  # (D,)

    # Pairwise ranking loss: for each adjacent pair in expected order,
    # penalise if faster driver (lower rank) has higher total time
    margin = 0.001

    # Sort total_time by expected rank to get winner first
    sorted_times = total_time[jnp.argsort(expected_rank)]  # (D,)

    # Adjacent diff: sorted_times[i] should < sorted_times[i+1]
    diffs = sorted_times[:-1] - sorted_times[1:]  # (D-1,)  want all < 0
    violations = jnp.maximum(0.0, diffs + margin)
    return jnp.sum(violations ** 2)


# vmap over the race dimension — evaluates ALL races simultaneously on GPU
loss_all_races = vmap(
    loss_single_race,
    in_axes=(None, 0, 0, 0, 0, 0, 0)   # params fixed, all arrays race-batched
)


@jit
def total_loss(params, arrays):
    """Sum loss across all races. Single GPU call."""
    per_race = loss_all_races(
        params,
        arrays['compound_seq'],
        arrays['tire_age_seq'],
        arrays['fixed_cost'],
        arrays['track_temp'],
        arrays['lap_count'],
        arrays['expected_rank'],
    )
    return jnp.sum(per_race)


# ── Test the compiled function ────────────────────────────────────────────────
print("\nWarm-up compile (first call triggers JIT — takes ~10s)...")
test_params = jnp.array([-1.5, -0.7, 0.08, 0.04, 0.02,
                          5.0, 12.0, 20.0, 0.015])
t0 = time.time()
_ = total_loss(test_params, arrays).block_until_ready()
print(f"First call (JIT compile): {time.time()-t0:.1f}s")

t0 = time.time()
for _ in range(20):
    _ = total_loss(test_params, arrays).block_until_ready()
avg = (time.time()-t0)/20*1000
print(f"Compiled call (avg over 20): {avg:.1f}ms per evaluation")
print(f"→ 1000 DE evaluations would take: {avg*1000/1000:.1f}s")


# ═══════════════════════════════════════════════════════════
# CELL 4 — Phase 1: Scipy DE with JAX loss (parallel candidates)
# ═══════════════════════════════════════════════════════════

print("\n" + "="*55)
print("Phase 1 — Differential Evolution (scipy + JAX loss)")
print("="*55)

# Scipy DE calls the loss for each candidate.
# Each call is now a fast GPU operation (~Xms).
eval_count = [0]
best_loss  = [float('inf')]
t_start    = time.time()

pbar = tqdm(total=150, desc="DE generations", ncols=65)

def de_callback(xk, convergence=0):
    pbar.update(1)
    loss_val = float(total_loss(jnp.array(xk, dtype=jnp.float32), arrays))
    if loss_val < best_loss[0]:
        best_loss[0] = loss_val
    pbar.set_postfix({'loss': f'{best_loss[0]:.5f}'})

def scipy_loss(p_np):
    """Wrapper: numpy array → JAX → scalar float."""
    p_jax = jnp.array(p_np, dtype=jnp.float32)
    return float(total_loss(p_jax, arrays))

t0 = time.time()
de_result = differential_evolution(
    scipy_loss,
    BOUNDS,
    maxiter      = 150,
    tol          = 1e-12,
    seed         = 42,
    popsize      = 15,
    mutation     = (0.5, 1.5),
    recombination= 0.7,
    workers      = 1,      # JAX handles parallelism on GPU; scipy workers=1
    init         = 'sobol',
    callback     = de_callback,
    polish       = False,
    disp         = False,
)
pbar.close()
print(f"\nDE done in {time.time()-t0:.1f}s  |  loss={de_result.fun:.6f}")


# ═══════════════════════════════════════════════════════════
# CELL 5 — Phase 2: Gradient descent with JAX autodiff
# ═══════════════════════════════════════════════════════════

print("\n" + "="*55)
print("Phase 2a — JAX gradient descent (Adam via Optax)")
print("  JAX can compute exact gradients of our formula via autodiff.")
print("  This is MUCH faster than scipy's finite-difference approximation.")
print("="*55)

# Use JAX grad to get exact gradients of total_loss w.r.t. params
grad_fn = jit(grad(total_loss))

# Optax Adam optimiser
LR     = 3e-4
N_ITER = 800

params = jnp.array(de_result.x, dtype=jnp.float32)
opt    = optax.adam(LR)
opt_state = opt.init(params)

# Clip params to bounds after each step
lo = jnp.array([b[0] for b in BOUNDS], dtype=jnp.float32)
hi = jnp.array([b[1] for b in BOUNDS], dtype=jnp.float32)

@jit
def step(params, opt_state):
    g = grad_fn(params, arrays)
    updates, new_state = opt.update(g, opt_state)
    new_params = optax.apply_updates(params, updates)
    new_params = jnp.clip(new_params, lo, hi)
    return new_params, new_state

print(f"Running {N_ITER} Adam steps...")
t0 = time.time()
loss_history = []
pbar2 = tqdm(range(N_ITER), ncols=65, desc="Adam steps")
for i in pbar2:
    params, opt_state = step(params, opt_state)
    if i % 50 == 0:
        lv = float(total_loss(params, arrays))
        loss_history.append(lv)
        pbar2.set_postfix({'loss': f'{lv:.6f}'})

print(f"Adam done in {time.time()-t0:.1f}s  |  loss={float(total_loss(params, arrays)):.8f}")

# ── Phase 2b: L-BFGS-B polish ────────────────────────────────────────────────
print("\nPhase 2b — L-BFGS-B polish...")
t0 = time.time()

def scipy_loss_and_grad(p_np):
    p_jax  = jnp.array(p_np, dtype=jnp.float32)
    lv     = float(total_loss(p_jax, arrays))
    gv     = np.array(grad_fn(p_jax, arrays), dtype=np.float64)
    return lv, gv

local = minimize(
    scipy_loss_and_grad,
    np.array(params, dtype=np.float64),
    jac     = True,      # we return (loss, grad) together — faster
    method  = 'L-BFGS-B',
    bounds  = BOUNDS,
    options = {'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-12},
)
best = local.x
print(f"L-BFGS-B done in {time.time()-t0:.1f}s  |  loss={local.fun:.10f}")


# ═══════════════════════════════════════════════════════════
# CELL 6 — Validation & save
# ═══════════════════════════════════════════════════════════

def predict_race(race, p):
    cfg   = race['race_config']
    base  = cfg['base_lap_time']
    laps  = cfg['total_laps']
    pit_t = cfg['pit_lane_time']
    temp  = cfg['track_temp']
    offsets   = {'SOFT': p[0], 'MEDIUM': p[1], 'HARD': 0.0}
    deg_rates = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    cliffs    = {'SOFT': p[5], 'MEDIUM': p[6], 'HARD': p[7]}
    temp_mult = 1.0 + p[8] * (temp - TEMP_REF)

    times = {}
    for s in race['strategies'].values():
        did  = s['driver_id']
        tire = s['starting_tire']
        pits = {ps['lap']: ps['to_tie'] for ps in s.get('pit_stops', [])}
        # fix key name
        pits = {ps['lap']: ps['to_tire'] for ps in s.get('pit_stops', [])}
        t = 0.0; age = 0
        for lap in range(1, laps + 1):
            age += 1
            eff  = max(0.0, age - cliffs[tire])
            t   += base + offsets[tire] + deg_rates[tire] * eff * temp_mult
            if lap in pits:
                t   += pit_t
                tire = pits[lap]
                age  = 0
        times[did] = t
    return sorted(times, key=lambda d: times[d])


print("\nValidating on 500 held-out races...")
val_races = load_races(2500)[-500:]
p_list    = best.tolist()
correct   = sum(
    predict_race(r, p_list) == r['finishing_positions']
    for r in tqdm(val_races, desc="Validating", ncols=55)
)
acc = correct / len(val_races)
bar = '█' * int(acc * 40) + '░' * (40 - int(acc * 40))
print(f"\nAccuracy: [{bar}] {acc*100:.1f}%")

# Print params with sanity checks
print("\nFitted parameters:")
checks = [best[0]<0, best[1]<0, best[2]>0, best[3]>0, best[4]>0,
          best[5]<best[6], best[6]<best[7], True, best[8]>0]
for name, val, ok in zip(PARAM_NAMES, best, checks):
    mark = '✓' if ok else '⚠ CHECK'
    print(f"  {mark}  {name:<15} = {val:+.6f}")

# Save
out = dict(zip(PARAM_NAMES, best.tolist()))
Path('solution').mkdir(exist_ok=True)
with open(OUTPUT, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {OUTPUT}")
print("Download this file and put it in solution/ before submitting.")
