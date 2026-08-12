# Performance Tracking — `examples/aircraft_design` quantify scripts

Goal: bring `quantify.py` (numpy) and `quantify_JAX.py` to a comparable, repeatable state and
close the gap — `quantify_JAX.py` is currently far slower.

All timings: Windows 11, wall-clock around `python <script>`, runs strictly sequential so they
don't contend for CPU. Plots suppressed via `MPLBACKEND=Agg`.

---

## Changes made

### 1. Dropped `force_alloc_complex=True` (both scripts)

Complex-step allocation doubles vector memory and disables some fast paths; nothing in either
script uses `method='cs'` anymore (the `check_partials` calls are commented out).

- `quantify.py:32` — `prob.setup(force_alloc_complex=True)` -> `prob.setup()`
- `quantify.py:260` — same, uncertain problem
- `quantify_JAX.py:256` — same, deterministic problem
- `quantify_JAX.py:484` — same, uncertain problem

> Caveat: re-enabling either commented-out `check_partials(..., method='cs')` will now fail.
> Switch it to `method='fd'` or restore the flag for that run.

### 2. Seeded the numpy RNG (both scripts)

UQPCE's resampled variable basis is drawn from the *global* numpy RNG
(`uqpce/mdao/uqpcegroup.py:213-214`), so each run previously took a different optimization
path, making timings non-comparable.

- `quantify.py` — added `import numpy as np`; `np.random.seed(0)` as first statement of `main()`
- `quantify_JAX.py` — `np.random.seed(0)` as first statement of `main()`

Set before `interface.initialize()` / `UQPCEGroup` construction, where the draws happen.

### 3. Replaced `ArmijoGoldsteinLS` with `BoundsEnforceLS` on the AeroStruct Newton solver

**Biggest win so far — ~3x on numpy, ~5x on JAX.**

The ArmijoGoldstein linesearch was already attached in both paths (it did not need adding). Its
backtracking was rejecting the full Newton step on nearly every iteration, collapsing quadratic
convergence into a linear contraction — the residual halved at *exactly* 0.5 per iteration, and
near the optimum degraded to ~0.99, producing ~700-iteration solves.

- `organize.py:94` (numpy path)
- `quantify_JAX.py:102` (JAX path)

Both keep `bound_enforcement='vector'` and `print_bound_enforce`. Bounds enforcement itself is
load-bearing: with **no** linesearch at all the run dies with
`RuntimeError: NaN entries found in 'AeroStruct'` (singular Jacobian).

Convergence before vs. after, same first solve:

```
ArmijoGoldstein                     BoundsEnforceLS
NL: Newton 0 ; 15662.4              NL: Newton 0 ; 15662.4
NL: Newton 1 ; 296.47               NL: Newton 1 ; 5.3298
NL: Newton 2 ; 5.586                NL: Newton 2 ; 1.7132
NL: Newton 3 ; 0.10700              NL: Newton 3 ; 0.30554
NL: Newton 4 ; 0.010365             NL: Newton 4 ; 0.0097384
NL: Newton 5 ; 0.005074  <- 0.5x    NL: Newton 5 ; 1.5611e-05  Converged
NL: Newton 6 ; 0.002532  <- 0.5x
...continues halving to ~700
```

Linesearch comparison (`quantify.py`, one run each):

| Linesearch | Time | Newton iters | Solves | Exit |
|---|---|---|---|---|
| ArmijoGoldstein (was) | 38.22s | 5860 | 96 | 0 |
| **BoundsEnforceLS (now)** | **10.67s** | **846** | 99 | 0 |
| none | 7.68s | 96 | 45 | 1 (NaN crash) |

> Result note: `lambd_50` moved from `0.020696715222417356` to `0.020695741286746148`
> (4.7e-5 relative).
>
> **Correction (after round 5):** this was originally recorded here as the new value being
> *more* converged. That was wrong. Once `res_ref` scaling forced a tight solve (residual to
> 9.7e-12), `lambd_50` came back to `0.020696712556301372` — which agrees with the **original**
> Armijo value to ~1.3e-7 relative, not with the BoundsEnforceLS-only value. So the
> BoundsEnforceLS-only run was the under-converged outlier, and the original Armijo result was
> closer to correct all along. The linesearch swap is still a large, real speedup; it just also
> silently loosened the solve until change 4 restored the accuracy.

### 4. Added `res_ref` scaling to the AeroStruct residuals

**Bought accuracy, not speed.** Initial residual norm 15662 -> 2.758 (5680x), but wall time was
unchanged (see round 5).

The commented-out `res_ref=1.0` on the balance was **not** the lever: `normalize=True` already
held the `m_fuel` residual at ~1.0. A residual diagnostic at the initial point showed the norm
was 99.8% unscaled *explicit* outputs:

| residual | norm | share of norm^2 |
|---|---|---|
| `Weight.m_empty` [kg] | 14593.6 | 86.82% |
| `Aero.WL` [N/m**2] | 5639.25 | 12.96% |
| `Weight.m_wing` [kg] | 731.229 | 0.22% |
| `Aero.LD` [unitless] | 15.5087 | 0.00% |
| `Balance.m_fuel` [kg] | 0.999985 | 0.00% |

So `res_ref` went on those `add_output` calls, in both backends:

- `disciplines/weight.py:39-40` — `m_empty` 1e4, `m_wing` 1e3
- `disciplines/aero.py:41-42` — `LD` 10, `WL` 5e3
- `disciplines_JAX/WeightsCompJAX.py:44-45`, `disciplines_JAX/aero_jax.py:41-42` — mirrored
- `organize.py:77`, `quantify_JAX.py:85` — balance `res_ref=1.0` uncommented as requested
  (near-neutral, but makes the scaling explicit)

`LD` needed scaling too: negligible at ~15.5 before, but it would have become the *largest*
remaining residual once the big three were scaled.

Convergence after scaling — still quadratic, and now reaching 9.7e-12 where the
BoundsEnforceLS-only run stopped at 1.6e-5, because `atol=1e-8` against a smaller-scaled
residual demands a tighter *physical* solve:

```
NL: Newton 0 ; 2.7577      NL: Newton 4 ; 0.0022250
NL: Newton 1 ; 0.34798     NL: Newton 5 ; 3.6302e-06
NL: Newton 2 ; 0.40094     NL: Newton 6 ; 9.7144e-12  Converged
NL: Newton 3 ; 0.058662
```

### 5. Plot suppression — harness-only so far, NOT a code change

`helpers.py` ends every `plot_*` with a blocking `plt.show()` and never calls `savefig`, so an
interactive window stalls any timed run. The harness exports `MPLBACKEND=Agg`, making
`plt.show()` a no-op. No script edit needed, and it applies equally to the baselines, so it does
not bias the comparison.

**Open decision:** leave as env var (scripts stay interactive by hand) vs. add a `--no-plots`
flag or switch the helpers to `savefig`.

---

## Timing results

### Round 1 — baseline (git HEAD) vs. modified

| Variant | Time | vs. baseline |
|---|---|---|
| baseline numpy | 41.28s | — |
| modified numpy | 34.67s | −6.6s (−16%) |
| baseline JAX | 228.37s | — |
| modified JAX | 217.21s | −11.2s (−4.9%) |

**JAX is 6.3× slower than numpy** on the modified versions (217.21 vs 34.67). Consistent across
baseline and modified, so it is a real property of the JAX path, not an artifact of our changes.

> The two speedup figures are **not yet trustworthy**. Seeding changes the optimizer's path, so
> a baseline-vs-modified delta mixes the flag's effect with path variance. Round 2 isolates it.

### Round 2 — isolating the flag

Same seed, so `force_alloc_complex` is the only difference; plus a repeat of each seeded
config to measure run-to-run noise.

| Config | numpy | JAX |
|---|---|---|
| seeded + complex | 35.83s | 228.80s |
| seeded, no complex | 34.67s / 32.18s | 217.21s / 205.15s |
| **apparent flag cost** | +2.4s vs 2.5s noise | +17.6s (~8%) vs 12.1s noise |

The JAX effect is probably real; the numpy effect is inside the noise. Neither is nailed down —
one sample per config, and see the determinism problem below.

**Keep the change regardless:** it is free and provably result-neutral (bit-identical `lambd_50`).

### Round 3 — determinism test (single-threaded BLAS)

`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`, `PYTHONHASHSEED=0`, `quantify.py` twice:

| Run | S | AR | SFC_tech | DOC mean | Time |
|---|---|---|---|---|---|
| run 1 | 140.0237 | 15.4098 | 0.00247282 | 55702.4 | 36.57s |
| run 2 | 139.8434 | 15.4350 | 0.00398580 | 55685.8 | 29.79s |

Did **not** fix it. Also note the 6.8s spread (~20%) on identical single-threaded runs — timing
noise here is large enough that no small delta above is trustworthy without repeats.

### Round 4 — with `BoundsEnforceLS` (current state of both scripts)

Two reps each, sequential, `MPLBACKEND=Agg`:

| Variant | Run 1 | Run 2 | Newton iters |
|---|---|---|---|
| `quantify.py` (numpy) | 12.40s | 10.30s | 853 / 845 |
| `quantify_JAX.py` | 42.43s | 37.47s | 852 / 851 |

**Cumulative, vs. the original unmodified scripts:**

| | original | now | speedup |
|---|---|---|---|
| numpy | 41.28s | ~11.4s | **3.6x** |
| JAX | 228.37s | ~40.0s | **5.7x** |
| **JAX/numpy gap** | **6.3x** | **3.5x** | — |

Two things worth pulling out:

- The linesearch fix dwarfs the `force_alloc_complex` change (which was within noise on numpy
  and ~8% on JAX). Nearly all of the gain above is the linesearch.
- Much of what looked like *JAX overhead* was JAX paying ~6x the price for the same wasted
  Newton iterations. Removing the waste cut the backend gap from 6.3x to 3.5x. The remaining
  3.5x is the real per-call JAX dispatch cost and is the next thing worth profiling.
- Newton iteration counts now match closely across backends (853/845 vs 852/851), confirming
  both paths are doing the same work.

### Round 5 — with `res_ref` scaling (current state of both scripts)

Two reps each, sequential, `MPLBACKEND=Agg`:

| Variant | Run 1 | Run 2 | Newton iters | vs. round 4 |
|---|---|---|---|---|
| `quantify.py` (numpy) | 10.56s | 11.90s | 861 / 859 | unchanged (within noise) |
| `quantify_JAX.py` | 39.02s | 41.11s | 859 / 850 | unchanged (within noise) |

**No speed gain from scaling** — round 4 was 12.40/10.30s (numpy) and 42.43/37.47s (JAX), and
the spreads overlap completely. Newton iteration counts even rose slightly (853/845 -> 861/859),
which is expected: the tighter effective tolerance costs about one extra iteration per solve.

What it bought instead is **accuracy**. `lambd_50` is now `0.020696712556301372` (numpy) /
`0.02069671255618622` (JAX), agreeing with each other to 1e-11 and with the original
pre-change value to ~1.3e-7 — versus the 4.7e-5 discrepancy the linesearch swap had introduced.
Worth keeping for that reason, not for throughput.

**Cumulative, vs. the original unmodified scripts:**

| | original | now | speedup |
|---|---|---|---|
| numpy | 41.28s | ~11.2s | **3.7x** |
| JAX | 228.37s | ~40.1s | **5.7x** |
| **JAX/numpy gap** | **6.3x** | **3.6x** | — |

---

## OPEN PROBLEM: runs are still not reproducible

**The seed did not deliver run-to-run repeatability.** Rerunning the *same* seeded script twice
gives different optima:

| `quantify.py`, identical seed | S | AR | DOC mean |
|---|---|---|---|
| run A | 140.1776 | 15.5437 | 55745.5 |
| run B | 139.7506 | 15.4297 | 55685.4 |

What we know:

- `lambd_50` is **bit-identical across all numpy runs** within a threading config, so the seed
  *did* fix the resampled basis and everything up through `run_model` is reproducible.
- The divergence is therefore entirely inside `run_driver` (the SLSQP optimization).
- Multithreaded-BLAS nondeterminism is **ruled out** — pinning to 1 thread did not fix it.
  (It did shift `lambd_50` to `0.020696715222429895`, consistent with threading affecting FP
  reduction order, but both single-threaded runs share that value and still diverge.)
- `PYTHONHASHSEED=0` did not fix it either.

Candidate causes not yet tested:

- The Newton solve runs to ~688 iterations against `maxiter=700`. If some evaluations hit the
  cap and others don't, tiny FP differences flip convergence and change the optimizer's path.
  Raising `maxiter` / loosening `atol`+`rtol`, or fixing whatever makes convergence so slow,
  may remove the knife-edge.
- Something in the UQPCE CDF groups reseeds the global RNG mid-run
  (`uqpce/mdao/cdfresidcomp.py:84`, `uqpce/mdao/cdfgroup.py:98` both call `np.random.seed(1)`).
  Deterministic in isolation, but it means RNG state depends on call *order* and *count*.

Suggested next step: record the objective at each driver iteration in two runs and diff them to
find the first differing evaluation, rather than guessing.

### Update after round 4 (BoundsEnforceLS) — still unresolved

The maxiter knife-edge candidate above is now **largely ruled out**: solves converge in ~5
iterations instead of ~700, nowhere near `maxiter=700`, yet the four round-4 runs still land on
different optima:

| Round-4 run | S | DOC mean |
|---|---|---|
| numpy r1 | 139.5759 | 55680.9 |
| numpy r2 | 139.9658 | 55730.6 |
| JAX r1 | 140.0178 | 55696.4 |
| JAX r2 | 140.0104 | 55716.1 |

`lambd_50` remains bit-identical *within* each backend (numpy `0.020695741286746148`, JAX
`0.020695741286744392`), so `run_model` is still fully reproducible and the divergence is still
confined to `run_driver`. The RNG-reseeding candidate is now the leading suspect; the
objective-per-iteration diff remains the right next step.

---

## Correctness checks

- `lambd_50` is **bit-identical** baseline-vs-modified within each backend
  (numpy `0.020696715222417356`, JAX `0.020696715222464617`) — confirms removing
  `force_alloc_complex` changed no results.
- numpy and JAX agree to ~1e-12 on that value, so **JAX is running float64**; the 6.3× gap is
  not a precision-mode issue.

---

## Open leads on the JAX gap

**Status after round 4:** the Newton-iteration lead below turned out to be the dominant cost and
is now **fixed** (see change 3) — it cut the JAX/numpy gap from 6.3x to 3.5x. The remaining ~3.5x
is genuine per-call JAX overhead and is now the top open lead.

- **Newton iteration count.** *(RESOLVED — see change 3.)* `CoupledDisciplines` converged at
  iteration 688 against
  `maxiter=700` — near the ceiling — and `iprint=2` prints every iteration. Both the iteration
  count and the console I/O hit both scripts equally, likely inflating absolute numbers and
  compressing the apparent JAX gap. Worth attacking next.
- Not yet investigated: per-call JAX dispatch/tracing overhead on small vectors, and whether
  the JAX components are re-tracing rather than reusing compiled code. **Now the top lead** —
  with iteration counts matched across backends (853/845 vs 852/851), the residual 3.5x is
  per-call cost, not extra work.
- **Balance residual scaling.** *(DONE — see change 4, and note the diagnosis here was wrong:
  the balance residual was already O(1); the magnitude came from unscaled explicit outputs
  `m_empty`/`WL`/`m_wing`. Fixing it improved accuracy but gave no speedup.)* The residual
  spans ~12 orders of magnitude
  (starts ~15662, `atol=1e-8`) because it is an unnormalized range error in meters; `res_ref`
  is commented out at `organize.py:77`. Independent of the linesearch, and still worth trying.

---

## Scratch files to delete before committing

Created in `examples/aircraft_design/` for the comparison, all untracked:
`baseline_quantify.py`, `baseline_quantify_JAX.py`, `isolate_quantify.py`,
`isolate_quantify_JAX.py`
