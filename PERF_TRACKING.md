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

### 5. Declared sparse partials on all JAX components

**Biggest JAX-side win — 39-41s -> ~25s, and it was a missing-declaration bug, not JAX overhead.**

Profiling (cProfile, `quantify_JAX.py` vs `quantify.py`) showed the hotspot was not JAX at all:

| | JAX run | numpy run |
|---|---|---|
| `_superlu.gstrf` (sparse LU) | **20.49s** / 1035 calls | **0.25s** / 1043 calls |
| `backend_compile_and_load` | 4.11s / 126 | 1.98s / 73 |
| `csr_matrix._update_from_submat` | 1.51s / 46632 | 0.22s / 46544 |
| total (profiled, inflated) | 46.4s | 15.0s |

Sparse LU was 44% of the JAX run at ~82x the per-call cost with an identical call count — the
signature of a much denser matrix. Confirmed by measuring the assembled AeroStruct Jacobian:

| backend | dr_do nnz | density |
|---|---|---|
| numpy (before) | 3,120 | 0.158% |
| JAX (before) | **269,100** | **13.65%** |
| JAX (after fix) | **3,120** | **0.158%** |

**Cause:** all eight `disciplines_JAX/*.py` components declared no partials, so OpenMDAO built
dense 156x156 subjacobians (n=156 samples) where the true structure is diagonal. The numpy
components have always declared sparse diagonals (`disciplines/weight.py:56-62`; 35 such calls
in `disciplines/aero.py`).

**Fix:** added `setup_partials` to all eight JAX components declaring
`rows=arange, cols=arange` for vector inputs and dense for scalar inputs, mirroring the numpy
side exactly.

> **On the re-jitting hypothesis:** ruled out. `JAX_LOG_COMPILES=1` shows 126 compiles, each
> cached after first use, 4.1s total — one-time, not per-iteration. (Side note: the *numpy* run
> also does 73 JAX compiles for ~2.0s, so something in the UQPCE stack pulls in JAX regardless.)
> The compiles are of individual primitives (`jit(log)`, `jit(multiply)` on `float64[1]`), i.e.
> op-by-op dispatch rather than one fused kernel per component — worth a separate look, but it
> was not the dominant cost.

> **`declare_coloring()` was tried first and did not work.** It ran on every component but
> reported *"Coloring was deactivated. Improvement of 0.0% was less than min allowed (5.0%)"* —
> automatic sparsity detection failed to find the diagonal structure of elementwise
> `compute_primal` functions. It was removed in favor of explicit declarations since it added
> startup cost for zero benefit. Possibly worth an upstream OpenMDAO report.

> **Fairness caveat:** this means the two backends were never solving structurally equivalent
> linear systems. A large share of the original "JAX is far slower" gap was this missing
> declaration, not JAX. Only the post-fix numbers are a fair backend comparison.

### 6. OpenMDAO fix: numpy-ify jax derivs before sparse indexing (1.5x on JAX)

**This change is in the OpenMDAO checkout, not in this repo.**
`C:/Users/robfa/Codes/OpenMDAO.git`, `openmdao/utils/jax_utils.py:1093`, uncommitted:

```python
# before -- fancy-indexes a JAX array, routing every gather through jax dispatch
partials[ofname, wrtname] = dvals[rows, sjmeta['cols']]
# after
partials[ofname, wrtname] = np.asarray(dvals)[rows, sjmeta['cols']]
```

Re-profiling after change 5 showed `_jax_derivs2partials` at **20.6s cumulative of a 40.3s
profiled run (51%)**, with `_index_to_gather` alone at 13.0s over 20,473 calls. The branch only
runs when a component declares sparse partials (`rows is not None`) — i.e. change 5 is what
activated it. The two fixes compound: sparsity makes the LU cheap, this makes the extraction
cheap.

Measured via runtime monkeypatch before patching: 25.26s -> 17.56s / 16.57s, with `lambd_50`
**bit-identical** (`0.020696712556186183`). Purely a performance change.

> Worth reporting upstream — it affects any jax component with declared sparsity, not just this
> example. `expt_fastpartials.py` in the example dir holds the standalone monkeypatch
> reproduction.

### 7. Disabled `print_bound_enforce` — log hygiene, NOT a speedup

`organize.py`, `quantify_JAX.py`: `print_bound_enforce` True -> False.

Log volume dropped from ~39,300 lines to ~1,440 (96%), because the option printed a full
156-element array ~700 times per run.

> **Correction:** this was initially predicted to be worth ~1s (~7%) based on cProfile showing
> `arrayprint.recurser` at 0.99s cumulative and 437,786 `dragon4_positional` calls. Measurement
> did not support that — numpy 10.36/12.10s -> 11.01/10.60s, JAX 25.07/24.92s -> 26.11/24.65s,
> i.e. no change on either side. cProfile exaggerates formatting cost and redirected stdout is
> cheap. Kept for readability only; **do not count it as a performance win.**

### 8. `m_fuel` bound experiment — TRIED AND REVERTED

Setting the balance `upper` from 100000.0 to 50000.0 was a **regression on both axes** and was
reverted:

| | upper=100000 | upper=50000 |
|---|---|---|
| Newton iters | ~860 | **~3,640** (4.2x) |
| bound clips | 697 | **3,500** (5x) |
| numpy | ~10.4s | 13.3 / 13.4s |
| JAX | ~16.8s | 25.2 / 25.4s |

Worse, **the cap was binding on a real solution**: one of the 156 samples converged at exactly
50000.0, i.e. a physically valid state was being clamped. It showed in the output —
`lambd_50` moved to `0.0206992464` from `0.0206967126` (~1e-6 relative).

> **Correction to an earlier note in this file:** converged `m_fuel` is *not* ~16,000 kg. That
> figure was the deterministic single-point case. Across the 156 uncertainty samples it is
> min 15,634 / **mean 32,021** / max >50,000. So the original 100000.0 bound is ~2x headroom,
> not ~6x, and 50000.0 has effectively none.

### 9. Balance `ref` retune 20000 -> 32000 — NEUTRAL, kept on principle

`organize.py:76`, `quantify_JAX.py:84`. Hypothesis: `ref=20000` put the true state (~32,000 kg
mean) at 1.6 in scaled space, so Newton's scaled steps overshoot into the upper bound.
Matching `ref` to the actual magnitude should size the steps proportionately.

**The hypothesis did not hold.** Measured against `ref=20000`:

| | ref=20000 | ref=32000 |
|---|---|---|
| bound clips | 697 | 700 |
| Newton iters | ~860 | ~857 |
| numpy | ~10.4s | 10.31 / 9.75s |
| JAX | ~16.8s | 15.48 / 15.05s |

Clips and iterations are unchanged, so `ref` is **not** the mechanism behind the clipping. The
JAX times look slightly better but the spread overlaps the previous round — treat as noise, not
a win. `lambd_50` = `0.020696712556270848` vs `0.020696712556301372` before, agreeing to 1e-12,
confirming this is pure scaling with nothing clamped.

Kept because matching `ref` to the actual state magnitude is the more defensible setting, but it
bought nothing measurable. Reverting it would cost nothing either.

**The ~700 bound clips per run remain unexplained and open.**

### 10. Plot suppression — harness-only so far, NOT a code change

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

### Round 6 — with sparse JAX partials (current state of both scripts)

Two reps each, sequential, `MPLBACKEND=Agg`:

| Variant | Run 1 | Run 2 | Newton iters | vs. round 5 |
|---|---|---|---|---|
| `quantify.py` (numpy) | 10.36s | 12.10s | 858 / 860 | unchanged (control) |
| `quantify_JAX.py` | **25.07s** | **24.92s** | 863 / 854 | **39-41s -> ~25s (1.6x)** |

An intermediate attempt with `declare_coloring()` measured 44.22s / 40.96s — i.e. no better than
round 5, consistent with the coloring self-deactivating.

Cross-backend agreement holds: `lambd_50` = `0.020696712556186183` (JAX) vs
`0.020696712556301372` (numpy), agreeing to ~1e-11.

**Cumulative, vs. the original unmodified scripts:**

| | original | now | speedup |
|---|---|---|---|
| numpy | 41.28s | ~11.2s | **3.7x** |
| JAX | 228.37s | ~25.0s | **9.1x** |
| **JAX/numpy gap** | **6.3x** | **2.2x** | — |

### Round 7 — with the OpenMDAO jax_utils patch (current state)

Two reps each, sequential, `MPLBACKEND=Agg`:

| Variant | Run 1 | Run 2 | vs. round 6 |
|---|---|---|---|
| `quantify.py` (numpy) | 11.22s | 9.64s | unchanged (control) |
| `quantify_JAX.py` | **17.63s** | **15.89s** | **~25s -> ~16.8s (1.5x)** |

**Cumulative, vs. the original unmodified scripts:**

| | original | now | speedup |
|---|---|---|---|
| numpy | 41.28s | ~10.4s | **4.0x** |
| JAX | 228.37s | ~16.8s | **13.6x** |
| **JAX/numpy gap** | **6.3x** | **1.6x** | — |

Note the JAX result depends on the OpenMDAO working-tree patch (change 6). On a stock OpenMDAO
install the JAX script is ~25s and the gap is ~2.4x.

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
  - *(PARTLY RESOLVED — see change 5. Re-tracing was ruled out: 126 cached compiles, one-time.
    The gap was dense Jacobians, now fixed; 3.5x -> 2.2x. What remains as the live lead is the
    op-by-op primitive dispatch seen in the compile log — the components are not executing as
    one fused kernel each.)*
- **`m_fuel` is hitting its bound ~697 times per run.** *(STILL OPEN. Two fixes tried and
  neither worked — see changes 8 and 9. Tightening the bound to 50000 was a 4.2x iteration
  regression that clamped a valid sample; retuning `ref` to 32000 left the clip count flat at
  700. The mechanism is still unidentified.)* The balance declares `upper=100000.0` kg
  while converged fuel mass is ~16,000 kg, so Newton overshoots ~6x on intermediate steps and
  gets clipped, repeatedly. Real wasted solver work and a sign the balance bounds / `ref` are
  loose. Now easy to miss, since change 7 silenced the warnings that revealed it.
  *(The ~16,000 kg figure in the line above is wrong — it is the deterministic case. The
  156-sample mean is 32,021 kg. Corrected in change 8.)*
- **JAX still executes op-by-op, not as fused kernels.** `dispatch.apply_primitive` is 184,539
  calls / 3.13s, and the compile log shows individual primitives (`jit(log)`, `jit(multiply)` on
  `float64[1]`) rather than one compiled function per component. Likely the largest remaining
  structural difference between the backends.
- **UQPCE builds jax components regardless of backend.** `jax_explicit_comp.__init__` is called
  20x from `uqpce/mdao/cdf/cdfgroup.py:33`, so even the "numpy" script pays ~2.0s of jax
  compilation (73 compiles). Not fixable from the example scripts.
- **Unverified:** the new sparse patterns were confirmed structurally (JAX nnz now matches numpy
  exactly at 3,120) and by `lambd_50` agreement to 1e-11, but `check_partials` has **not** been
  run against the JAX components. Worth doing before relying on these derivatives.
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
