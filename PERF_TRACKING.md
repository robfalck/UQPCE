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

### 3. Plot suppression — harness-only so far, NOT a code change

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

---

## Correctness checks

- `lambd_50` is **bit-identical** baseline-vs-modified within each backend
  (numpy `0.020696715222417356`, JAX `0.020696715222464617`) — confirms removing
  `force_alloc_complex` changed no results.
- numpy and JAX agree to ~1e-12 on that value, so **JAX is running float64**; the 6.3× gap is
  not a precision-mode issue.

---

## Open leads on the JAX gap

- **Newton iteration count.** `CoupledDisciplines` converged at iteration 688 against
  `maxiter=700` — near the ceiling — and `iprint=2` prints every iteration. Both the iteration
  count and the console I/O hit both scripts equally, likely inflating absolute numbers and
  compressing the apparent JAX gap. Worth attacking next.
- Not yet investigated: per-call JAX dispatch/tracing overhead on small vectors, and whether
  the JAX components are re-tracing rather than reusing compiled code.

---

## Scratch files to delete before committing

Created in `examples/aircraft_design/` for the comparison, all untracked:
`baseline_quantify.py`, `baseline_quantify_JAX.py`, `isolate_quantify.py`,
`isolate_quantify_JAX.py`
