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

### Round 2 — isolating the flag (in progress)

Times a seeded run that *keeps* `force_alloc_complex` (same path, flag is the only difference)
plus a repeat of the seeded runs to quantify run-to-run noise.

| Variant | Time |
|---|---|
| seeded + complex, numpy | pending |
| seeded, no complex, numpy (rep 2) | pending |
| seeded + complex, JAX | pending |
| seeded, no complex, JAX (rep 2) | pending |

---

## Correctness checks

- `lambd_50` is **bit-identical** baseline-vs-modified within each backend
  (numpy `0.020696715222417356`, JAX `0.020696715222464617`) — confirms removing
  `force_alloc_complex` changed no results.
- numpy and JAX agree to ~1e-12 on that value, so **JAX is running float64**; the 6.3× gap is
  not a precision-mode issue.
- Seeding works: the seeded runs land on nearly the same optimum across backends
  (S 140.18 vs 140.12, AR 15.54 vs 15.56, DOC mean 55745 vs 55734), where the unseeded
  baselines scattered.

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
