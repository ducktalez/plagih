# Parallel Benchmark Diagnosis

> **⚠️ Keep this file intact.** The measurement data in this document is the
> baseline for any future rework of the parallelization system (H1–H3 are
> closed, see "Decided" in `IMPLEMENTATION_PLAN.md`). Do not condense or
> remove data tables — they are needed for before/after comparison.

> Final evaluation of the current Windows parallelization in `plagih`.
>
> Sources:
> - `plagih/test/benchmarks/bench_output.txt`
> - `plagih/test/benchmarks/bench_resources_output.txt`
>
> Date: 2026-03-16

---

## Setup

- OS: Windows
- CPU: 8 physical cores / 16 threads
- Main benchmark: `plagih/test/benchmarks/bench_diagnose_full.py`
- Resource profiler: `plagih/test/benchmarks/bench_parallel_resources.py`
- Main configuration:
  - `pop=1000`
  - `gens=5`
  - `compare_pops=(1000, 10000)`
  - `batch_sizes=(1, 32, 128, 0)`
- Resource snapshot:
  - `pops=(1000, 10000)`
  - `gens=1`
  - `workers=(0, 8)`

---

## Executive Summary

### Key findings

1. **The current parallelization works well and scales in practice.**
   - Best steady-state configuration is **`parallel(8w)`**.
   - Steady-state speedup:
     - `pop=1000`: **2.98×**
     - `pop=10000`: **3.49×**

2. **The biggest historical IPC bottlenecks have been largely resolved.**
   - Shared memory for `df_train`: **1.68×** faster worker startup
   - Pre-selection instead of legacy update: **4.8×** less IPC cost

3. **The current batch sweet spot is roughly `32..128` tasks per batch.**
   - `pop=1000`: **32** is clearly best
   - `pop=10000`: **128** slightly ahead of `32`
   - Large auto-batches (`~1 batch/worker`) are not optimal

Wunderbar fahre nun mit der Implementierung im ZigZag-Plan fort. 4. **At benchmark time, `gen_create_initial()` was a large fixed sequential cost block.**
   - `pop=10000` initialization costs ~**44–47 s** per configuration
   - This block is nearly identical for sequential and parallel

5. **RAM is a real parallelization factor, but not an immediate killer yet.**
   - Peak RSS at `parallel(8w)`:
     - `pop=1000`: **1.29 GB**
     - `pop=10000`: **1.50 GB**
   - Notable: Most of the additional parallel RAM resides in worker processes (**~1.13–1.18 GB child RSS**)
   - The step from `pop=1000` to `pop=10000` increases peak parallel RAM less than expected, because the largest RAM block is apparently fixed worker/interpreter/import/pool overhead

---

## 1. Factor overview for the current parallelization

| Factor | Metric | Impact | Status |
|---|---|---|---|
| Worker startup / spawn | Pool init time | Medium | Improved via shared memory |
| `df_train` transport | Init payload / pool init | Small to medium | Improved |
| Population IPC | `_update_worker_state` vs. pre-selection | High | Greatly improved |
| Pre-selection in main process | `pre_select_for_tasks(...)` | Medium | Current main block in IPC path |
| Task granularity | Avg task time ~3.9 ms | High | Still limiting |
| Batch size | 1 / 32 / 128 / auto | High | Sweet spot found |
| Result return transport | Result pickle dumps | Low | Not a bottleneck |
| Worker count | 2 / 4 / 8 | High | 8 currently best |
| Initial population | `gen_create_initial()` | High | Sequential at measurement time |
| RAM (main process) | RSS | Medium | Scales with population |
| RAM (worker processes) | Child RSS | High | Fixed parallel overhead |
| CPU utilization | System CPU avg/peak | Medium | Clearly elevated in parallel, but not fully saturated |
| SymPy pathologies | Recursion / hangs | High risk | Mitigated by guards |
| Debug logging | Eager f-string / `str_as_expr()` | High risk | Fixed |

---

## 2. Transport and IPC factors

### 2.1 Pickle sizes (`pop=1000`)

| Object | Size |
|---|---:|
| `evolve` | 1.0 KB |
| `df_train` | 58.1 KB |
| `pop_genepool` | 388.3 KB |
| `paretofront` | 1.6 KB |
| Total | 449.6 KB |

**Interpretation:**
- `df_train` is small.
- `pop_genepool` dominates the data volume.
- Shared memory for `df_train` is useful, but **not** the biggest lever.

### 2.2 `df_train`: Pickle vs. shared memory

| Metric | Value |
|---|---:|
| Pickle payload | 58.1 KB |
| Shared memory raw buffer | 57.4 KB |
| Shared memory metadata | 75 B |
| Pickled DataFrame init | 3558.7 ms |
| Shared memory attach | 2123.1 ms |
| Startup speedup | 1.68× |

**Interpretation:**
- Shared memory measurably saves time during pool startup.
- The effect is real, but secondary compared to total generation costs.

### 2.3 Legacy IPC vs. pre-selection

| Metric | Legacy | Pre-selection | Factor |
|---|---:|---:|---:|
| IPC time (`4w`) | 1072.7 ms | 221.7 ms | 4.8× |

Breakdown of the new variant:

| Part | Time |
|---|---:|
| `pre_select_for_tasks(...)` | 197.6 ms |
| Batch dumps | 24.2 ms |
| Total | 221.7 ms |

**Interpretation:**
- The old population IPC was a genuine killer.
- Today the remaining large block is no longer pickle, but **pre-selection in the main process**.

---

## 3. Task granularity and batching

### 3.1 Per-task compute (`pop=1000`)

| Metric | Value |
|---|---:|
| Tasks | 900 |
| Successful candidates | 946 |
| Total | 3531.2 ms |
| Avg per task | 3.9 ms |

**Interpretation:**
- 3.9 ms per task remains short.
- This keeps scheduling, queue, and submission overhead relevant.
- Parallelization therefore needs **chunking**, not mini-tasks.

### 3.2 Batch comparison `pop=1000`

| Batch | #Batches | Payload | Avg time | Speedup vs `1` |
|---|---:|---:|---:|---:|
| `1` | 900 | 567.1 KB | 3328.2 ms | 1.00× |
| `32` | 29 | 320.1 KB | 1338.1 ms | 2.49× |
| `128` | 8 | 319.6 KB | 1500.1 ms | 2.22× |
| `auto(225)` | 4 | 310.4 KB | 1510.3 ms | 2.20× |

### 3.3 Batch comparison `pop=10000`

| Batch | #Batches | Payload | Avg time | Speedup vs `1` |
|---|---:|---:|---:|---:|
| `1` | 9000 | 5.5 MB | 14288.5 ms | 1.00× |
| `32` | 282 | 3.1 MB | 11903.3 ms | 1.20× |
| `128` | 71 | 3.0 MB | 11625.6 ms | 1.23× |
| `auto(2250)` | 4 | 3.0 MB | 12676.9 ms | 1.13× |

### 3.4 Cross-table: sweet spot by population

| Population | Best batch | Runner-up | Auto-batch relative to best |
|---|---|---|---|
| `1000` | `32` | `128` | 1.13× slower |
| `10000` | `128` | `32` | 1.09× slower |

**Interpretation:**
- The heuristic "roughly one batch per worker" is currently **too coarse**.
- The stable working range is **32 to 128 tasks per batch**.
- This aligns with the current runtime change in `parallel.py`, using multiple smaller chunks instead of one large batch.

---

## 4. Result return transport

| Batch | Payload | Dumps |
|---|---:|---:|
| `1` | 1.2 KB | 0.1 ms |
| `32` | 15.3 KB | 1.1 ms |
| `128` | 23.2 KB | 1.6 ms |
| `auto(63)` | 22.9 KB | 1.5 ms |

**Interpretation:**
- The return transport of `TaskResult`s is small.
- This is **not** a priority lever.

---

## 5. End-to-end scaling

## 5.1 `pop=1000`, steady-state (gen 2+)

| Config | Avg/Gen | Speedup | Efficiency |
|---|---:|---:|---:|
| sequential | 3969.4 ms | 1.00× | - |
| parallel(2w) | 2788.3 ms | 1.42× | 71% |
| parallel(4w) | 1817.9 ms | 2.18× | 55% |
| parallel(8w) | 1333.0 ms | 2.98× | 37% |

## 5.2 `pop=10000`, steady-state (gen 2+)

| Config | Avg/Gen | Speedup | Efficiency |
|---|---:|---:|---:|
| sequential | 41612.2 ms | 1.00× | - |
| parallel(2w) | 28550.6 ms | 1.46× | 73% |
| parallel(4w) | 17097.8 ms | 2.43× | 61% |
| parallel(8w) | 11921.6 ms | 3.49× | 44% |

## 5.3 Cross-table: worker scaling vs. population

| Population | 2 Worker | 4 Worker | 8 Worker | Best |
|---|---:|---:|---:|---|
| `1000` | 1.42× | 2.18× | **2.98×** | `8w` |
| `10000` | 1.46× | 2.43× | **3.49×** | `8w` |

## 5.4 Cross-table: efficiency vs. population

| Population | 2 Worker | 4 Worker | 8 Worker |
|---|---:|---:|---:|
| `1000` | 71% | 55% | 37% |
| `10000` | 73% | 61% | 44% |

**Interpretation:**
- The current parallelization scales **better** with larger populations.
- More work per generation amortizes parallel overheads better.
- `8w` is the best configuration on this 8-core system with the current architecture.
- Efficiency remains sub-linear, but is clearly usable for `pop=10000`.

---

## 6. Initial population as a separate cost block

At the time of these measurements, `gen_create_initial()` still ran sequentially.

| Population | Sequential init | Parallel(8w) init | Finding |
|---|---:|---:|---|
| `1000` | 4288.7 ms | 4493.1 ms | Practically equal |
| `10000` | 44366.6 ms | 45096.9 ms | Practically equal |

**Interpretation:**
- In this measurement baseline, the init block **did not** benefit from `parallel=`.
- For `pop=10000`, initialization alone costs ~45 s per configuration.
- This explains a significant portion of the total runtime.

---

## 7. CPU and RAM profiling

Direct resource values are from `bench_parallel_resources.py`.

**Important:** Resource runs were recorded with `gens=1`. They are ideal for
CPU/RAM comparisons and relative worker costs, but **not** for steady-state
rankings. For performance rankings, see section 5 with the `gens=5`
measurements from `bench_output.txt`.

### 7.1 Resource cross-table (full: `0/2/4/8` workers)

| Pop | Config | Avg/Gen | Peak RSS | Child RSS | Peak CPU |
|---|---|---:|---:|---:|---:|
| `1000` | sequential | 3521.6 ms | 161.4 MB | 0 B | 100.8% |
| `1000` | parallel(2w) | 4397.8 ms | 457.7 MB | 293.4 MB | 100.2% |
| `1000` | parallel(4w) | 3397.8 ms | 750.0 MB | 581.1 MB | 106.2% |
| `1000` | parallel(8w) | 3659.2 ms | 1.29 GB | 1.13 GB | 104.2% |
| `10000` | sequential | 36846.3 ms | 290.2 MB | 0 B | 106.1% |
| `10000` | parallel(2w) | 25739.3 ms | 632.9 MB | 319.4 MB | 106.2% |
| `10000` | parallel(4w) | 16716.7 ms | 940.9 MB | 622.5 MB | 106.2% |
| `10000` | parallel(8w) | 12007.6 ms | 1.49 GB | 1.18 GB | 106.2% |

### 7.2 Worker count × peak RSS / child RSS

| Worker | `pop=1000` Peak RSS | `pop=1000` Child RSS | `pop=10000` Peak RSS | `pop=10000` Child RSS |
|---|---:|---:|---:|---:|
| `0` | 161.4 MB | 0 B | 290.2 MB | 0 B |
| `2` | 457.7 MB | 293.4 MB | 632.9 MB | 319.4 MB |
| `4` | 750.0 MB | 581.1 MB | 940.9 MB | 622.5 MB |
| `8` | 1.29 GB | 1.13 GB | 1.49 GB | 1.18 GB |

### 7.3 Worker count × `gen_1` system CPU utilization

| Worker | `pop=1000` sys CPU avg | `pop=10000` sys CPU avg |
|---|---:|---:|
| `0` | 29.2% | 26.9% |
| `2` | 43.2% | 31.7% |
| `4` | 37.6% | 55.7% |
| `8` | 58.7% | 57.5% |

### 7.4 Population × RAM per candidate

This is a rough `Peak RSS / pop_size` view. Not "pure" memory per candidate,
but a useful density indicator.

| Pop | Config | Peak RSS per candidate |
|---|---|---:|
| `1000` | sequential | ~161 KB |
| `1000` | parallel(2w) | ~458 KB |
| `1000` | parallel(4w) | ~750 KB |
| `1000` | parallel(8w) | ~1.29 MB |
| `10000` | sequential | ~29 KB |
| `10000` | parallel(2w) | ~63 KB |
| `10000` | parallel(4w) | ~94 KB |
| `10000` | parallel(8w) | ~149 KB |

### 7.5 Speedup per additional GB RAM (`gens=1`, relative to sequential)

Formula:

```text
(speedup_vs_sequential - 1.0) / extra_peak_rss_in_GiB
```

| Pop | Config | Speedup vs seq | Additional peak RAM | Speedup gain per GiB |
|---|---|---:|---:|---:|
| `1000` | parallel(2w) | 0.80× | ~0.29 GiB | ~-0.69×/GiB |
| `1000` | parallel(4w) | 1.04× | ~0.57 GiB | ~0.06×/GiB |
| `1000` | parallel(8w) | 0.96× | ~1.10 GiB | ~-0.03×/GiB |
| `10000` | parallel(2w) | 1.43× | ~0.33 GiB | ~1.29×/GiB |
| `10000` | parallel(4w) | 2.20× | ~0.64 GiB | ~1.90×/GiB |
| `10000` | parallel(8w) | 3.07× | ~1.17 GiB | ~1.71×/GiB |

### 7.6 CPU interpretation

- Sequential essentially uses **one core fully**.
- Parallel clearly increases system CPU load, but not perfectly monotonically:
  - At `pop=1000`, `4w → 8w` jumps significantly in CPU load without becoming clearly faster in the `gens=1` resource run
  - At `pop=10000`, CPU load increases sensibly with worker count and correlates much better with speedup
- This suggests that small populations in the first generation run are still relatively dominated by setup/orchestration overhead.

### 7.7 RAM interpretation

- The main process scales visibly with population:
  - sequential `1000`: 161.4 MB
  - sequential `10000`: 290.2 MB
- The large parallel RAM block resides in the **worker processes**.
- Notable: the nearly linear child RSS increase with worker count:
  - `pop=1000`: 293 MB → 581 MB → 1.13 GB
  - `pop=10000`: 319 MB → 623 MB → 1.18 GB

**Practical implication:**
- A large portion of parallel RAM is fixed worker/interpreter/import/pool state.
- The population increase `1000 → 10000` only moderately increases worker RAM.
- Worker count overhead is thus a more important resource lever than population size alone for the current architecture.

### 7.8 Robustness observation

In the resource run `pop=10000`, `parallel(4w)`, a `SympyError` from the
known `Ifte`/`Piecewise` pathology occurred (`random_new`, `task_index=6584`).

Important:
- The error was **cleanly caught**,
- printed with debug context,
- and the run was **not** blocked or turned into a hang.

This confirms that the newly added error diagnostics in `parallel.py` and
`trees.py` serve their purpose.

---

## 8. Impact factors, prioritized

### Greatest positive impact
1. **Pre-selection instead of legacy population IPC**
2. **Shared memory for `df_train`**
3. **Chunked batching instead of mini-tasks or giant chunks**
4. **8 workers on 8 physical cores**

### Biggest remaining bottlenecks
1. **In this baseline, `gen_create_initial()` remained sequential**
2. **Pre-selection runs in the main process and costs ~200 ms per generation (`pop=1000`)**
3. **Task times remain short (~3.9 ms)**
4. **Worker RAM overhead is high (~0.29 GB at `2w`, ~0.58–0.62 GB at `4w`, ~1.13–1.18 GB at `8w`)**
5. **System CPU is not fully utilized**

---

## 9. Practical conclusions for the current architecture

### What the final numbers clearly show

- The current parallelization is **clearly successful**.
- The previous state "parallelization barely pays off" **no longer holds**.
- On this system, **`parallel(8w)`** is the best configuration for the current architecture.
- Larger populations improve parallel efficiency.
- The batch zone **32..128** is the currently most sensible granularity.

### What CPU/RAM reveal about the next lever

- **CPU-side**: Utilization is elevated but not ideal → scheduling/orchestration losses remain.
- **RAM-side**: The biggest jump is not `1000 → 10000`, but `sequential → parallel(8w)`.
- Therefore, the most important resource lever is currently:
  - Reduce worker state
  - Or consciously budget worker count/batching against RAM

### Concrete recommendations

1. **Do not revert default batching to "one batch per worker".**
2. **Consider `parallel(8w)` as the preferred benchmark configuration on this host.**
3. **For RAM-sensitive machines, document a second preset path**, e.g. `parallel(4w)`.
4. **If further optimization is pursued, start here:**
   - Re-measure the now unified generation-0 runner and amortize remaining init overhead
   - Make pre-selection more efficient
   - Reduce worker state/RAM

---

## 10. Open tasks

> **Note:** H1–H3 have since been resolved/closed — see the "Decided"
> section in `docs/IMPLEMENTATION_PLAN.md`.

- [ ] **Further optimize parallelization:** Specifically re-examine worker RAM, pre-selection, and the post-H1 generation-0 runner. Current measurements suggest untapped potential.
- [ ] **Repeat resource profiling with `gens>=2`** to get CPU/RAM cross-tables not just for the first generation run but also for steady state.
- [ ] **Test batch sweet spot under RAM budget**, e.g. `32`, `64`, `128` against `2w/4w/8w`, to derive a better perf/RAM preset.

---

## 11. Relevant files

- Full benchmark run: `plagih/test/benchmarks/bench_diagnose_full.py`
- Final output: `plagih/test/benchmarks/bench_output.txt`
- Resource profiler: `plagih/test/benchmarks/bench_parallel_resources.py`
- Resource output: `plagih/test/benchmarks/bench_resources_output.txt`
- Runtime parallel logic: `plagih/parallel.py`
- Pool/GP lifecycle: `plagih/trees.py`
