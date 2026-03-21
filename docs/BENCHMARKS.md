# Benchmarks for plagih GP

The framework includes several benchmark environments for testing and demonstration.

## Overview

| Benchmark | Folder | Complexity | Status | Description |
|-----------|--------|------------|--------|-------------|
| **MountainCar** | `benchmarks/mc/` | ⭐ Simple | ✅ Standard | Continuous control, 2 inputs |
| **CartPole** | `benchmarks/cp/` | ⭐ Simple | ✅ Available | Balancing, 4 inputs, discrete output |
| **Symbolic Regression** | `benchmarks/sr/` | ⭐ Simple | ✅ New | Classic GP benchmark, 1 input |
| **Industrial Benchmark** | `benchmarks/ib/` | ⭐⭐⭐ Complex | ⚠️ Experimental | Many inputs, industrial scenario |

## Parallel Performance Diagnosis

Detailed analysis of Windows parallelization (shared memory, batch sizes,
population comparison):

- `docs/PARALLEL_BENCHMARK_DIAGNOSIS.md`

Direct CPU/RAM profiling for the current parallel path:

- `plagih/test/benchmarks/bench_parallel_resources.py`
- Output: `plagih/test/benchmarks/bench_resources_output.txt`

## Tree-Creation Performance Diagnosis

Focused creation/evaluation diagnosis for cases where evolution appears slow or
stuck while producing trees:

- Runner: `plagih/test/benchmarks/bench_tree_creation.py`
- Shared harness: `plagih/test/benchmarks/_tree_creation_harness.py`
- Pytest smoke coverage: `plagih/test/test_tree_creation_benchmark.py`
- Output: `plagih/test/benchmarks/bench_tree_creation_output.json`

The harness separates four scenarios:

1. **Raw random creation** (`evolve_create_random`) — pure tree building
2. **Depth-goal creation** (`evolve_new_tree_depth`) — targeted deeper trees
3. **Initial population** (`gen_create_initial`) — creation + evaluation
4. **Active generation** (`run_generation`) — strategy mix with timing records

Each summary reports `mean/p50/p95/max`, dominant phase counts, failed-stage
counts, and the slowest example expressions.

### First benchmark findings (2026-03-20, local Windows run)

- **Raw tree creation is cheap**: `raw_random_creation` averaged **~1.46 ms**
  per tree and `depth_goal_creation` **~2.06 ms** per tree.
- **The bottleneck is usually evaluation, not construction**:
  `initial_population` had **112/120 successful trees dominated by evaluation**,
  with `max_evaluate_ms ≈ 55.9 ms`.
- **Mixed generations show the same pattern**:
  `active_generation` had **73/108 successful trees dominated by evaluation**,
  with `max_evaluate_ms ≈ 92.0 ms` and `p95_total_ms ≈ 25.1 ms`.
- **Simplification was negligible in the current active-test preset**:
  `max_simplify_ms ≈ 0.09 ms`, because the debug-safe strategy mix no longer
  includes the dedicated `simplicate` stage.
- **Evaluation failures are clustered in the evaluation phase**:
  `initial_population` produced **22 evaluation-stage errors**, and
  `active_generation` **12**, while creation/simplification errors were `0`.

Implication: if a long run appears stuck during tree production, inspect
`tree_to_candidate()` / `evaluate_tree_standalone()` first; raw tree generation
is comparatively cheap in the current architecture.

---

## MountainCar (Standard Benchmark)

**Used in:** `demo_minimal()`, standard test runs

**Problem:**
- A car must drive out of a valley onto a hill
- The car lacks the power to drive up directly
- It must build momentum (drive back and forth)

**Specification:**
```
Inputs:  cartPos (position: -1.2 to 0.6)
         cartVel (velocity: -0.07 to 0.07)
Output:  action (0=left, 1=nothing, 2=right)
```

**Files:**
```
benchmarks/mc/
├── gp_files/
│   ├── samples200.csv              # Small training set (200 samples)
│   ├── samples75.csv               # Very small set (75 samples)
│   ├── behaviour_samples.csv       # Full samples (~2000)
│   └── tree_*.csv                  # Various initial trees
└── agents/
    └── ...                         # Evaluation agents
```

**Example:**
```python
from plagih_gp import demo_minimal
demo_minimal()
```

**Known good solutions:**
- `sign(cartVel)` — very simple, works basically
- `sign(cartPos + cartVel)` — better

---

## CartPole (Alternative Benchmark)

**Used in:** `demo_cartpole()`

**Problem:**
- A cart with a pole must be balanced
- The pole must not fall over
- Classic reinforcement learning problem

**Specification:**
```
Inputs:  cartPos       (cart position)
         cartVel       (cart velocity)
         poleAngle     (pole angle, formerly observation2)
         poleVel       (angular velocity, formerly observation3)
Output:  action (0=left, 1=right) — binary!
```

**Files:**
```
benchmarks/cp/
├── gp_files/
│   ├── samples.csv                    # Training data (~5800 samples)
│   ├── operators.csv                  # Operator set
│   └── tree_labels(simple).csv        # Simple initial tree
└── agents/
    ├── cartpole_eval.py               # Gymnasium evaluation
    └── yingzwang.py                   # Literature agent
```

**Example:**
```python
from plagih_gp import demo_cartpole
demo_cartpole()
```

**Known good solutions:**
- `poleAngle < 0` — angle only (simplest solution)
- `poleVel < 0` — angular velocity only

---

## Symbolic Regression (Classic GP Benchmark)

**Used in:** `demo_symbolic_regression()`

**Problem:**
- Find a mathematical formula that approximates data
- Classic GP standard benchmark
- No external environment needed

**Specification:**
```
Inputs:  x (value between -2 and 2)
Output:  target = x³ + x² + x (target function)
```

**Files:**
```
benchmarks/sr/
└── gp_files/
    └── polynomial.csv    # f(x) = x³ + x² + x
```

**Example:**
```python
from plagih_gp import demo_symbolic_regression
demo_symbolic_regression()
```

**Target function:**
The formula to be found is: `f(x) = x³ + x² + x`

This is a classic benchmark because:
- There is a unique solution
- The solution is relatively simple
- Success can be easily measured

---

## Industrial Benchmark (Complex)

**Recommended for:** Advanced tests, scalability, publications

⚠️ **Note:** This benchmark is significantly more complex than the others.

**Problem:**
- Simulation of an industrial process
- Many input variables
- Complex, non-linear dynamics

**Specification:**
```
Inputs:  Shift_0, Shift_1, ...     (shifts)
         Gain_0, Gain_1, ...       (gains)
         Setpoint, Velocity, ...   (additional variables)
Output:  control action
```

**Files:**
```
benchmarks/ib/
├── gp_files/
│   ├── samples_prepared.csv    # Prepared data (very large!)
│   └── samples_raw.csv         # Raw data
├── ib_eval_agents.py           # Evaluation
└── IDS.py                      # Industrial Data Simulator
```

---

## Adding a Benchmark

To add a new benchmark:

1. **Create folder:**
   ```
   benchmarks/BENCHMARK_NAME/
   └── gp_files/
       └── samples.csv          # Training data
   ```

2. **Samples format:**
   ```csv
   input1:float,input2:float,target:float
   0.1,0.2,1.0
   0.3,-0.1,0.0
   ...
   ```

3. **Add demo function** (in `plagih_gp.py`):
   ```python
   def demo_BENCHMARK_NAME():
       df = pd.read_csv('benchmarks/BENCHMARK_NAME/gp_files/samples.csv')
       gp = ExplainableGP.create(
           symbols=['input1', 'input2'],
           df_train=df_train,
           ...
       )
   ```
