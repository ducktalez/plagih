# Targeted Evolutionary Optimization

> Design document for **tree-specific, individual optimization** beyond
> random evolution. Covers pseudo-backpropagation, node-level analysis,
> population-wide orchestration, and hybrid GP↔NN approaches.
>
> Status: **Phase 1 + 2 implemented** — Phase 3+ pending.
> Tracked in `IMPLEMENTATION_PLAN.md` § D5.

---

## 1. Motivation

In classical GP, individual candidates are **not optimized in isolation** —
improvement comes from the stochastic interplay of selection, mutation, and
crossover across a population. This works well for smooth search landscapes,
but fails systematically for structures that require **coordinated
multi-component optimization** (e.g., IF/Piecewise, where condition and both
branches must fit together).

This document collects ideas for **targeted, per-tree optimization** that
complements — not replaces — the evolutionary loop.

---

## 2. Fundamental Concepts

### 2.1 Three pillars of targeted optimization

| Pillar | Goal | Key question |
|---|---|---|
| **Node-level potential analysis** | Identify child nodes with the greatest improvement headroom | "Which subtree hurts fitness the most?" |
| **Targeted recombination** | Combine formula fragments that fit together, not randomly | "Which donor subtree would improve *this* slot?" |
| **Structural dead-end detection** | Recognize when a position *cannot* produce the needed value range | "Does this slot need negative values, but only has `Square`?" |

### 2.2 Per-node backpropagation signals

For each internal node, the following **analysis signals** can be computed
by evaluating the tree on the training data and inspecting intermediate
values:

| Signal | Description | Use |
|---|---|---|
| **Activation function** | What mathematical transformation does this node apply? | Gradient / sensitivity analysis |
| **Input variables** | Which `Symbol`s occur in this subtree? | Feature-relevance, variable importance |
| **Value range (min/max)** | Observed output range on training data | Dead-end detection, clipping hints |
| **Differentiability** | Is the subtree differentiable w.r.t. its inputs? | Gradient-based local refinement |
| **Cyclicity / continuity** | Is the output periodic (trig) or discontinuous (Ifte)? | Structural compatibility checks |

### 2.3 Best-per-datapoint analysis

For each row in the training data, there exists a candidate in the
population that predicts it best. Analyzing **which candidates cover which
rows** reveals:

- Which formula fragments are responsible for which data regions
- Where the population has **blind spots** (no candidate is good)
- Which candidates are **redundant** (always dominated by another)

> **Related literature:** This is closely related to the concept of an
> *Oracle Selector* (also called *Optimal Oracle* or *Best-of-Population
> Bound*) in ensemble learning. The oracle always picks the best ensemble
> member per sample — its error is the theoretical lower bound that any
> selector/combiner over the ensemble can achieve.
>
> - Kuncheva, *Combining Pattern Classifiers* (2014), §5.3 — Oracle
>   and single-best selector as ensemble baselines.
> - Caruana et al., *Ensemble Selection from Libraries of Models* (2004)
>   — greedy set-cover for ensemble pruning (analogous to Minimum Set).
> - In GP specifically: Vanneschi et al., *Measuring Bloat, Overfitting
>   and Search Difficulty in GP* (2010) — per-sample error decomposition
>   across populations.

### 2.4 SymPy unification as enabler

SymPy simplification and chaining create **commutative operator chains**
(e.g., `Add(a, b, c, …)`, `Mul(…)`) whose individual operands can be
targeted independently:

- **Add a summand** to an `AddChain` without touching the rest
- **Remove a factor** from a `MulChain`
- **Replace one branch** in a `Piecewise` while keeping other cases

This makes chained operators the **natural interface** for targeted
mutation and recombination.

---

## 3. Concrete Plans

> Theoretically implementable. Each needs design decisions before coding.

### 3.1 Pseudo-Backpropagation for Ifte / Piecewise

**Problem:** IF-structures are systematically disadvantaged in GP:

| Issue | Why it hurts |
|---|---|
| Complexity multiplier | Even with generous weighting, an Ifte always spreads the tree significantly (spreading factor) |
| Random optimization is ineffective | Condition, then-branch, and else-branch must cooperate — random crossover almost never improves all three simultaneously |
| Requires explicit encouragement | Without operator-weight bonuses or complexity discounts, polynomials and trig combinations usually win |

**Core insight (from Masterarbeit → Familiarity):** Humans solve problems
by *encapsulating* sub-problems with IF/case-splits. GP lacks this
capability because it cannot reason about "which branch *should* fire
when." The fundamental difference is that humans can decompose a problem
and solve sub-problems independently — GP treats the entire tree as one
opaque unit.

**Proposed algorithm:**

1. Evaluate the full tree on all training rows.
2. For each `Ifte(cond, then, else)` node, also evaluate `then` and `else`
   **unconditionally** on all rows.
3. Compute a **performance score** for each component:

   | Component | Metric Option A (count) | Metric Option B (error sum) |
   |---|---|---|
   | **Condition** | How often does `cond` select the branch that is closer to the target? | When `cond` selects the worse branch: sum of \|better − worse\| |
   | **Then-branch** | On rows where `cond=True`: how often is `then` closer to target than `else`? | Sum of \|then − target\| on those rows |
   | **Else-branch** | Analogous for `cond=False` rows | Sum of \|else − target\| on those rows |

4. The component with the **worst score** is the mutation priority target.
5. Apply **focused mutation** only to that component (keep the rest fixed).

> **Open decision:** Option A (count) is simpler and more robust to
> outliers. Option B (error sum) captures magnitude but is sensitive to
> scale. A weighted hybrid is possible.

**Extends to Piecewise:** Same logic, applied to each
`ExprCondPair_Dummy` independently. Each condition and each expression
gets a performance score.

### 3.2 Node-level optimal-value comparison

For any operator node, compute the **optimal output** that would minimize
the tree's overall error, then compare it to the actual output:

1. Fix the entire tree except the subtree at position `k`.
2. For each training row, compute what value at position `k` would make the
   tree output equal to the target → this is the **ideal child value**.
3. Compare ideal vs. actual → the child with the largest deviation
   (by sum or count) is the weakest link.

**Example:** For `Add(a, b, c)`, the ideal value of `c` on row `i` is
`target_i − a_i − b_i`. The sum of `(c_i − ideal_c_i)²` is the
**optimization gap** for child `c`.

> This generalizes the Ifte pseudo-backpropagation to *all* invertible
> operators. Non-invertible operators (e.g., `Abs`, `Sign`) require
> interval analysis instead of exact inversion.

### 3.3 SoftOptimum operator (population-level)

A virtual **super-operator** that, for each training row, returns the
prediction of whichever candidate in the population is closest to the
target.

| Concept | Definition |
|---|---|
| **SoftOptimum value** | `SO(row_i) = prediction of argmin_c \|pred_c(row_i) − target_i\|` |
| **SoftOptimum error** | `Σ_i \|SO(row_i) − target_i\|` — the best error achievable by *any* per-row combination of the current population |
| **Minimum Set** | Smallest subset of candidates (or total node count) needed to achieve the SoftOptimum error |

**Uses:**
- **Convergence bound:** If SoftOptimum error is already low, the population
  *contains* the pieces — the problem is *combination*, not *discovery*.
- **Targeted crossover:** Candidates that contribute many rows to the
  Minimum Set are high-value donors.
- **Stagnation diagnosis:** If SoftOptimum error stops improving across
  generations, the population lacks diversity in the right dimensions.

> **Related literature:** The SoftOptimum is equivalent to the *Oracle
> Selector Bound* in ensemble methods — the performance ceiling of any
> point-wise selector over a fixed set of predictors.
>
> - Kuncheva (2014), §5.3 — Oracle bound as baseline.
> - Caruana et al. (2004), *Ensemble Selection from Libraries of Models*
>   — greedy forward selection to approximate the oracle, directly
>   analogous to the Minimum Set computation proposed here.
> - The Minimum Set problem is a weighted **set cover** instance and thus
>   NP-hard in general, but the greedy algorithm achieves a
>   `ln(n)`-approximation (Chvátal, 1979).

### 3.4 Chained-operator targeted mutation

Chained operators (`AddChain`, `MulChain`, `MinChain`, etc.) allow
fine-grained manipulation:

- **Add/remove operands** (e.g., add a summand to a sum)
- **Replace one operand** via targeted crossover (take a summand from
  another candidate's chain)
- **Adjust factors** in a `MulChain` (local constant optimization)

This could be implemented as a new `Strategy` in `parallel.py`.

### 3.5 Merged-tree population analysis

The existing `population_merge.py` DAG can be extended for targeted
optimization:

- **"Thickest trunks"**: Subtrees shared by many candidates → high
  confidence fragments, good base structures.
- **Base structure extraction**: If a common skeleton is found across Pareto
  candidates, spawn a new sub-population with that skeleton as `origin_tree`
  and optimize only the variable parts.
- **Dead branch pruning**: In the merged DAG, branches that lead only to
  low-fitness candidates can be deprioritized or pruned.
- **Bottom-up feeding**: A complete merged tree that grows from the bottom
  by adding nodes that appear most useful across the gene pool, like
  constructing a consensus tree.

---

## 4. Distant / Tangential Ideas

> These are **not part of the core targeted-optimization concept** but are
> related research directions that emerged during brainstorming. Recorded
> here to avoid losing them.

### 4.1 Iterative GP↔NN approximation (EM-style)

An Expectation-Maximization–inspired loop:

1. **GP phase:** Evolve Pareto-efficient symbolic candidates until
   saturation.
2. **NN phase:** Use the GP candidates' outputs (or a subset of the gene
   pool) as **input features** for a neural network. The NN trains to
   optimize the residual that GP cannot capture.
3. **Feedback:** The NN's learned structure suggests which GP fragments are
   most useful. Iterate.

**Sub-ideas:**
- Choose NN architecture based on GP structure: If Pareto candidates are
  polynomials → Sigmoid activations may be better than ReLU. If candidates
  improve with complexity → use a wider/deeper network.
- Training speed of the NN as a meta-signal: If the NN converges quickly
  on GP inputs, those inputs capture meaningful structure.
- After NN training, attempt to **re-symbolize** the NN's learned function
  back into GP trees (knowledge distillation → symbolic regression).
- The NN finds the big differences between a GP-tree and the target
  solution, and the GP then tries to find Ifte-structures that mimic the
  NN behaviour.

### 4.2 NN-as-subtree replacement

Replace a specific subtree (e.g., an Ifte node) with a small neural
network (2 inputs → softmax → result layer). Train the NN on the
training data. Then force the NN to extremes (binary decisions) while
maintaining entropic efficiency → this may discover sharp logical
distinctions that can be expressed symbolically.

**Variant — "backpropagable tree":** Construct trees using
`tanh`/`sigmoid` + `ExprCondPair` structures that can mime NN layers for
IF-conditions. This makes the tree itself gradient-trainable.

### 4.3 Population mining heuristics

Automatic detection and countermeasures for population pathologies:

| Symptom | Possible reaction |
|---|---|
| Trees too large | Increase parsimony pressure, reduce `nodes_max` |
| Trees too small | Add deeper random trees, reduce tournament size |
| Too similar (low diversity) | Inject more `random_new`, use novelty bonus |
| Too bad (fitness plateau) | Increase tournament size, try different operator set |
| Same structures repeatedly found | Ban dominant sub-structures, force alternative operators |

### 4.4 Separate sub-population training

- **Core extraction:** If top candidates share a structural core, start a
  new sub-population with that core as `origin_tree`.
- **Anti-core population:** Simultaneously start a population that
  explicitly *excludes* the dominant core → forces exploration of
  alternative structures.
- **Partnering / complementary search:** Candidates search for partners
  whose per-row errors are complementary (one is good where the other is
  bad). Lay the per-datapoint prediction results over each other to find
  which trees complement each other — like a partner search among
  individuals. The combination can be the basis for a Piecewise analysis
  of the data. A metric similar to entropy could quantify how well two
  trees' coverage patterns fit together.

### 4.5 Multi-signal fitness composition

A hypothesis: combining **similarity** (to other candidates or to a
reference), **fitness**, **entropy** (of per-row errors), and
**complexity** into a multi-dimensional signal could enable much more
targeted optimization than any single metric. This connects to
lexicase selection and multi-objective optimization literature.

### 4.6 GPU-accelerated evaluation

Evaluate trees on GPU (e.g., via TensorFlow or CuPy). The existing merged
DAG from `population_merge.py` could serve as the computation graph for a
feed-forward batch evaluation. Open question: Is TensorFlow necessary, or
can NumPy + CuPy suffice?

---

## 5. Rough Implementation Roadmap

> Minimal consensus plan. Each phase is independently useful.

### Phase 1 — Analysis infrastructure (no evolution changes)

```
                 ┌─────────────────────────────┐
                 │  eval_predict_numpy_now()    │
                 │  + intermediate_values=True  │
                 └──────────┬──────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Per-node value vectors     │
              │  (dict: Node → np.ndarray)  │
              └─────────────┬──────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
  │ Best-per-    │  │ SoftOptimum  │  │ Value range /    │
  │ datapoint    │  │ error bound  │  │ dead-end detect  │
  │ mapping      │  │ (monitoring) │  │ per node         │
  └──────────────┘  └──────────────┘  └──────────────────┘
```

- [x] **Per-node intermediate-value evaluation**: Extend `eval_predict_numpy_now`
      to optionally return intermediate results at each node, not just the root.
      → `eval_node_intermediates()` in `plagih/targeted_optimization.py`
- [x] **Best-per-datapoint mapping**: For a population, compute which candidate
      wins each training row. Return a `(n_rows,)` array of candidate indices.
      → `best_per_datapoint()` in `plagih/targeted_optimization.py`
- [x] **SoftOptimum error**: Compute the oracle-selector bound for the current
      population as a monitoring metric (add to `GPMonitor`).
      → `soft_optimum_error()` in `plagih/targeted_optimization.py`

### Phase 2 — Ifte/Piecewise pseudo-backpropagation

```
  Ifte(cond, then, else)
    │
    ├── eval cond on ALL rows  ──► bool mask
    ├── eval then on ALL rows  ──► then_vals
    ├── eval else on ALL rows  ──► else_vals
    │
    ▼
  ┌──────────────────────────────────┐
  │  score_condition(mask, target,   │
  │                  then_vals,      │
  │                  else_vals)      │
  │  score_then(mask, target, ...)   │
  │  score_else(mask, target, ...)   │
  └──────────┬───────────────────────┘
             │
             ▼
  weakest component ──► focused mutation
```

- [x] **Component scoring** (§3.1): Implement Option A (count-based) first.
      → `ifte_component_scores()` + `piecewise_component_scores()` in
      `plagih/targeted_optimization.py`
- [x] **Focused mutation strategy**: New `Strategy("targeted_ifte", rate=…)` in
      `parallel.py` that only mutates the weakest Ifte component. Falls back to
      standard branch mutation if no Ifte nodes are found.
- [x] **Integration**: Wired into `run_generation()` as optional strategy.
      Runtime context (`_df_train`, `_target`) is injected automatically by
      `run_task_sequential()` when the strategy name is `targeted_ifte`.

### Phase 3 — General node-level optimization

- [ ] **Optimal-value computation** (§3.2) for invertible operators
      (`Add`, `Mul`, `Sub`, `Div`).
- [ ] **Optimization-gap metric**: Report per-node gap as analysis output.
- [ ] **Gap-guided mutation**: Preferentially mutate the child with the
      largest gap.

### Phase 4 — Population-level orchestration

- [ ] **Minimum Set computation** (§3.3): Greedy set-cover approximation.
- [ ] **Chained-operator targeted mutation** (§3.4) as new `Strategy`.
- [ ] **Merged-tree trunk analysis** (§3.5): Identify shared subtrees,
      suggest `origin_tree` candidates.

### Phase 5 — Hybrid GP↔NN (research / exploratory)

- [ ] Prototype EM loop (§4.1) with a simple MLP on GP outputs.
- [ ] NN-as-subtree experiment (§4.2) for a single Ifte node.

