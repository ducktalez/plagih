# NN+GP Co-Evolution — Paper Blueprint

> **Auto-generated from experiment:** `{{experiment_id}}`
> **Benchmark:** `{{benchmark}}`
> **Date:** `{{date}}`
> **Results directory:** `{{results_dir}}`

---

## Abstract

We present a framework that combines Genetic Programming (GP) and Neural Networks (NN)
in an iterative Expectation-Maximisation loop.  In each iteration, GP evolves a
Pareto-optimal set of symbolic expressions whose outputs serve as additional input features
for a minimal neural network.  The NN then trains to predict the same target as the GP,
exploiting the symbolic features to reduce its required capacity.  The gap between GP
prediction and ground truth (the *residual*) becomes the GP target for the next iteration,
progressively shifting the GP towards modeling structure that the NN cannot yet capture.

On the **{{benchmark}}** benchmark (`{{n_train_rows}}` training rows, `{{n_features}}` raw
features), a baseline MLP requires **{{baseline_nn_params}} parameters** (architecture:
`{{baseline_nn_arch}}`) to achieve MSE `{{baseline_nn_mse:.5f}}`.  After **{{n_iterations}}
EM iterations**, the GP-enriched NN achieves MSE `{{final_nn_mse:.5f}}` with
**{{final_nn_params}} parameters** (architecture: `{{final_nn_arch}}`); the NN parameter
count **{{param_reduction_direction}} by {{param_reduction_pct:.1f}}%**.

---

## 1. Introduction

*[Stub — expand with motivation, related work pointers, contributions]*

Symbolic regression via GP produces interpretable models but struggles with complex,
piecewise behaviour.  Neural networks excel at capturing such complexity but are opaque.
Our approach bridges both worlds:

- Pareto-optimal GP candidates provide **interpretable building blocks** (features).
- A minimal NN uses these features to **reduce its own parameter budget**.
- Residual analysis drives **iterative refinement** of the symbolic component.

Key contributions:
1. A general EM-loop pipeline combining GP and NN with a shared normalised MSE loss.
2. Evidence that GP-derived symbolic features reduce NN size on {{benchmark}}.
3. A paper-blueprint pipeline that auto-generates this document from experiment artefacts.

---

## 2. Method

### 2.1 Shared Error Metric

Both GP and NN optimise the same normalised MSE:

```
MSE(y_pred, y_true) = mean((y_pred - y_true)^2)
```

All targets and predictions are normalised to `[0, 1]` via MinMax scaling.

### 2.2 EM Loop

```
Iteration 0:
  target_0  = normalised ground truth
  GP_0      = evolve Pareto set against target_0   ({{gp_pop_size}} pop, {{gp_gen_end}} gen)
  F_0       = [raw_features | GP_0(x) for x in training_data]
  NN_0      = find_minimal_nn(F_0, target_0, baseline_mse={{baseline_nn_mse:.5f}})
  residual_0 = normalise(target_0 - best_GP_0_prediction_per_row)

Iteration k:
  target_k  = residual_{k-1}
  GP_k      = evolve Pareto set against target_k
  F_k       = [raw_features | GP_k(x)]
  NN_k      = find_minimal_nn(F_k, target_0, baseline_mse={{baseline_nn_mse:.5f}})
  residual_k = normalise(target_0 - best_GP_k_prediction_per_row)
```

### 2.3 Minimal NN Search

Architectures are tested in order from smallest to largest:
`[8], [16], [32], [8,8], [16,8], [32,16], [64], [32,32], [64,32], [64,64], [128,64], [128,128]`.
The first architecture achieving `MSE ≤ baseline × 1.05` (5% tolerance) is selected.

---

## 3. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Benchmark | {{benchmark}} |
| Training rows | {{n_train_rows}} |
| Raw features | {{n_features}} |
| GP population | {{gp_pop_size}} |
| GP generations | {{gp_gen_end}} |
| GP error metric | MSE (normalised) |
| NN search tolerance | 5% above baseline |
| NN optimiser | Adam (lr=1e-3) |
| NN max epochs | {{nn_epochs}} |
| EM iterations | {{n_iterations}} |

> **Note — categorical targets.** On MountainCar the target (`action`) is a
> 3-class discrete signal (0/1/2). The current pipeline normalises the labels
> to `[0, 1]` floats and uses **MSE** as the loss for both GP and NN. This
> keeps the pipeline uniform across regression and classification benchmarks
> but loses the categorical structure (e.g. mistaking class 0 for 2 is
> penalised four times as hard as mistaking 0 for 1). A future variant should
> compare against one-hot targets + cross-entropy. Tracked in
> `docs/IMPLEMENTATION_PLAN.md` (I10.4).

---

## 4. Results

### 4.1 Baseline NN

| Metric | Value |
|--------|-------|
| Architecture | `{{baseline_nn_arch}}` |
| Parameters | {{baseline_nn_params}} |
| MSE (normalised) | {{baseline_nn_mse:.5f}} |

### 4.2 EM Iteration Summary

| Iter | GP Pareto | GP Best MSE | NN Architecture | NN Params | NN MSE | Residual MSE | GP Time (s) | NN Time (s) |
|------|-----------|-------------|-----------------|-----------|--------|--------------|-------------|-------------|
{{iteration_table_rows}}

### 4.3 Parameter Change

Baseline parameters: **{{baseline_nn_params}}**
Final GP-enriched NN parameters: **{{final_nn_params}}**
Change: **{{param_reduction_direction}} by {{param_reduction_pct:.1f}}%**

*Figure 1 — EM progress (residual MSE + NN params per iteration):*
![EM Progress](figures/em_progress.png)

*Figure 2 — NN size comparison:*
![NN Size](figures/nn_size_comparison.png)

### 4.4 GP Pareto Fronts

{{pareto_expressions_block}}

*Figure 3 — Pareto front scatter (final iteration):*
![Pareto](figures/pareto_iter_{{last_iter_id}}.png)

*Figure 4 — Top GP trees (final iteration):*
![GP Trees](figures/gp_trees_iter_{{last_iter_id}}.png)

---

## 5. Discussion

*[Stub — expand with interpretation, limitations, open questions]*

- The parameter count {{param_reduction_direction}} by {{param_reduction_pct:.1f}}% across {{n_iterations}} iterations.
- The residual MSE after iteration {{last_iter_id}} is `{{final_residual_mse:.5f}}`.
- *Open question:* Does the residual converge to a structurally simpler pattern after each iteration?
- *Limitation:* The current pipeline uses a fixed GP configuration across all iterations.
  Adaptive hyperparameters (e.g. smaller `nodes_max` in later iterations when residual is smoother)
  may improve convergence speed.

---

## 6. Conclusion

*[Stub — expand for final paper]*

We demonstrated a pipeline in which GP-derived symbolic features progressively reduce
the required NN parameter budget while maintaining prediction quality.  The approach is
generic and benchmark-independent; the pipeline auto-generates this blueprint from any
experiment run.

---

## References

*[Stub — add relevant citations before publication]*

- Kuncheva, L.I. (2014). *Combining Pattern Classifiers*. §5.3 Oracle bound.
- Caruana et al. (2004). *Ensemble Selection from Libraries of Models*.
- Vanneschi et al. (2010). *Measuring Bloat, Overfitting and Search Difficulty in GP*.
- Koza, J.R. (1992). *Genetic Programming*.

---

*Generated by `benchmarks/nn_gp/paper_blueprint.py` on {{date}}.*

