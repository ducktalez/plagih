"""
NN+GP Co-Evolution Benchmark
=============================

Implements an EM-loop that alternates between:
  1. GP phase: Evolve symbolic candidates against (residual) target
  2. NN phase: Train minimal PyTorch MLP using GP candidate outputs as extra features

The goal is to find the smallest NN architecture that achieves baseline accuracy
when augmented with GP-derived symbolic features, and to iteratively symbolize
the remaining "residual logic" that the NN still needs to learn.

Entry point: ``benchmarks/nn_gp/run_mc.py``
"""
