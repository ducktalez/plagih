"""
GP Monitoring Module

Provides a flexible monitoring system for tracking GP evolution metrics.
Replaces the simple pandas DataFrame approach with a more flexible class-based solution.

Usage:
    monitor = GPMonitor()
    monitor.record_generation(gen_id, population, gen_time, ...)
    monitor.plot_performance(save_path)
    monitor.to_dataframe()  # For compatibility with existing code
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from pathlib import Path
import json
import time
import numpy as np

if TYPE_CHECKING:
    from .trees import Candidate


@dataclass
class GenerationMetrics:
    """Stores metrics for a single generation.

    Attributes:
        gen_id: Generation number.
        timestamp: When this generation was recorded (seconds since monitor start).
        metrics: Dictionary of metric name -> value.
    """
    gen_id: int
    timestamp: float
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to metrics."""
        return self.metrics[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dict-like setting of metrics."""
        self.metrics[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric with default value."""
        return self.metrics.get(key, default)


class GPMonitor:
    """Flexible monitoring system for GP evolution.

    Features:
    - Track arbitrary metrics per generation
    - Compute derived metrics automatically
    - Export to various formats (DataFrame, JSON, CSV)
    - Support for custom callbacks (on_generation, on_improvement, on_pareto_update)
    - Built-in plotting functionality

    Example:
        >>> monitor = GPMonitor()
        >>> monitor.record_generation(
        ...     gen_id=0,
        ...     population=population,
        ...     gen_time=1.5,
        ...     pareto_updated=True
        ... )
        >>> monitor.plot_performance('monitoring.png')
    """

    def __init__(self,
                 custom_metrics: Optional[Dict[str, Callable]] = None,
                 auto_compute: bool = True):
        """Initialize the monitor.

        Args:
            custom_metrics: Dict of metric_name -> callable(population) for custom metrics.
            auto_compute: Whether to automatically compute standard metrics.
        """
        self.generations: List[GenerationMetrics] = []
        self.custom_metrics = custom_metrics or {}
        self.auto_compute = auto_compute
        self.start_time = time.perf_counter()

        # Track Pareto front changes
        self.gens_since_last_pareto = 0
        self._best_fitness_ever = np.inf

        # Callbacks
        self._on_generation_callbacks: List[Callable[[GenerationMetrics], None]] = []
        self._on_improvement_callbacks: List[Callable[[GenerationMetrics, float], None]] = []
        self._on_pareto_update_callbacks: List[Callable[[GenerationMetrics], None]] = []

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_generation(self, callback: Callable[[GenerationMetrics], None]):
        """Register callback for every generation end.

        Args:
            callback: Function(metrics) called after each generation.

        Can be used as decorator:
            @monitor.on_generation
            def my_callback(metrics):
                print(f"Gen {metrics.gen_id} done!")
        """
        self._on_generation_callbacks.append(callback)
        return callback

    def on_improvement(self, callback: Callable[[GenerationMetrics, float], None]):
        """Register callback for when best fitness improves.

        Args:
            callback: Function(metrics, improvement_amount) called on improvement.
        """
        self._on_improvement_callbacks.append(callback)
        return callback

    def on_pareto_update(self, callback: Callable[[GenerationMetrics], None]):
        """Register callback for Pareto front updates.

        Args:
            callback: Function(metrics) called when Pareto front changes.
        """
        self._on_pareto_update_callbacks.append(callback)
        return callback

    def add_custom_metric(self, name: str, func: Callable):
        """Add a custom metric computation.

        Args:
            name: Metric name.
            func: Function that takes (population) and returns a value.

        Example:
            monitor.add_custom_metric('avg_depth', lambda pop: np.mean([c.tree.depth for c in pop]))
        """
        self.custom_metrics[name] = func

    # =========================================================================
    # Recording
    # =========================================================================

    def record_generation(self,
                          gen_id: int,
                          population: List['Candidate'],
                          gen_time: float,
                          pareto_updated: bool = False,
                          lut_size: int = 0,
                          extra_metrics: Optional[Dict[str, Any]] = None):
        """Record metrics for a generation.

        Args:
            gen_id: Generation number.
            population: List of Candidate objects.
            gen_time: Time taken for this generation.
            pareto_updated: Whether Pareto front was updated this generation.
            lut_size: Size of the lookup table.
            extra_metrics: Additional custom metrics to record.
        """
        # Update Pareto tracking
        if pareto_updated:
            self.gens_since_last_pareto = 0
        else:
            self.gens_since_last_pareto += 1

        # Create metrics entry
        metrics = GenerationMetrics(
            gen_id=gen_id,
            timestamp=time.perf_counter() - self.start_time
        )

        # Basic metrics
        metrics['gen_time'] = gen_time
        metrics['gens_since_pareto'] = self.gens_since_last_pareto
        metrics['lut_size'] = lut_size

        # Population metrics
        if self.auto_compute and population:
            pop_metrics = self._compute_population_metrics(population)
            metrics.metrics.update(pop_metrics)
        elif not population:
            metrics.metrics.update(self._empty_population_metrics())

        # Custom metrics
        for name, func in self.custom_metrics.items():
            try:
                metrics[name] = func(population)
            except Exception:
                metrics[name] = np.nan

        # Extra metrics
        if extra_metrics:
            metrics.metrics.update(extra_metrics)

        self.generations.append(metrics)

        # Check for improvement and fire callbacks
        current_best = metrics.get('fit_best', np.inf)
        if np.isfinite(current_best) and current_best < self._best_fitness_ever:
            improvement = self._best_fitness_ever - current_best
            self._best_fitness_ever = current_best
            for cb in self._on_improvement_callbacks:
                try:
                    cb(metrics, improvement)
                except Exception:
                    pass

        # Fire pareto update callbacks
        if pareto_updated:
            for cb in self._on_pareto_update_callbacks:
                try:
                    cb(metrics)
                except Exception:
                    pass

        # Fire generation callbacks
        for cb in self._on_generation_callbacks:
            try:
                cb(metrics)
            except Exception:
                pass

    def _compute_population_metrics(self, population: List['Candidate']) -> Dict[str, Any]:
        """Compute standard population metrics."""
        fitnesses = np.array([c.get_fitness() for c in population])
        parsimony = np.array([c.get_parsim() for c in population])

        # Count unique expressions
        try:
            unique_exprs = set(str(c.tree.get_sympy_expr()) for c in population)
            n_unique = len(unique_exprs)
        except Exception:
            n_unique = len(population)

        return {
            # Population size
            'pop_size': len(population),
            'pop_unique': n_unique,
            'diversity_ratio': n_unique / len(population) if population else 0,

            # Fitness statistics
            'fit_best': float(np.min(fitnesses)),
            'fit_worst': float(np.max(fitnesses)),
            'fit_mean': float(np.mean(fitnesses)),
            'fit_std': float(np.std(fitnesses)),
            'fit_median': float(np.median(fitnesses)),
            'fit_q25': float(np.percentile(fitnesses, 25)),
            'fit_q75': float(np.percentile(fitnesses, 75)),

            # Parsimony statistics
            'parsim_best': int(np.min(parsimony)),
            'parsim_worst': int(np.max(parsimony)),
            'parsim_mean': float(np.mean(parsimony)),
            'parsim_std': float(np.std(parsimony)),
            'parsim_median': float(np.median(parsimony)),
            'parsim_q25': float(np.percentile(parsimony, 25)),
            'parsim_q75': float(np.percentile(parsimony, 75)),
        }

    def _empty_population_metrics(self) -> Dict[str, Any]:
        """Return NaN/zero metrics for empty population."""
        return {
            'pop_size': 0,
            'pop_unique': 0,
            'diversity_ratio': 0,
            'fit_best': np.nan,
            'fit_worst': np.nan,
            'fit_mean': np.nan,
            'fit_std': np.nan,
            'fit_median': np.nan,
            'fit_q25': np.nan,
            'fit_q75': np.nan,
            'parsim_best': np.nan,
            'parsim_worst': np.nan,
            'parsim_mean': np.nan,
            'parsim_std': np.nan,
            'parsim_median': np.nan,
            'parsim_q25': np.nan,
            'parsim_q75': np.nan,
        }

    # =========================================================================
    # Access Methods
    # =========================================================================

    def __len__(self) -> int:
        """Return number of recorded generations."""
        return len(self.generations)

    def __getitem__(self, gen_id: int) -> GenerationMetrics:
        """Get metrics for a specific generation by ID."""
        for gen in self.generations:
            if gen.gen_id == gen_id:
                return gen
        raise KeyError(f"Generation {gen_id} not found")

    def get_metric_series(self, metric_name: str) -> np.ndarray:
        """Get a metric across all generations as numpy array."""
        return np.array([g.get(metric_name, np.nan) for g in self.generations])

    def get_generation_ids(self) -> np.ndarray:
        """Get array of generation IDs."""
        return np.array([g.gen_id for g in self.generations])

    @property
    def latest(self) -> Optional[GenerationMetrics]:
        """Get the most recent generation metrics."""
        return self.generations[-1] if self.generations else None

    @property
    def best_fitness(self) -> float:
        """Get the best fitness seen so far."""
        return self._best_fitness_ever

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_dataframe(self):
        """Convert to pandas DataFrame for compatibility.

        Returns:
            pandas.DataFrame with generation index.
            Column names are mapped for backwards compatibility.
        """
        import pandas as pd

        if not self.generations:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'pop_len', 'pop_unique', 'lut_symex_fitness-len', 'time',
                'fit_avg', 'fit_var', 'fit_quantile_25', 'fit_quantile_50',
                'fit_quantile_75', 'fit_best', 'parsim_avg', 'parsim_var',
                'parsim_quantile_25', 'parsim_quantile_50', 'parsim_quantile_75',
                'parsim_best', 'gens_since_last_pareto'
            ])

        # Collect all unique metric names
        all_metrics = set()
        for gen in self.generations:
            all_metrics.update(gen.metrics.keys())

        # Build data dict
        data = {metric: [] for metric in all_metrics}
        index = []

        for gen in self.generations:
            index.append(gen.gen_id)
            for metric in all_metrics:
                data[metric].append(gen.get(metric, np.nan))

        df = pd.DataFrame(data, index=index)

        # Rename columns for backwards compatibility with existing code
        rename_map = {
            'pop_size': 'pop_len',
            'gen_time': 'time',
            'fit_mean': 'fit_avg',
            'fit_std': 'fit_var',
            'parsim_mean': 'parsim_avg',
            'parsim_std': 'parsim_var',
            'fit_q25': 'fit_quantile_25',
            'fit_median': 'fit_quantile_50',
            'fit_q75': 'fit_quantile_75',
            'parsim_q25': 'parsim_quantile_25',
            'parsim_median': 'parsim_quantile_50',
            'parsim_q75': 'parsim_quantile_75',
            'gens_since_pareto': 'gens_since_last_pareto',
            'lut_size': 'lut_symex_fitness-len',
        }

        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        return df

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def convert_value(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        return {
            'generations': [
                {
                    'gen_id': g.gen_id,
                    'timestamp': g.timestamp,
                    **{k: convert_value(v) for k, v in g.metrics.items()}
                }
                for g in self.generations
            ],
            'summary': {
                'total_generations': len(self.generations),
                'total_time': self.generations[-1].timestamp if self.generations else 0,
                'best_fitness': float(self.best_fitness) if np.isfinite(self.best_fitness) else None,
            }
        }

    def to_json(self, path: Optional[Path] = None, indent: int = 2) -> str:
        """Export to JSON string, optionally save to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent, default=str)

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(json_str)

        return json_str

    def to_csv(self, path: Path):
        """Export to CSV file."""
        df = self.to_dataframe()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def plot_performance(self, save_path: Path, show: bool = False):
        """Plot performance metrics over generations.

        Creates a 2x2 figure with:
        - Fitness evolution
        - Parsimony evolution
        - Population diversity
        - Time per generation
        """
        import matplotlib.pyplot as plt

        if not self.generations:
            print("Warning: No generations to plot")
            return

        gen_ids = self.get_generation_ids()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Fitness Evolution
        ax1 = axes[0, 0]
        fit_best = self.get_metric_series('fit_best')
        fit_mean = self.get_metric_series('fit_mean')
        fit_q25 = self.get_metric_series('fit_q25')
        fit_q75 = self.get_metric_series('fit_q75')

        ax1.plot(gen_ids, fit_best, 'b-', linewidth=2, label='Best')
        ax1.plot(gen_ids, fit_mean, 'g--', label='Mean')
        ax1.fill_between(gen_ids, fit_q25, fit_q75, alpha=0.3, label='25-75%')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (lower is better)')
        ax1.set_title('Fitness Evolution')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Parsimony Evolution
        ax2 = axes[0, 1]
        parsim_best = self.get_metric_series('parsim_best')
        parsim_mean = self.get_metric_series('parsim_mean')
        parsim_q25 = self.get_metric_series('parsim_q25')
        parsim_q75 = self.get_metric_series('parsim_q75')

        ax2.plot(gen_ids, parsim_best, 'b-', linewidth=2, label='Best')
        ax2.plot(gen_ids, parsim_mean, 'g--', label='Mean')
        ax2.fill_between(gen_ids, parsim_q25, parsim_q75, alpha=0.3, label='25-75%')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Parsimony (complexity)')
        ax2.set_title('Parsimony Evolution')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Population Diversity
        ax3 = axes[1, 0]
        pop_size = self.get_metric_series('pop_size')
        pop_unique = self.get_metric_series('pop_unique')

        ax3.plot(gen_ids, pop_size, 'b-', label='Total')
        ax3.plot(gen_ids, pop_unique, 'r--', label='Unique')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Population Count')
        ax3.set_title('Population Diversity')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        # Time per Generation
        ax4 = axes[1, 1]
        gen_time = self.get_metric_series('gen_time')

        ax4.bar(gen_ids, gen_time, alpha=0.7, color='steelblue')
        if len(gen_time) > 0 and not np.all(np.isnan(gen_time)):
            ax4.axhline(np.nanmean(gen_time), color='red', linestyle='--',
                        label=f'Mean: {np.nanmean(gen_time):.2f}s')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Time per Generation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def summary(self) -> str:
        """Get a text summary of the monitoring data."""
        if not self.generations:
            return "No generations recorded."

        gen_times = self.get_metric_series('gen_time')
        parsim_best_series = self.get_metric_series('parsim_best')

        lines = [
            "=" * 50,
            "GP MONITORING SUMMARY",
            "=" * 50,
            f"Total generations: {len(self.generations)}",
            f"Total time: {self.generations[-1].timestamp:.2f}s",
            f"Avg time/generation: {np.nanmean(gen_times):.2f}s",
            "",
            "Fitness:",
            f"  Best ever: {self.best_fitness:.6f}",
            f"  Final best: {self.latest.get('fit_best', np.nan):.6f}",
            f"  Final mean: {self.latest.get('fit_mean', np.nan):.6f}",
            "",
            "Parsimony:",
            f"  Best: {np.nanmin(parsim_best_series):.0f}",
            f"  Final mean: {self.latest.get('parsim_mean', np.nan):.1f}",
            "",
            "Population:",
            f"  Final size: {self.latest.get('pop_size', 0)}",
            f"  Final unique: {self.latest.get('pop_unique', 0)}",
            f"  Final diversity: {self.latest.get('diversity_ratio', 0)*100:.1f}%",
            "=" * 50,
        ]

        return "\n".join(lines)

    def print_summary(self):
        """Print summary to console."""
        print(self.summary())
