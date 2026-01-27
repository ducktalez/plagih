"""
Example usage of the unified tree visualization system.

This module demonstrates how to use the tree_renderer for:
1. Normal GP trees with different orientations
2. Merged evaluation graphs with different display modes
3. Custom styling and configuration

Run this file directly to generate example visualizations:
    python visualization/render_examples.py
"""

import sympy
from pathlib import Path

# Import tree structures
from plagih.trees import Add, Mul, Sin, Cos, Number, Symbol

# Import visualization
from visualization.tree_renderer import (
    render_tree,
    render_merged_tree,
    TreeRenderer,
    TreeRendererConfig,
    Orientation,
    MergedDisplayMode,
)


def example_basic_tree():
    """Example 1: Basic tree visualization with different orientations."""
    print("\n" + "="*60)
    print("Example 1: Basic Tree Visualization")
    print("="*60)

    # Create a simple expression tree: (x + 2) * sin(y)
    tree = Mul(
        Add(Symbol(sympy.Symbol('x')), Number(2)),
        Sin(Symbol(sympy.Symbol('y')))
    )

    # Render with different orientations
    orientations = {
        "TB": "Top-to-Bottom (root at top)",
        "BT": "Bottom-to-Top (root at bottom)",
        "LR": "Left-to-Right (root at left)",
        "RL": "Right-to-Left (root at right)",
    }

    for orient, description in orientations.items():
        path = render_tree(
            tree,
            filename=f"example_simple_{orient.lower()}",
            orientation=orient,
            title=f"(x + 2) * sin(y)\n{description}"
        )
        print(f"  Created: {path}")


def example_complex_tree():
    """Example 2: More complex tree."""
    print("\n" + "="*60)
    print("Example 2: Complex Tree")
    print("="*60)

    # Create a more complex expression
    tree = Add(
        Mul(Number(3), Symbol(sympy.Symbol('x'))),
        Cos(Add(Symbol(sympy.Symbol('y')), Number(1))),
        Sin(Mul(Symbol(sympy.Symbol('z')), Number(2)))
    )

    # Render with custom config
    config = TreeRendererConfig()
    config.min_level_gap = 1.0  # More vertical space
    config.min_sibling_gap = 0.5  # More horizontal space

    path = render_tree(
        tree,
        filename="example_complex_custom",
        orientation="TB",
        config=config,
        title="3*x + cos(y+1) + sin(z*2)"
    )
    print(f"  Created: {path}")


def example_merged_tree():
    """Example 3: Merged evaluation graph visualization."""
    print("\n" + "="*60)
    print("Example 3: Merged Evaluation Graph")
    print("="*60)

    from plagih.population_merge import build_one_evaluation_tree

    # Create a small population of trees that share subexpressions
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')

    tree1 = Add(Symbol(a), Symbol(b))              # a + b
    tree2 = Mul(Add(Symbol(a), Symbol(b)), Symbol(c))  # (a + b) * c
    tree3 = Add(Add(Symbol(a), Symbol(b)), Number(1))  # (a + b) + 1

    population = [tree1, tree2, tree3]

    # Build merged graph
    graph = build_one_evaluation_tree(population)

    # Render with label-only mode (compact)
    path1 = render_merged_tree(
        graph,
        filename="example_merged_label",
        orientation="BT",
        display_mode="label",
        title="Merged Graph - Label Only Mode"
    )
    print(f"  Created (label mode): {path1}")

    # Render with full expression mode (detailed)
    path2 = render_merged_tree(
        graph,
        filename="example_merged_expr",
        orientation="LR",
        display_mode="expression",
        title="Merged Graph - Expression Mode"
    )
    print(f"  Created (expression mode): {path2}")


def example_programmatic_rendering():
    """Example 4: Programmatic usage with TreeRenderer class."""
    print("\n" + "="*60)
    print("Example 4: Programmatic Rendering")
    print("="*60)

    import matplotlib.pyplot as plt

    # Create a custom configuration
    config = TreeRendererConfig(
        orientation=Orientation.LEFT_RIGHT,
        min_level_gap=1.2,
        min_sibling_gap=0.6,
        figure_dpi=200,
    )

    # Create renderer
    renderer = TreeRenderer(config)

    # Create tree
    tree = Mul(
        Add(Symbol(sympy.Symbol('x')), Number(1)),
        Symbol(sympy.Symbol('y'))
    )

    # Render to figure (without saving)
    fig, ax = renderer.render_tree(tree, title="Programmatic Rendering")

    # You can now customize the figure further
    ax.annotate('Root', xy=(0, 0), fontsize=8, color='red')

    # Save manually
    output_path = Path(__file__).parent.parent / "tree_output" / "example_programmatic.png"
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"  Created: {output_path}")


def main():
    """Run all examples."""
    print("="*60)
    print("Tree Visualization Examples")
    print("="*60)

    example_basic_tree()
    example_complex_tree()
    example_merged_tree()
    example_programmatic_rendering()

    print("\n" + "="*60)
    print("All examples complete!")
    print("Check the tree_output/ folder for generated images.")
    print("="*60)


if __name__ == "__main__":
    main()
