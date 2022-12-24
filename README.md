# Familiar Genetic Programming with TensorFlow

- Lightning fast. Believe me.
- 

## Ablage/Todos

- trees with one node just very rarely?
- https://deap.readthedocs.io/en/master/api/tools.html

- tree evaluate nodes for best improvement 
- population cluster / races
- Make regular Function Classes Sympy Functions
- introduce integer
- replacing
  - replace x**2 with Squared(x)
  - replace -1*x, coming from sympy functions
  - 
- Tests for:
  - Auto-testruns: loop/reload through [random, origin, origin_fixed] [MC, IB]
  - TF-evaluation equals python-evaluation equals sympy evaluation
- BackPropagation through nodes, rank value for whole tree 
- separate monitoring class
- introduce NN in alpha-tree, at well-mutable nodes
- evaluate one very large TF-graph containing the whole population
- parallelisation
- evolve-operatoren: werte runden, runden einbauen,

Always check
- xxxxx
- sfeh
- asd

## Compared to DEAP
- DEAP has non-Programming options (altering arrays). Plagih is only for GP.
- complexity node-count based, instead of depth-baased only.
- trees that are not lists

## crazy ideas
- evolution+ranking of node evolutions
- GP individuals for evolution process

## Checks for when you have altered code
- Functions need to be in ops_dict (and some more)
- reconstruction of trees

Attention: If you want to write your own code, look for important developer informations in the section below!

## name ideas
- AnnaGP
- FamGP (Familiar GP)
- plagih: plausible genetic improvements
  ...is a genetic programming framework. 
- It is specialized on fast and efficient evaluation (faster than KarooGP) and especially introduces the concept of familiarity.

## Paper ideas
- unified mathematics as performance improvement for gp
- cluster-based evolution of subpopulations / races
- Recombining if cases in 
- BackPropagation through nodes, rank value for whole tree
- Genetic Backpropagation
- transforming the whole population into one graph, bottom to top. Do something with the "reversed" tree

More ideas:
- EM algo gp/nn process
  Step 1: Train a gp process to replace a NN.
  Step 2: Allow the pareto candidates as input to a nn, making the network smaller
- propagate matrizes of gradual improvements

Matrix

## Intro

compared to other gp frameworks, this one provides the opportunity to use human written code as basis for the gp-process.


*Please note that a lot of stuff is about to change in the future (last update 18.03.2020)*

Main features:
- Tree-based genetic programming
- tensorflow-based evaluation
- visualisation with latex
- (example available)


## Python 3.9 packages

I am using Anaconda. Some packages are only available in pip.
sfeh:save this as requirements

Conda packages:
matplotlib
pathlib
sympy
apted
tensorflow
sklearn
pandas
sympy
tikzplotlib (optional)
gym (optional)

`conda install matplotlib pathlib sympy apted tensorflow tensorflow-gpu pandas sympy gym`
`pip install sklearn gym apted tikzplotlib`

non-conda packages:
apted

update of the yaml module might be necessary:
conda install -U PyYaml
or
pip install -U PyYaml

Also, plotting with latex might require texlife:

`sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super`

### Example run
`python plagih_gp.py`
 
This will automatically start a run (probably Mountaincar).

### Starting your own run

- create a folder with all needed files (see Tutorial or example)
- `python start.py –i <folder_name>`

### Run folder structure

The input parameter must be a folder with the following content:
- `/run_files/operators.csv`: Set of mathematical operators to build programs with
- `/run_files/samples.csv`: Target-data with states + actions. (Caution, weird syntax)
- `/run_files/tree_labels.csv`: Self-written program (as label-list)
- `Config.yaml` (Optional): GP parameters like `pop_size`, `evolve_rates`


### How to: behaviour_samples.csv

The data set contains decisions and the associated observations that the agent has made. 
- Exploration should be set to 0
- (`type` options are currently either `float` or `bool`)

The .csv must (currently) have a header line that looks like this example:

| cartPos:float | cartVel:float | action0:float |
|:------------------ | ------------------ |:------------- |
| 0.1                | 0.2                | 0.3           |
| 0.1                | 0.2                | 0.3           |

- observations starting with `observation`
- One action, called `action0`
- types (`float`, `bool`) are separated with `:`

In the initial run, the entries are split in train- and test-data.
Also, the structure of the entries are analysed for the gp-process.

The result is then saved as `samples_ready.p`.
If this file exists, it is loaded instead of the .csv file.

### How to: operators.csv

This file contains all possible operators (one per line).
Operators get randomly picked, so adding some more often will change the result.

Possible operators are:

| Group                      | examples                                                                                                                                         |
|:---------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| Mathematical operators     | `+`, `-`, `*`, `/`, `**`, `abs`, `sign`, `Square`, `sqrt`, `log`, `log1p`, `cos`, `sin`, `tan`, `acos`, `asin`, `atan`, `BinaryMin`, `BinaryMax` |
| Logical operators          | `BinaryAnd`, `BinaryOr`, `not`                                                                                                                   |
| Comparative operators      | `==`, `!=`, `<`, `<=`, `>`, `>=`                                                                                                                 |
|                            |                                                                                                                                                  |
| Conditional (If-then-else) | `Ifte`                                                                                                                                           |


### How to: tree_labels.csv

(More convenient way coming soon)
1. Write a computer program to solve the problem. (At least, provide a logical structure)
2. Display it as computational tree (functions: see below or in `op`-array in `plagih/modules/dicts`)
3. Breadth-first search the tree, align the labels as list. Types must match.


### How I analyse results:
- `plots/average-fitness.png` - to see if there was an improvement
- `plots/best_candidate.png` - for the current best solution
- `plots/pareto.png` - showing the best candidates related to the complexity. Find your prefered combination of performance and complexity
- `info/pareto.txt` - pareto-efficient candidates in math-expression
- `trees/#all_trees.tex` - A Latex file with all computational trees visualized


## ====Everything below here is garbage====

## ideas for names
- The Elves and the Shoemaker
- plausible, explainable, annex, interpretable, traceable, comprehensible

## Features

- python3
- Genetic Programming features
    - Tensorflow-evaluation (offline)
    - strongly typed

### Tree architecture
The evolution process uses the Node/Fintree class, but there are 4 tree versions which need conversion.
1. Loadable trees (nested List, E.g. ['+':fix, ['a'], ['b']])
2. The tree class
3. Sympy-trees
4. Tensorflow graphs

Sympy trees need a lot of conversion options.

## Description

The Genetic-Programming Framework is primarily intended to extend a human written program to achieve the same 
performance as a (better) NN solution. Decisive for the "explainability" is the number of changes to the reference 
program that are necessary to get to the target solution ("tree_edit_distance").    

PLAGIH stands for PLAusible Genetic Improvements to Heuristics. The name will probably be changed soon.
This is a Project resulting from my Masters Thesis (with yet unknown name). 
Its aim is to provide a framework where a heuristic is improved by genetic programming towards a machine learned approach. 
The new code shall both perform significantly better and also be an explainable, 
plausible addition to the original program which the developer does understand.


### All included Plagih stuff


## Developer information
- Do not save paths in pickle between runs; if required, only save strings. 
- Otherwise, pickle files between systems may not work.

### The structure of the code

The plagih_gp run depends on the following modules:
- a loop, in which new generations of evolved trees are created
- trees, which are a recursive node-structure, that allow evolution
- 