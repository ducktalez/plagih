# PLAGIH Genetic Programming (Name will change (soon?))

TODO. This read-me is outdated.

...is a genetic programming framework.

compared to other gp frameworks, this one provides the opportunity to use human written code as basis for the gp-process.


*Please note that a lot of stuff is about to change in the future (last update 18.03.2020)*

Main features:
- Tree-based genetic programming
- tensorflow-based evaluation
- visualisation with latex
- (example available)


## Python 3.7 packages

Anaconda 3.7 (2020.2) with conda packages. Some packages are only available in pip though.

Conda packages:
pathlib
sympy
apted
tensorflow
sklearn
pandas
sympy
tikzplotlib (optional)
gym (optional)

non-conda packages:
apted

update of the yaml module might be necessary:
conda install -U PyYaml
or
pip install -U PyYaml


### Example run
Run `start.py`
 
This will automatically start the run in `/runs/example1_cartpole/`.

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

|Group|examples|
|:------------------ |:-----------------|
|Mathematical operators|`+`, `-`, `*`, `/`, `**`, `abs`, `sign`, `Square`, `sqrt`, `log`, `log1p`, `cos`, `sin`, `tan`, `acos`, `asin`, `atan`, `Mini`, `Maxi`|
|Logical operators | `Andb`, `Orb`, `not`|
|Comparative operators|`==`, `!=`, `<`, `<=`, `>`, `>=`|
|Conditional (If-then-else)|`Ifte`|


### How to: tree_labels.csv

(More convenient way coming soon)
1. Write a computer program to solve the problem. (At least, provide a logical structure)
2. Display it as computational tree (functions: see below or in `op`-array in `plagih/modules/dicts`)
3. Breadth-first search the tree, align the labels as list. Types must match.

#### Tree Example

Code:
```
if (cartPos < 0):
    return 1
else:
    return 2
```
`tree_labels.csv`:
```
label_list,Ifte, <, 1, 2, cartPos, 0
```

`tree_labels.csv` (with `if`,`return 1`,`return 2` as fix nodes):
```
label_list,Ifte, <, 1, 2, cartPos, 0
modify_list,0,1,0,0,1,1
```

...Breadth-first seems counter-intuitive, but that is what Karoo gave me :P

### How I analyse results:
- `plots/average-fitness.png` - to see if there was an improvement
- `plots/best_candidate.png` - for the current best solution
- `plots/pareto.png` - showing the best candidates related to the complexity. Find your prefered combination of performance and complexity
- `info/pareto.txt` - pareto-efficient candidates in math-expression
- `trees/#all_trees.tex` - A Latex file with all computational trees visualized


## ====Everything from here is garbage====

## ideas for names
- The Elves and the Shoemaker
- plausible, explainable, annex, interpretable, traceable, comprehensible

## Features

- python3
- Genetic Programming features
    - Tensorflow-evaluation (offline)
    - strongly typed

## Description

The Genetic-Programming Framework is primarily intended to extend a human written program to achieve the same performance as a (better) NN solution. Decisive for the "explainability" is the number of changes to the reference program that are necessary to get to the target solution ("tree_edit_distance").    

PLAGIH stands for PLAusible Genetic Improvements to Heuristics. The name will probably be changed soon.
This is a Project resulting from my Masters Thesis (with yet unknown name). 
Its aim is to provide a framework where a heuristic is improved by genetic programming towards a machine learned approach. 
The new code shall both perform significantly better and also be an explainable, 
plausible addition to the original program which the developer does understand.


### All included Plagih stuff

## Run Plagih
Required packages: `tensorflow` `numpy` `sympy` `Apted` `pickle`

Optional: `tikzplotlib` (For additional Latex-graph)


