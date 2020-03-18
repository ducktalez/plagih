# PLAGIH Genetic Programming (Name is still needed)

Genetic programming framework.
Its main goal is to improve a human written program without making too many changes.

*Please note that a lot of stuff is about to change in the future (last update 18.03.2020)*

## How to

### Example run
Run `start.py`
 
This will automatically start the run in `runs/example1_cartpole/`.

### Starting your own run

- create a folder with all needed files (see Tutorial or example)
- `python start.py –i <folder_name>`

### Run folder structure

The input parameter must be a folder with the following structure:
- `/run_files/operators.csv`: Set of mathematical operators to build programs with
- `/run_files/samples.csv`: Target-data with states + actions. (Caution, weird syntax)
- `/run_files/tree_labels.csv`: Self-written program (as label-list)
- `Config.json` (Optional): GP parameters like `pop_size`, `evolve_rates`


### How to: behaviour_samples.csv

The data set contains decisions and the associated observations that the agent has made. 
- Exploration should be set to 0
- (`type` options are currently either `float` or `bool`)

The .csv must (currently) have a header line that looks like this example:

| observation0:float | observation1:float | action0:float |
|:------------------ | ------------------ |:------------- |
| 0.1                | 0.2                | 0.3           |
| 0.1                | 0.2                | 0.3           |

- observations starting with `observation`
- One action, called `action0`
- types (`float`, `bool`) are separated with `:`

### How to: operators.csv

This file contains all possible operators (one per line).
Operators get randomly picked, so adding some more often will change the result.

Possible operators are:

|Group|examples|
|:------------------ |:-----------------|
|Mathematical operators|`+`, `-`, `*`, `/`, `**`, `abs`, `sign`, `square`, `sqrt`, `log`, `log1p`, `cos`, `sin`, `tan`, `acos`, `asin`, `atan`, `Mini`, `Maxi`|
|Logical operators | `&`, `Or`, `Xor`, `Nand`, `Xand`, `Nor`, `Xnor`, `Not`|
|Comparative operators|`==`, `!=`, `<`, `<=`, `>`, `>=`|
|Conditional (If-then-else)|`Ifte`|


### How to: tree_labels.csv

(More convenient way coming soon)
1. Write a computer program to solve the problem. (At least a logical structure)
2. Display it as computational tree (functions: see below or in `op`-array in `plagih/modules/dicts`)
3. Breadth-first search the tree, align the labels as list. Types must match.

#### Tree Example

Code:
```
if (observation0 < 0):
    return 0
else:
    return 2
```
`tree_labels.csv`:
```
label_list,Ifte, <, 0, 2, observation0, 0
```

`tree_labels.csv` (with `if`,`return 0`,`return 2` as fix nodes):
```
label_list,Ifte, <, 0, 2, observation0, 0
modify_list,0,1,0,0,1,1
```


## ideas for names
- The Elves and the Shoemaker
- plausible, explainable, annex, interpretable, traceable, comprehensible

## Features

- python3
- Genetic Programming features
    - Tensorflow-evaluation (offline)
    - strongly typed

## Description

The Genetic-Programming Framework is primarily intended to extend a human written program to achieve the same performance as a (better) NN solution. Decisive for the "explainability" is the number of changes to the reference program that are necessary to get to the target solution ("tree edit distance").    

PLAGIH stands for PLAusible Genetic Improvements to Heuristics. The name will probably be changed soon.
This is a Project resulting from my Masters Thesis (with yet unknown name). 
Its aim is to provide a framework where a heuristic is improved by genetic programming towards a machine learned approach. 
The new code shall both perform significantly better and also be an explainable, 
plausible addition to the original program which the developer does understand.


### All included Plagih stuff
Todo: Describe all files and folders + their functions here

## Run Plagih
Required packages: `tensorflow` `numpy` `sympy` `Apted` `pickle`

Optional: `tikzplotlib` (For additional Latex-graph)


