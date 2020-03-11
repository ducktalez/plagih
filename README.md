# PLAGIH (I still need a name for this :P)

PLAGIH stands for PLAusible Genetic Improvements to Heuristics. The name will probably be changed soon.
This is a Project resulting from my Masters Thesis (with yet unknown name). 
Its aim is to provide a framework where a heuristic is improved by genetic programming towards a machine learned approach. 
The new code shall both perform significantly better and also be an explainable, 
plausible addition to the original program which the developer does understand.

## required packages
TODO Apted
tensorflow
numpy
sympy
pickle


## Including

- Karoo-based gp library
- mountain car tests

## How to use this

### Creating behaviour samples:
1. Find a good agent for your problem or train one.
2. Let the trained agent perform a decent amount of times
3. save the observed states and the chosen actions as .csv of this format:

| observation0:float | observation1:float | action0:float |
|:------------------ | ------------------ |:------------- |
| 0.1                | 0.2                | 0.3           |
| 0.1                | 0.2                | 0.3           |

(`type` options are currently either `float` or `bool`)

### Prepare your own code
(More convenient way coming soon)
1. Write your own heuristic solution
2. Convert it to a computational tree using the functions specified in the 'op'-array in plagih/modules/dicts.
3. Breadth-first search the tree and align the labels as list. Types must match.

Example: `['Ifte', '<', 0, 2, 'observation0', 0]`

Here is a set of possible operators:

|Group|examples|
|:------------------ |:-----------------|
|Mathematical operators|`+`, `-`, `*`, `/`, `**`, `abs`, `sign`, `square`, `sqrt`, `log`, `log1p`, `cos`, `sin`, `tan`, `acos`, `asin`, `atan`, `Mini`, `Maxi`|
|Logical operators | `&`, `Or`, `Xor`, `Nand`, `Xand`, `Nor`, `Xnor`, `Not`|
|Comparative operators|`==`, `!=`, `<`, `<=`, `>`, `>=`|
|Conditional (If-then-else)|`Ifte`|

### Start run
1. specify the configuration in plagih/plagih_gp.py
2. start `start.py` in the main folder