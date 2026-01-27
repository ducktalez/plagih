# Familiar Genetic Programming with TensorFlow

- Fast
- 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


## Ablage/Todos

- Merged-tree mit chatgpt erstellen
- lint/black/etc einbauen
  - Empfohlene Tools: black, flake8, isort?
  - Wie würdest du das einbauen?
- pandas-Monitoring ersetzen?
  - pandas-monitoring wird momentan verwendet, um die Daten zu analysieren und zu visualisieren.
    Es ist aber nicht sehr flexibel und ich habe das Gefühl, dass es nicht optimal für meinen Anwendungsfall ist.
    Ich möchte es durch eine eigene Lösung ersetzen, die besser auf meine Bedürfnisse zugeschnitten ist.
    Hast du Vorschläge, wie ich das machen könnte? Vielleicht eine Monitoring-klasse? Was ist hier üblich?
- better structure
  - plagih_gp.py ist momentan das Hauptskript, welches den Ablauf steuert. Es ist aber sehr lang und unübersichtlich.
    Ich möchte es in mehrere Dateien aufteilen, z.B.:
    - main.py (Hauptablauf)
    - trees.py (Baumstruktur und Operationen)
    - evolution.py (Evolutionsprozess)
    - evaluation.py (Evaluation der Bäume)
    - visualization.py (Visualisierung der Bäume)
    - utils.py (Hilfsfunktionen)
  - Wie würdest du das strukturieren?
- self.evolve.origin_tree
- Anweisungen für Copilot (Instructions erstellen), am besten, sodass sie aich auch selbst notfalls erweitern.
- Dokumentation als .md/.pdf
- Pseudo-Backpropagation durch Bäume
- evaluation alternatives
  - tf-fun in every class
  - regular python code implementation

# Copilot Aufgaben

## Copilot-Anfrage für Baum-Merging-Strategien

# Sub-tasks

- save-evaluation: nan-handling, forcing real numbers
- discuss: allow_chain is probably not required at so many places...
- gen_create_initial -> create random pop if pop empty? with leftovers?
- tree -> evaluate nodes for best improvement 
- population cluster/races
- Tests for:
  - Auto-testruns: loop/reload through [random, origin, origin_fixed] [MC, IB]
  - TF-evaluation equals python-evaluation equals sympy evaluation
- BackPropagation through nodes, rank value for whole tree 
- introduce NN in alpha-tree, at well-mutable nodes
- evaluate one very large graph (TF?) containing the whole population
  - from low to high  (terminal to root)
    - list unique nodes/branches
    - use outputs as inputs for next layer
- Terminal-Mutation: Build tree with inputs, but only change terminals
- use sympy.count_ops() to count operators
- parallelisation
- numba.pydata.org https://www.youtube.com/watch?v=x58W9A2lnQc
- If no float-symbols found, return (1) true or (2) an operator? 
- sympy exprtools abchecken
- sfeh:discussion especially with mc: there can be more than one pareto entry with the same parsimony/fitness!
- build trees like sympy.factor() structure?
- Division-multiplicator node as non-len() chain input?
- print(sympy.parsing.sympy_parser.transformations)
- Symbol-time (for IB), choosing the time-step as input variable
- "Ban" trees, if they are too dominant
- Different print types for trees, also visualization
- make categorical options categorical. For mountain car, it is scalable, but a categorical options (aka a 3 nodes last layer) should be an option
- prevent the LUT of becoming too big; make a counter whenever a result is hit and delete the smallest in each cycle. Reset the numbers aswell.
- pseudo-backpropagation: ALL functions in a tree can be represented as a neural network. E. g. if-function is two input variables, a softmax layer and a result layer. replace a tree with a NN in the next step and train it.
- sympy facttor (up/downfactor), so it adds stuff together, expand(), 
- user-functions (=nodes with n inputs, given by a user)
- sfeh:idea sympy.nsimplify('3.333333*x+0.522', tolerance=0.1, rational=True) for
  - Terminals 
  - Even whole formulae!
  - Especially! Powers x**y
- Introduce rounding/clipping for many more functions
- adjust tournament_size to general fitness skew
- adding the pareto-trees visualized to the paretofront plot
- mutate chained operators specifically, crossover too. add summands, remove summands, .... as option.
- List of all potential inputs as layer, just multiplied with 1 or 1
- check if at least one node is forced!
- Introduce Tree-"styles", one expression can be represented in many ways
  - Raw (=as generated)
  - Isolate inputs in formulae as long as possible
  - Factorized
  - Simplified
  - create "better mutable" trees?
- Discuss: Input normalization. Leads to different formulae; which is NOT desired, right?
- Node/Number type rational?
- introduce a scale-oparetor, that only multiplies with a number, in order to be a good building block
- https://github.com/sympy/sympy/issues/27364
  ```python
  import sympy
  
  a = sympy.symbols('a', real=True)
  
  print(sympy.simplify('Max(a, 1) >= a', locals={'a': a}))  # a <= Max(1, a), should be True.  # noqa
  print(sympy.simplify('a <= Abs(-a)', locals={'a': a}))  # -> a <= Abs(a), should be True  # noqa
  print(sympy.simplify('a <= Abs(a)', locals={'a': a}))  # -> a <= Abs(a), should be True  # noqa
  ```
- Inputrange, outputrange, derivable, goalrequirement, iscyclic
- ``sympy.piecewise_exclusive()``
- Abs only with simplify
  - sympy.S('(Abs(a)*Abs(b))')
  - sympy.simplify('(Abs(a)*Abs(b))')
  - sympy.core.numbers.Zero
- lut for every subbranch ()
  - "free symbols" in tree?
- simplifications as evolution-factor
- nsimplify()
- variable-names = symbols
- Introduce min/max input operator, that is the number of the min/max value of inputs in the column.
- apted-discance can also compute the actual edit steps
- Crossover - make fix nodes insertable
- create_random should be the same for first generation and other randoms
- warning for sympifyable origin tree
- only allow variables in specific subtrees/branches
-  insert constants like pi, e, into trees
- choose_operator should be adjusted a little towards actually useful functions
- pop-mining?
  - trees too large? -> reduce size
  - trees too small?
  - too smiliar? -> more random new anything
  - too bad? -> increase tournament_size
  - wenn ein gp lauf immer wieder dieselben Lösungen findet, verbiete einige Grundstrukturen. ...oder andere einschränkungen. sin verbieten. wenn nichts besseres gefunden wird, weiter
- Bring back latex?
- group tree branches based on used variables?
- if constant filter process improved a tree -> try again with slightly smaller/bigger adaption
- CMA-ES, ffs. Deap does this, this is looooowest prio.
- separate populations training
  - Tree-structure mining; Good candidates may all have the same core. If they do, start new population with just this core? 
  - start separate population with specifically NOT this core?
- Partnering: trees search for other trees, that match solutions better
  - omg: what, if we simply make a df-results list for each datapoint and then
    lay them over each other, in order to see, which parts can be optimized?
    This might be big.
    Lets assume the df is the gene-sequence you hit correctly (rs error from 0 to 1) 
    and then, we look at the intersection of trees data results. Like with regular individuals, 
    in a partner search, we simply look for generally good partners that specifically have
    Good individuals can be crossed, but also; this can be the base for a piecewise
    analysis of the dataframe (sorted... somehow?), where another ML entity puzzles together
    the mosaiks of solutions. It can be a tree, too, but the sorting of the data is not obvious.
    The data should only be "sorted" according to how well it was predicted.
    Actually, we should make an ordered Dataframe, to do this.
    We just need a measurement, a metric, to value the amount of actual hits and non/bad hits.
    Similar to entropy, in order to know, which trees should be merged together.
- progress-print anzeigen des aktuell erzeugten Baum, zum durchlaufen
- nonzero-operator? making zero-ish inputs slightly positive? Also nan/complex-number exits?
- Creating "backpropagable"-tree -> tanh/sigmoid + exprcondtuples can mime nn-layers for if-conditions?
- Iterate between approximating with NNs and representing the NNs with GP-trees
  - NNs find the big differenced to a gp-tree and the solution
  - GP tries to find ifte-structures that mimic the NN behaviour
  - repeat
- Sleeppropagation GP Problem "mathematisch" nachbauen (mit GP zum Beispiel), dann neues env erzeugen 
 (mit dem fake-zeug), dann
 neu trainieren und selbst trainieren. 
- Zum Thema "großer Einheitsbaum": Hier könnte man Äste absterben lassen, wenn sie zu
  schlecht sind. Also z.B. wenn alle weiteren Äste unter diesem Knoten schlecht sind,
  und es über 1000 Stück schon gab, dann wird dieser Knoten an der Stelle gekillt.
- Komplexitätsmaß Assembler-operatoren?
    ```
  import dis

  dis.dis(lambda x: x+1)
  
  3           0 RESUME                   0
              2 LOAD_FAST                0 (x)
              4 LOAD_CONST               1 (1)
              6 BINARY_OP                0 (+)
             10 RETURN_VALUE
    ```
    ````
  from numba import njit
  @njit
   def f(x):
       return np.log(x**2 + 1e-6)
   
   f.inspect_asm(np.float64)
    ````
- BIGTREE
  A complete tree, feeding bottom-up on nodes that seem very userful
  like putting the genepool actually all together
  - after stagnation, change the tree architecture with sympy tricks. 
    Or after branches are imbalanced.
- "Scale"-operator, that just multiplies with a number

# Working with this Framework...
The following are bitches:
- Usub-class (-x).I am not even sure, if should be an operator.
- Round-class. There is a dummy-class, as that did not work on its own.
- Min/Max class. Those specifically due to sympy.
- ExprCondPair in Piecewise

# Simplifications/representations

- `simplify()`
  - uses all, but looks for string-length
  - factor

All the possible simplifications in simplify(), as introduced in [sympy](https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html).

Polynomial
- ``factor()``, ``expand``()
- ``collect()`` collects common powers of a term in an expression.
- ``cancel()``  rational function into  standard canonical form
- ``apart()`` performs a partial fraction decomposition

Trigonometric
- ``trigsimp()``
- ``expand()``  (``sin(x + y) -> sin(x)⋅cos(y) + sin(y)⋅cos(x)``)
- (``tan(x).rewrite(cos)``) (see special functions)

Power
- ``powsimp(force=False)``
- ``expand_power_exp()``
- ``expand_power_base()``
- ``powdenest(force=False)`` applies identity 3, from left to right.

Exponentials, logs
- ``expand_log(force=False)``
- ``logcombine(force=False)``

Sympy special functions:
- ``factorial(n)``  (ignored)
- ``binomial(n, k)``  (ignored)
- ``rewrite()``  (trigonometry)
- ``expand_func()``  (ignored)
- ``hyperexpand()``  (ignored)
- ``combsimp()``  (ignored)
- ``gammasimp()``  (ignored)
- ``list_to_frac()``  (ignored)

the string representation is very important, as it represents a 'genetic code'. 
I compare formulas as they are "in strings", like a touring-machine. So, the first
operator is also the most important one. 
However, flattening a tree also leaves the question: depth vs. width?
Should we see the trees from top down (all levels)?


## sfeh: make assumptions for inputs

- concerning range (positive, natural numbers)
- complex numbers 


### Semi-interesting

- trees with one node just very rarely?
- https://deap.readthedocs.io/en/master/api/tools.html
- continuous evolution (with mp), select from 3 and also replace 3
- Move all benchmarks, experiments, etc., to another project

## Always check for notes in code
- sfeh/asd/delete(me)/debug(me)
- debug this / debug me
- check, if every sympy-to-tree reconstruction works

## Compared to DEAP
- DEAP has non-Programming options (altering arrays). Plagih is only for GP.
- complexity node-count based, instead of depth-based only.
- trees that are not lists

## crazy ideas
- evolution+ranking of node evolutions
- GP individuals for evolution process

## Checks for when you have altered code
- Functions need to be in ops_dict (and some more)
- reconstruction of trees

Attention: If you want to write your own code, look for important developer informations in the section below!

## name ideas
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


## Python 3.9 Anaconda packages

All packages: `matplotlib pathlib sympy tensorflow pandas sympy scipy pyYAML scikit-learn scipy gym apted tikzplotlib`

`conda install -c matplotlib pathlib sympy tensorflow pandas sympy scipy pyYAML`

`pip install scikit-learn scipy gym apted tikzplotlib graphviz`


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
- Sympy evaluation and numpy-evaluation have slightly different results.
  Especially when 'Round' is oart of the calculation. NP is faster, sympy more accurate.

### The structure of the code

The plagih_gp run depends on the following modules:
- a loop, in which new generations of evolved trees are created
- trees, which are a recursive node-structure, that allow evolution


# Biographie
Große Weltveränderer stehen auf jeden Fall für Veränderung


- real vs. symbolisch  (Bill Gates vs. Mutter Theresa)
  - Namenlose Unternehmer  (Hätte auch jeder andere machen können)
  - Bill Gates
  - Musk  ("Herausragend")
- "Großer Sprung überwunden"
  - Rückschlag
  - Zero-to-Hero
  - Besonderheit
- Extrempunkte
- Extremlösungen

# Sammlung Pasteten


#######################################
# sfeh:idea check these options
#     import sympy
#     a, b = sp.symbols('a b')
#     expr = b - a**2 * a**3 + 2 + 3 * a * b**(-2)
#     [expr,
#      expr.as_expr(),
#      expr.as_poly(),
#      expr.as_base_exp(),
#      expr.as_coeff_add(),
#      expr.as_coeff_Add(),
#      expr.as_coeff_mul(),
#      expr.as_coeff_Mul(),
#      # expr.as_coeff_exponent(),
#      # expr.leadterm(),  # not working
#      # expr.subs(),
#      expr.as_coefficients_dict(),
#      expr.as_content_primitive(),
#      expr.as_dummy(),
#      expr.as_expr(),
#      expr.as_leading_term(),
#      expr.as_numer_denom(),
#      expr.as_two_terms(),
#      expr.as_independent(),
#      expr.expand(),
#      expr.factor(),
#      expr.assumptions0,
#      expr.normal(),
#      expr.nsimplify(),
#      # expr.extract_multiplicatively(),
#      # USEFUL
#      expr.atoms(),  # for leaf nodes
#      ]
###########################


# whathappened

```
WHATTPPENDED SFEH
	old: [Sign, [Square, [Mul, [cartPos], [cartVel]]]]
	sym: [Sign, [Mul, [Square, [cartPos]], [Square, [cartVel]]]]
	old: sign(((cartPos * cartVel))**2)
	sym: sign(((cartPos)**2 * (cartVel)**2))
	sym: Node(Sign, [Node(Square, [Node(Mul, [Node(Symbol, ["cartPos"]), Node(Symbol, ["cartVel"])])])])
```

```

WHATTPPENDED SFEH
	old: [Square, [Square, [PowRounded, [7.00], [cartPos]]]]
	sym: [Pow, [7.00], [MulChain, [4], [Round, [cartPos]]]]
	old: ((7)**RoundDummy(cartPos))**2)**2
	sym: (7)**((4 * RoundDummy(cartPos,1)))
	sym: Node(Square, [Node(Square, [Node(PowRounded, [Node(Number, [7]), Node(Symbol, ["cartPos"])])])])
```