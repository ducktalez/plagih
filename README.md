# Familiar Genetic Programming

Very fast GP-Framework that introduces the concept of ``familiarity'',  which is a measure of how similar a candidate 
solution is to a reference program. The algorithm was filed as a patent in Germany in 2023.

Trees are optionally unified and simplified with sympy, which is a powerful symbolic mathematics library. 
This also allows to create a merged tree representing all trees in a population.

We are currently trying to introduce a  pseudo-backpropagation-algorithm, with primary focus on the if/piecewise-operator.

Enjoy!

## Ablage/Todos
### Code Dokumentation und instructions überarbeiten

- Ich sehe momentan folgende Simplifizierung:   
    roundtrip expr: sign(Abs(361/cartVel**2 + 0.423*cartVel**2))
    grouped expr  : sign(Abs(0.423*cartVel**2 + 361/cartVel**2))
    Grundproblem ist hierbei die Umkehrung einer Multiplikation zu einer Division. Hier sollte schon mal vermerkt werden, dass die Division vielleicht nur mit bestimmten Faktoren erfolgen können sollte. 361 ist meiner Meinung nach zu groß. Gedacht war es ursprünglich mal für (0.5*a -> a/2) oder so.
    Hier sollte man die Grundarchitektur diskutieren und anpassen - Gruppierungen sollten nach idempotenz kategorisierbar sein. 
- Beim Printing könnte man die Größe der Bäume In einer Population mit einer Höhen-darstellungsfarbe wie bei einer Map anzeigen. 
- TED-Distance diff branches anzeigen lassen und als idee in zukunft behandeln. 
- Performance: GPU-evaluation? Ist dafür TensorFlow nötig oder geht das auch mit NumPy? 
  - Wenn irgendwann sowieso neuronale Netze verwendet werden sollen, muss vermutlich TensorFlow oder Keras oder irgendwas verwendet werden. 
  - Bitte entscheide welches Framework du dafür normalerweise verwenden würdest. 
  - Falls es TensorFlow ist gibt eine mittlerweile nicht mehr so verbreitete Option, erst einen Graphen zu bauen ohne ihn direkt auszuwerten, was ab Version 2 nicht mehr der Standard ist. 
- Merged-tree visualisierung
  - mit chatgpt erstellen: Weitere merge-tree Version (ohne Terminal nodes)
  - Bereits erstellte Evaluierungskombination nutzen
- self.evolve.origin_tree
- Dokumentation als .md/.pdf
- print/logging-system erneuern. sequenzielle prints, die den Fortschritt anzeigen, wären schön. Vielleicht auch mit einem Fortschrittsbalken? Der Verbosity-string könnte hier auch angepasst werden,
- Pseudo-Backpropagation durch Bäume
- In Vorbereitung für eine potenzielle Parallelisierung soll der generelle Ablauf der Evolution überdacht werden.
  - Alter `@gp.create_trees` Decorator bleibt als deprecated Wrapper für Abwärtskompatibilität erhalten.
  - Beispiel:
    ```python
    from plagih.parallel import Strategy
    gp.run_generation([
        Strategy("reproduction", rate=0.2, tournament_n=3),
        Strategy("mutation", rate=0.4, depth_goal=3, p_term=0.3),
        Strategy("random_new", rate=0.2, depths=[2, 3, 4]),
        Strategy("crossover", rate=0.2, crossover=True, tournament_n=3),
    ])
    ```
- Übersicht über Distanzmaße mit Beispielbäumen und deren Auswirkungen auf die Evolution erstellen.
- introduce a "strategy" system for evolution, that can be easily parallelized. 
  - This also allows to easily adjust the rates of different strategies, and to add new ones.
- save-evaluation: nan-handling, forcing real numbers
- discuss: allow_chain is probably not required at so many places...
- gen_create_initial -> create random pop if pop empty? with leftovers? 
  - How are missing individuals in an evolution handled usually? (try force?)
- tree -> evaluate nodes for best improvement 
- population cluster/races
- tf/vectorized implementation allows parallel evaluation of trees.
  - This required another numpy-dimensionality handling
  - [1] vs. [1, 2, 3] vs. [[1,1,2],[1,2,3],[2,3,3]]
    - [1] vs. [1, 2, 3]: Was a problem in np-evaluation for matching shapes
    - [[1,1,2],[1,2,3],[2,3,3]] might become another problem, if we want to evaluate multiple trees at once
- introduce NN in alpha-tree, at well-mutable nodes
- Terminal-Mutation: Build tree with inputs, but only change terminals
- use sympy.count_ops() to count operators
- numba.pydata.org https://www.youtube.com/watch?v=x58W9A2lnQc
- If no float-symbols found, return (1) true or (2) an operator? 
- Add a test to check how simplifying a tree with SymPy before evaluating it saves time or performance. 
- sympy exprtools abchecken
- Division-multiplicator node as non-len() chain input?
- print(sympy.parsing.sympy_parser.transformations)
- Symbol-time (for IB), choosing the time-step as input variable
- "Ban" trees, if they are too dominant
- make categorical options categorical. For mountain car, it is scalable, but a categorical options (aka a 3 nodes last layer) should be an option
- prevent the LUT of becoming too big; make a counter whenever a result is hit and delete the smallest in each cycle. Reset the numbers aswell.
- sympy factor (up/downfactor), so it adds stuff together, expand(), 
- Plan/feature: Iteratives ersetzen eines NN durch einen Baum, der das NN nachahmt. Das NN findet die großen Unterschiede zu einem GP-Baum, und der GP-Baum versucht, if-Strukturen zu finden, die das NN-Verhalten nachahmen.
- sfeh:idea sympy.nsimplify('3.333333*x+0.522', tolerance=0.1, rational=True) for
  - Terminals 
  - Even whole formulae!
  - Especially! Powers x**y
- Introduce rounding/clipping for many more functions
- adjust tournament_size to general fitness skew
- adding the pareto-trees visualized to the paretofront plot
- mutate chained operators specifically, crossover too. add summands, remove summands, .... as option.
- List of all potential inputs as layer, just multiplied with 1 or 1
- Introduce Tree-"styles", one expression can be represented in many ways
  - Raw (=as generated)
  - Isolate inputs in formulae as long as possible
  - Factorized
  - Simplified
  - create "better mutable" trees?
- Node/Number type rational?
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
- Crossover - make fix nodes insertable
- create_random should be the same for first generation and other randoms
- warning for sympifyable origin tree
- Make a plan for the following distinction: Genetic programming has an evolutionary and a fitness aspect. Some trees might have some evolutionary features that suit them better. And I want to make distinction in that way. 
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
- Für eine neue Knotenart „Non-Replace“ oder „Non-Holding“ oder „Save Evaluation“, die im Fall von „Not a Number“ oder von Evaluierungsproblemen einen Standardwert zurückgibt. Hierbei sollte auf jeden Fall unterschieden werden zwischen NRNs, die bereits beim Sympy-Prozess entstehen, und NRNs, die während der Auswertung entstehen. 
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
- BIGTREE
  A complete tree, feeding bottom-up on nodes that seem very userful
  like putting the genepool actually all together
  - after stagnation, change the tree architecture with sympy tricks. 
    Or after branches are imbalanced.

## Configuration (`.env`)

All framework defaults are managed via a `.env` file in the project root.
Copy `.env.example` to `.env` and adjust values to your needs.

### Minimal default profile

The default configuration is intentionally **minimal** — every feature is off:

| Feature | Default | `.env` Key |
|---|---|---|
| SymPy Simplification | off | `PLAGIH_SIMPLIFICATION` |
| Visualisation during runs | off | `PLAGIH_VISUALIZATION` |
| Merged population tree | off | `PLAGIH_MERGED_TREE` |
| Origin tree tracking | off | `PLAGIH_ORIGIN_TREE` |
| Look-Up Tables (LUT) | **on** | `PLAGIH_LUT_ENABLED` |
| Parallelisation | 0 (sequential) | `PLAGIH_PARALLEL` |


### Recommended profile for real runs

```dotenv
PLAGIH_LUT_ENABLED=true
PLAGIH_VISUALIZATION=true
PLAGIH_PARALLEL=4
PLAGIH_SIMPLIFICATION=true
```

### Override hierarchy

```
.env file  →  environment variables  →  code-level parameters
(lowest)                                  (highest priority)
```

Code-level overrides (e.g. `ExplainableGP.create(parallel=8)`) always win.
See `docs/ARCHITECTURE.md` § Configuration System for full details.

## Systemanforderungen, Laufzeit und KPI-Richtwerte

Diese Sektion ist ein **erster Startpunkt** für grobe Systemanforderungen und
Laufzeit-/Speichererwartungen. Die Werte sind **keine Garantien**, sondern
gemessene Richtwerte für die aktuelle Implementierung der Parallelisierung.

**Messbasis (Stand: 2026-03-16):**
- Windows
- 8 physische Kerne / 16 Threads
- aktuelle Parallelisierung mit
  - Pre-selection
  - Shared Memory für `df_train`
  - chunked batching
- Details: `docs/PARALLEL_BENCHMARK_DIAGNOSIS.md`
- Ressourcenprofiling: `plagih/test/benchmarks/bench_parallel_resources.py`

### Gemessene Richtwerte für die aktuelle Parallelisierung

Die folgenden Werte stammen aus den aktuellen Benchmark-Outputs und dienen als
Orientierung, wie stark `pop_size`, Workerzahl und Parallelmodus Zeit und RAM
beeinflussen.

| Population | Worker | Steady-State pro Generation | Init-Zeit | Peak RAM |
|---|---:|---:|---:|---:|
| `1000` | `0` | ~`4.0 s` | ~`4.3 s` | ~`161 MB` |
| `1000` | `4` | ~`1.8 s` | ~`4.7 s` | ~`750 MB` |
| `1000` | `8` | ~`1.3 s` | ~`4.5 s` | ~`1.29 GB` |
| `10000` | `0` | ~`41.6 s` | ~`44.4 s` | ~`290 MB` |
| `10000` | `4` | ~`17.1 s` | ~`44.7 s` | ~`941 MB` |
| `10000` | `8` | ~`11.9 s` | ~`45.1 s` | ~`1.49 GB` |

**Wichtige Beobachtungen:**
- Die untenstehenden Init-Zeiten stammen aus einem **historischen Messstand vor
  der Vereinheitlichung von Generation 0 mit dem Task-Runner**. `gen_create_initial()`
  nutzt inzwischen denselben deklarativen Runner wie normale Generationen;
  die Tabelle dient daher als Baseline, nicht als aktuelle Architekturbehauptung.
- Für diese aktuelle Architektur war auf dem Benchmark-System **`8` Worker** am
  schnellsten.
- Der große RAM-Sprung kommt vor allem durch die **Worker-Prozesse**, nicht nur
  durch die Population selbst.

### Einflussmatrix der Laufparameter

| Parameter | Einfluss auf Zeit | Einfluss auf RAM | Einfluss auf Parallelisierung | Kommentar |
|---|---|---|---|---|
| `pop_size` | sehr hoch | mittel bis hoch | positiv | Mehr Arbeit pro Generation amortisiert Parallel-Overhead besser, erhöht aber Laufzeit und RAM deutlich. |
| `gen_end` / Anzahl Generationen | linear hoch | niedrig bis mittel | neutral | Gesamtlaufzeit wächst fast linear mit der Zahl der Generationen. |
| Workerzahl (`parallel`) | niedrig bis sehr positiv | sehr hoch | direkt | Mehr Worker beschleunigen den Lauf, erhöhen aber Worker-RAM stark. |
| Batchgröße / Chunking | hoch | niedrig | sehr hoch | Zu kleine oder zu große Batches kosten Performance; aktuell liegt der Sweet Spot grob bei `32..128` Tasks pro Batch. |
| Mathematische Komplexität (mehr Operatoren, tiefere Bäume) | hoch | mittel | negativ | Mehr SymPy-/NumPy-Arbeit pro Tree; komplexere Populationen verlangsamen sowohl sequential als auch parallel. |
| Logik-/`Ifte`-/`Piecewise`-Anteil | hoch | mittel | potenziell negativ | Kann SymPy stark belasten; pathologische Fälle werden inzwischen abgefangen, bleiben aber teuer. |
| `nodes_max`, `depth_max` | hoch | mittel | gemischt | Erlauben größere Ausdrucksbäume; steigern Suchraum, Kosten und Risiko teurer SymPy-Pfade. |
| Initialpopulation | sehr hoch | mittel | gemischt | Historisch war dies ein sequentieller Fixkostenblock; nach der Runner-Vereinheitlichung sollte dieser Pfad separat neu vermessen werden. |
| `enable_analysis=True` | hoch | niedrig bis mittel | negativ | Zusätzliche IO/Plots/Rendering verfälschen Benchmarkzeiten. |
| Debug-/Detail-Prints | mittel bis hoch | niedrig | negativ | Teure Ausdrucksrepräsentationen dürfen nicht in Hot Paths aktiviert werden. |

### Grobe Presets für die Praxis

| Ziel | Empfehlung |
|---|---|
| Wenig RAM, robustes Arbeiten | `parallel=4` als vorsichtiger Startwert |
| Maximaler Durchsatz auf 8 physischen Kernen | `parallel=8` |
| Kleine Populationen / kurze Testläufe | erst `parallel=0` oder `parallel=2` prüfen |
| Große Populationen (`>=10000`) | Parallelisierung lohnt sich deutlich mehr als bei kleinen Läufen |

### Benchmark-/Pflegehinweis

Diese Tabelle sollte **regelmäßig aktualisiert** werden, insbesondere wenn sich
eine der folgenden Grundlagen ändert:

- Batching-/Chunking-Strategie in `plagih/parallel.py`
- Worker-Init / Shared-Memory-Verhalten
- SymPy-Handling / Tree-Komplexität
- Standard-Operatorenset oder Evolutionsparameter
- Benchmark-Hardware oder Betriebssystem

Empfohlene Update-Quellen:
- `plagih/test/benchmarks/bench_diagnose_full.py`
- `plagih/test/benchmarks/bench_parallel_resources.py`
- `docs/PARALLEL_BENCHMARK_DIAGNOSIS.md`

**Offene Aufgabe:** Die Parallelisierung hat vermutlich noch weiteres Potenzial.
Insbesondere Worker-RAM, Pre-selection und die **neu vereinheitlichte Generation-0-
Orchestrierung** sollten weiter gemessen und diese Tabelle anschließend nachgezogen
werden.

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

- !! (not gp-related) Database for LLM (for cross-available information)
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

All packages: `matplotlib pathlib sympy tensorflow pandas sympy scipy pyYAML scikit-learn scipy gym tikzplotlib`

`conda install -c matplotlib pathlib sympy tensorflow pandas sympy scipy pyYAML`

`pip install scikit-learn scipy gym tikzplotlib graphviz`


non-conda packages:
(none – apted has been replaced by an intrinsic Zhang-Shasha implementation)

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