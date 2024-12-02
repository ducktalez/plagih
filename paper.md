# we
We propose unifying the architecture of GP candidates mathematically, which offers possibilities in fine-tuning of
candidates, comparing their similarity or by reducing overfitting.
The metaheuristic nature of a gp process consists of basic concepts (selection, randomness, mutation and crossover),
are knowingly adjusted in order to drive the evolution process in certain aspects.
Those improvements mostly address a problem rooted in the unstructured randomness, which requires
respectively long for some rather simple and obvious mathematically improvements (e.g. Filter-optimization, 
simplifying basic terms or weighting the use of operators). 




# 
Creating mathematical expressions with GP allows candidates to have any mathematical structure. However, equivalent candidates are then selected for simplicity, which arguably leads the final candidates to be expression``.zip''s. We consider the gene pool to thus always hold a suboptimal set of building blocks, rather suitable for a high-level optimization puzzle rather than for a fruit-bearing, creative construction process. 
It seems impossible to create everlasting rules for the intuition 

I created a gp-algorithm, that uses differentiates between the computational complexity of candidates (parsimony),   while providing 
Mathematical expressions consist of their identity (the actual computation), but are written 

###

The complexity restriction of a GP-process leads to a search of the best candidates fitting the complexity. Complexity measures may consider computational complexity, the depth or parsimony of formulae, their similarity to familiar approaches, etc., but at best apply a suitable tradeoff between all of the options. Da es schier unmöglich scheint, ein eindeutig bestes Komplexitätsmaß zu beschreiben, gibt es nur wenig enthusiasmus in der Forschung. Es scheint allerdings unumstritten, dass das Komplexitätsmaß für den Evolutionsprozess

###

Im Evolutionsprozess der GP können komplexere Kandidaten meist besser weiterentwickelt werden als ihre einfachste Version. Meist wird versucht, das Komplexitätsmaß so einzuschränken, dass der Genpool optimiert wird - dennoch nehmen wir an, dass es einen Tradeoff zwischen den Zielkandidaten und den bestgeeigneten Genpool-kandidaten liegt.
Dies ist leicht zu beweisen [].

###

We claim, that the solution candidates are not the most suitable ones for an evolution-process. Even a complexity measure tuned to create the best suitable gene-pool candidates is static and does not accreditate to dynamic evolution improvements that might be temporarily required. Instead, a high enough amount of random mutation is hoped for to sufficiently provide evolvable candidates.
However, the general process always drives to a standardized solution.

###

Often, the candidates completely switch strategy within a paretofront, showing the most efficient solution within a complexity
We assume, that 

###

Overfitting describes the problem, that models can become too specific and loose their general purpose. 
In general, this happens when models allow for very complex and fine-granular solutions and can be reduced with many 
measurements, for example early stopping.
In GP, candidates are a priori limited in complexity to reduce the bloat problem. 
However, even candidates with low complexity can overfit the data. 
These overfittings are characterized by the 
a metaheuristic approach, meaning that candidates can ``overfit'' 
However, the
hence, as there is usually a very strict general limitation to the complexity of GP-solutions, 

Computational complexity, mathematical complexity, human intuition

Many developers experience in the first years of computation, that they try to minimize the length of their written code, almost into one-liners. Only after getting back to your own code and having to reverse-engineer your thought process, you start understanding the value of readable code. Another example, in OO-programming, it often happens, that while creating the most accurate abstract classes hierarchy for the most general usability, you loose track of their actual purpose. It is overly complex, and people argue that OO-programming is dead just for that reason.

We will distinguish the following complexities:
- computational: the amount of raw computation on the machine-level (CPU, assembler, ...).
e.g.
  - length of assembler code
  - length of CPU-evaluation
  - memory consumption
- mathematical
  - number of operators and terminals
  - depth, max. amount of nesting
- explainability

The following trade-offs are further considered
- gene-pool vs. final candidates

The 

## 
The ``No free lunch''-theorem ist die Annahme, dass keine weitere optimierung möglich ist, die für alle Probleme gilt.


###

The theoretically pareto-efficient candidates are shaped, according to the complexity measure. An evolutionary process assumes, that the best candidates can be found meta-heuristically, by in

###

Multiple extensions to GP
- 

###

- find candidates with the same behaviour
- unify their structure for general analysis
- check their ...

###