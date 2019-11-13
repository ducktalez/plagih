# PLAGIH (I still need a name for this :P)

PLAGIH stands for PLAusible Genetic Improvements to Heuristics. The name will probably be changed soon.
This is a Project resulting from my Masters Thesis (with yet unknown name). 
Its aim is to provide a framework where a heuristic is improved by genetic programming towards a machine learned approach. 
The new code shall both perform significantly better and also be an explainable, 
plausible addition to the original program which the developer does understand.

## Including

- Karoo-based gp library
- mountain car tests


**Differences to Karoo:** (To reduce confusion a little if you look into my code)
- User adds his own program (origin-tree), which is the origin for all other trees
- Tree has new parameter `tree.node_modify`, which lets the user specify if this node shall stay the same over time (aka, lets him specify program code that he really likes)
- midway I noticed I could have solved some stuff differently
- probably a lot of other stuff will also be confusing, I did not document very much
- Also, **A LOT** of TODOs