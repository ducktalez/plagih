"""
#!/usr/bin/env bash
pdflatex agents_trees.tex
find . -name "visualisation/*.tex" -exec pdflatex {} \;
find . -name "visualisation/*.pdf" -exec pdftoppm {} {} -png \;
a*b * Ifte ** a*b*c *I *Ifte
"""
tpls = [[3, 5], [5, 3], [4, 4], [3, 5]]
tpls = sorted(tpls, key=lambda x:(x[0], x[0]))
print(tpls)
