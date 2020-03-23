import inspect
import ast

try:
    import astunparse
except:
    raise ImportError

og_file = ast.parse(inspect.getsource(ast))
# get back the source code
test1 = astunparse.unparse(og_file)
# get a pretty-printed dump of the AST
test2 = astunparse.dump(ast.parse(og_file))

simon1 = ast.parse('return 1', mode='eval').body
# simon1 = ast.parse('1+2+3')
unparse1 = astunparse.dump(simon1)

print(unparse1)
