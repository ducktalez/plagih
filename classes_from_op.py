"""
Simon's file for brainstorming new stuff
"""
import sys
from pathlib import Path
from plagih.modules.operators import op, op_what
import inspect
import re

sys.path.append(str(Path('C:/Users/Rapid/PycharmProjects/plagih')))


def make_classes():
    print('import os\nimport tensorflow as tf\nimport ast\nimport math\nfrom plagih.modules.plagih_data import *\n\n')
    print('class Plabel:')
    print('    pass\n')
    for key, v in op_what.items():
        # print(key, v)
        # key = '>='
        # fun = v['fun_label']
        # inhalt = ['arity',    'xtype',    'c-weight',    'tf',    'latex1',    'latexF',    'sym_str',    'pycode']
        #
        # for inh in inhalt:
        #     print(f'print("self.{inh} = {{v[{}]}})"')
        print('')
        classname = v['fun_class']
        if classname == 'SKIP':
            continue
        print(f"class {classname}(Plabel):\n")

        print(f"    fun_label = '{v['fun_label']}'")
        print(f"    arity = {v['arity']}")
        print(f"    xtype = '{v['xtype']}'")
        print(f"    c_weight = {v['c-weight']}")
        # vtf = inspect.getsource(v['tf'])
        # vtf = re.sub(".*tf': ", "", str(vtf))
        # vtf = re.sub(",.*': ", "", vtf)
        print(f"    tf = tf.{v['tf_name']}")
        latex1 = v['latex1']
        latex1 = latex1.replace('\\', '\\\\')
        print(f"    latex1 = '{latex1}'")
        latexF = v['latexF']
        latexF = latexF.replace('\\', '\\\\')
        print(f"    latexF = '{latexF}'")
        print(f"    sym_str = '{v['sym_str']}'")
        pycc = v['pycode']
        print(f"    pycode = '{pycc}'")

        print('')


make_classes()
