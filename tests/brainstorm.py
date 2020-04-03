import numpy as np

choose_oparray = [[[], ['sin', 'cos', '~'], ['+', '+', '+', '-', '*', '/'], []],
                  [[], [], ['<', '>', '==', '!='], []],
                  [[], ['Not', 'Not'], ['&'], []],
                  [[], [], [], []],
                  [[], [], [], ['Ifte']]]

node_choose_dict = {
    '2f': {
        0: {'observ': ['cartPos', 'cartVel'],
            'distribution': [lambda: np.random.normal(1, 2), lambda: np.random.choice([1, 2, 3])]},
        1: {'f2f': ['sin', 'abs'],
            'b2f': []},
        2: {'f2f': ['+', '+', '-'],
            'b2f': []},
        3: {'b2f2f': ['Ifte']}},
    '2b': {
        0: {'observ': [],
            'distribution': [lambda: np.random.choice([True, False])]},
        1: {'f2b': [],
            'b2b': ['Not']},
        2: {'f2b': ['<', '<=', '>'],
            'b2b': ['&', '|']},
        3: {None: []}}}

arity = 3
intype = 'f2f'
get1 = node_choose_dict.get('2f')
lst1 = list(node_choose_dict.values())
print('raw', lst1)
get2 = list(filter(None, map(lambda x: x.get(arity), lst1)))
lst2 = list(filter(None, map(lambda x: x.get(arity), lst1)))
print('arity {}\t'.format(arity), lst2)
lst3 = list(filter(None, map(lambda x: x.get(intype), lst2)))
print('intype: {}\t'.format(intype), lst3)

agent_file = """
import math
class MTC_simple0:
    def decide(self, input):
        cartPos, cartVel = input
        action = 0 if (cartVel < 0) else 2
        return max(0, min(2, int(round(action))))
agent_tupels = [('MTC_simple0', MTC_simple0())]"""

exec_agent_file = None
exec(agent_file)
for name, agntclass in agent_tupels:
    # evaluate
    # make plots
print('DECISION!!', agent.decide((1, 2)))
