"""plagih.trees — Node hierarchy, evolution, and GP engine.

Re-exports all public names so that ``from plagih.trees import *``
and ``from plagih.trees import Node, Add, ExplainableGP`` continue to work.
"""

from plagih.trees._evolution import *
from plagih.trees._gp_engine import *
from plagih.trees._nodes import *
