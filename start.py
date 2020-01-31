"""
Maybe this helps starting from the command line
"""
from plagih import plagih_gp
import sys
sys.path = sys.path / plagih
sys.path.append('plagih/')
sys.path.append('plagih/modules')
sys.path.append('mountaincar/')

plagih_gp.run()
