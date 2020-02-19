"""
This starts the whole genetic programming.
This extra file was added to make the program start from command line. (Path problems...)
"""
from pathlib import Path
from plagih import plagih_gp
import sys
# sys.root_dir = sys.root_dir / 'plagih'
sys.path.append('plagih/')
sys.path.append('plagih/modules')
sys.path.append('mountaincar/')

root_dir = Path.cwd() / 'plagih'

plagih_gp.run(root_dir)
