"""Useful utils
"""
from .misc import *
from .logger import *
from .visualize import *
from .eval import *

# progress bar
import os, sys
print(os.path.join(os.path.dirname(__file__), "progress"))
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar