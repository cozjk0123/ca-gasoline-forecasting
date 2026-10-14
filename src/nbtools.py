import nbformat as nbf

SETUP = '''import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import *
import plots; plots.setup()
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
df = load_weekly()'''


class NB:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip()))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip()))

    def write(self, path):
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, path)
