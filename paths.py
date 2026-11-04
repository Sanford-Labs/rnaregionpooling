"""Where the code, the working files, the data and the outputs are.

CODE is this repository. WORK holds what the scripts read and write: the per-model embeddings, the fits, the element
tables and the perturbation scores (`$REGIONS_WORK`, default `work/` in this repository). DATA holds the mRNABench
tables `fetch_mrnabench.py` writes (`$MRNA_BENCH_DATA`, default WORK/data), MODEL_DIR the model weights that are not
fetched from Hugging Face at run time (`$REGIONS_MODELS`, default WORK/models), and PAPER the directory the figures
and LaTeX tables are written into (`$REGIONS_PAPER`, default WORK/paper).
"""
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("REGIONS_WORK", CODE / "work"))
DATA = Path(os.environ.get("MRNA_BENCH_DATA", WORK / "data"))
MODEL_DIR = Path(os.environ.get("REGIONS_MODELS", WORK / "models"))
PAPER = Path(os.environ.get("REGIONS_PAPER", WORK / "paper"))
