"""
Configuration for reproduction benchmark (BCI IV-2a).
"""
from pathlib import Path

# Paths - adjust if needed
BENCH_DIR = Path(__file__).resolve().parent
PARENT_DIR = BENCH_DIR.parent
DEFAULT_BCI2A_PATH = PARENT_DIR.parent / "files2" / "Four class motor imagery (001-2014)"
DEFAULT_RESULTS_DIR = BENCH_DIR / "results"

# BCI IV-2a constants
N_SUBJECTS = 9
N_CLASSES = 4
FS = 250
# Original epoch: 1.5–6 s at 250 Hz -> 1125 samples
IN_SAMPLES = 1125
N_CHANNELS_FULL = 22

# 8-channel subset: motor cortex (indices 0-based from BCI IV-2a 22ch)
# Typical order: Fz, FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
# Using FC1, FC2, FCz, C3, Cz, C4, CP1, CP2 (indices 2,4,3,6,8,9,10,11) for motor imagery
EIGHT_CH_INDICES = [2, 3, 4, 6, 8, 9, 10, 11]

# Protocol F: K per class grid
K_PER_CLASS_GRID = [1, 5, 10, 20]

# Seeds
SEEDS = [0, 1, 2, 3, 4]

# Model names (plan + DB-ATCNet)
MODELS = ["eegnetv4", "shallow", "deep4", "conformer", "fbcsp_lda", "db_atcnet"]

# Experiment matrix: (channels) -> (protocol) -> [models]
# 8ch: full coverage per plan 4.1 + db_atcnet (user addition)
MATRIX_8CH = {
    "W": ["fbcsp_lda", "eegnetv4", "shallow", "deep4", "conformer", "db_atcnet"],
    "L": ["fbcsp_lda", "eegnetv4", "shallow", "deep4", "conformer", "db_atcnet"],
    "F": ["eegnetv4", "shallow", "conformer", "db_atcnet"],
    "TTA": ["eegnetv4", "shallow", "conformer", "db_atcnet"],
}
# 22ch: minimal per plan 4.2 (no W)
MATRIX_22CH = {
    "L": ["shallow", "conformer", "db_atcnet"],
    "F": ["shallow", "conformer", "db_atcnet"],
    "TTA": ["conformer"],
}

# Training defaults (MD: validation from training pool; no epochs specified -> use 500 + early stop)
DEFAULT_EPOCHS = 500
EARLY_STOP_START_EPOCH = 100
EARLY_STOP_PATIENCE = 80
