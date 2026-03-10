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

# 8-channel subset: motor cortex, MUST include C3, Cz, C4 (core MI channels).
# Indices 0-based into 22ch BNCI order: FCz(3), C3(7), C1(8), Cz(9), C2(10), C4(11), CP1(14), CP2(16).
# See CHANNELS_8ch_evidence.md for rationale and STANDARD_CHANNELS mapping.
EIGHT_CH_INDICES = [3, 7, 8, 9, 10, 11, 14, 16]

# Protocol F: K per class grid
K_PER_CLASS_GRID = [1, 5, 10, 20]

# Seeds
SEEDS = [0, 1, 2, 3, 4]

# Model names (plan + DB-ATCNet)
MODELS = ["eegnetv4", "shallow", "deep4", "conformer", "fbcsp_lda", "db_atcnet"]

# Experiment matrix: (channels) -> (protocol) -> [models]
# 8ch: full coverage per plan 4.1 (W,L,F,TTA for all models) + db_atcnet
MATRIX_8CH = {
    "W": ["fbcsp_lda", "eegnetv4", "shallow", "deep4", "conformer", "db_atcnet"],
    "L": ["fbcsp_lda", "eegnetv4", "shallow", "deep4", "conformer", "db_atcnet"],
    "F": ["fbcsp_lda", "eegnetv4", "shallow", "deep4", "conformer", "db_atcnet"],
    "TTA": ["fbcsp_lda", "eegnetv4", "shallow", "deep4", "conformer", "db_atcnet"],
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
