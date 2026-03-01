"""
Load BCI IV-2a data for reproduction benchmark (8ch or 22ch).
"""
import os
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from config import N_CHANNELS_FULL, IN_SAMPLES, EIGHT_CH_INDICES


def load_bci2a_raw(data_path, subject_id, session_train, n_channels=22):
    """
    Load raw BCI IV-2a data for one subject.
    subject_id: 1-9
    session_train: True = session1 (AxxT), False = session2 (AxxE)
    Returns: X (N, C, T), y (N,) int 0-3
    """
    from preprocess import load_BCI2a_data
    path = str(data_path).replace("\\", "/").rstrip("/") + "/"
    if not os.path.exists(os.path.join(path, f"A0{subject_id}T.mat")):
        path = os.path.join(path, f"s{subject_id}") + "/"
    X, y = load_BCI2a_data(path, subject_id, session_train)
    # X: (N, 22, 1125), y: (N,)
    if n_channels == 8:
        X = X[:, EIGHT_CH_INDICES, :]
    return X.astype(np.float32), y.astype(np.int32)


def load_all_subjects(data_path, n_channels=22):
    """
    Load all 9 subjects, session1 (train) and session2 (test).
    Returns: list of (X_train, y_train, X_test, y_test) per subject.
    """
    out = []
    for sub_id in range(1, 10):
        X_tr, y_tr = load_bci2a_raw(data_path, sub_id, True, n_channels)
        X_te, y_te = load_bci2a_raw(data_path, sub_id, False, n_channels)
        out.append((X_tr, y_tr, X_te, y_te))
    return out


def to_4d(X, n_channels):
    """Convert (N, C, T) -> (N, 1, C, T) for TF models."""
    return X[:, np.newaxis, :, :]


def standardize_fit_apply(X_train, X_val, X_test, n_channels):
    """Fit StandardScaler on train, apply to train/val/test. In-place on X_train, X_val, X_test."""
    from sklearn.preprocessing import StandardScaler
    scalers = []
    for j in range(n_channels):
        s = StandardScaler()
        s.fit(X_train[:, 0, j, :])
        scalers.append(s)
    for X in (X_train, X_val, X_test):
        for j in range(n_channels):
            X[:, 0, j, :] = scalers[j].transform(X[:, 0, j, :])
    return scalers


def get_subject_data(subject_list, data_path, n_channels=22):
    """
    Load data for specified subjects. subject_list: list of 1-9.
    Returns dict: sub_id -> (X_tr, y_tr, X_te, y_te) where X are (N,1,C,T).
    """
    from sklearn.model_selection import train_test_split
    out = {}
    for sub_id in subject_list:
        X_tr, y_tr = load_bci2a_raw(data_path, sub_id, True, n_channels)
        X_te, y_te = load_bci2a_raw(data_path, sub_id, False, n_channels)
        X_tr = to_4d(X_tr, X_tr.shape[1])
        X_te = to_4d(X_te, X_te.shape[1])
        out[sub_id] = (X_tr, y_tr, X_te, y_te)
    return out
