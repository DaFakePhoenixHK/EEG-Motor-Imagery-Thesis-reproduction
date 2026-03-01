"""
Model registry: return TF/Keras models for the benchmark.
Uses existing models.py (EEGNet, ShallowConvNet, DeepConvNet).
"""
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

N_CLASSES = 4


def get_model(name, n_channels, n_times, n_classes=N_CLASSES, seed=None):
    """
    Return compiled Keras model.
    name: eegnetv4 | shallow | deep4 | conformer | fbcsp_lda | db_atcnet
    conformer/db_atcnet: fallback to shallow if not yet integrated.
    """
    if seed is not None:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)

    import models

    if name == "fbcsp_lda":
        return None
    if name == "eegnetv4":
        m = models.EEGNet_classifier(n_classes=n_classes, Chans=n_channels, Samples=n_times)
    elif name == "shallow":
        m = models.ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=n_times)
    elif name == "deep4":
        m = models.DeepConvNet(nb_classes=n_classes, Chans=n_channels, Samples=n_times)
    elif name == "db_atcnet":
        try:
            m = models.DB_ATCNet(n_classes=n_classes, in_chans=n_channels, in_samples=n_times)
        except Exception:
            m = models.ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=n_times)
    elif name == "conformer":
        m = models.ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=n_times)
    else:
        raise ValueError(f"Unknown model: {name}")

    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m
