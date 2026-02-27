# Using Your Own Preprocessed Data

You can skip the built-in preprocessing and supply your own train/val/test arrays.

## Option 1: Single `.npz` file

Create a folder and put inside it a file named **`preprocessed.npz`** with these NumPy arrays (keys):

| Key | Shape | Description |
|-----|--------|-------------|
| `X_train` | (N, 1, n_channels, n_samples) | Training data, float |
| `y_train_onehot` | (N, n_classes) | One-hot labels for training |
| `X_val` | (M, 1, n_channels, n_samples) | Validation data |
| `y_val_onehot` | (M, n_classes) | One-hot labels for validation |
| `X_test` | (K, 1, n_channels, n_samples) | Test data (e.g. session 2) |
| `y_test_onehot` | (K, n_classes) | One-hot labels for test |

For BCI2a-style: `n_channels=22`, `n_samples=1125`, `n_classes=4`.

**Example (Python):**
```python
import numpy as np
# After you have X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot
np.savez_compressed('my_folder/preprocessed.npz',
    X_train=X_train, y_train_onehot=y_train_onehot,
    X_val=X_val, y_val_onehot=y_val_onehot,
    X_test=X_test, y_test_onehot=y_test_onehot)
```

## Option 2: Separate `.npy` files

In the folder, put these files:

- `X_train.npy`, `y_train_onehot.npy`
- `X_val.npy`, `y_val_onehot.npy`
- `X_test.npy`, `y_test_onehot.npy`

Shapes as above.

## How to use

- **CLI:** `python train_cli.py --mode fixed --subject 1 --preprocessed_dir /path/to/folder`
- **GUI:** In "Preprocessed data folder (optional)", enter or browse to the folder. Leave empty to use raw data and run built-in preprocessing.

If the folder or required keys are missing, training falls back to internal preprocessing using the raw data path.
