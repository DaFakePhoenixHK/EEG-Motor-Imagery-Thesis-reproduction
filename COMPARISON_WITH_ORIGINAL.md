# Comparison: Our Code vs Original EEG-ATCNet (GitHub)

**Original repo:** https://github.com/Altaheri/EEG-ATCNet  
**Purpose:** Find differences that could explain worse accuracy (especially preprocessing).

---

## 1. Preprocessing – CRITICAL DIFFERENCE

### Original (GitHub) `get_data()` for BCI2a

- Path: `path = path + 's{:}/'.format(subject+1)` (no flat-folder; expects `s1/`, `s2/`, ...).
- Calls `load_BCI2a_data(path, subject+1, True)` → **AxxT** (session 1) for train.
- Calls `load_BCI2a_data(path, subject+1, False)` → **AxxE** (session 2) for test.
- **Standardization:** `standardize_data(X_train, X_test, N_ch)`:
  - Fits `StandardScaler` on **full X_train** (entire session 1, AxxT).
  - Transforms **both** X_train and X_test with that scaler.
- Then in `train()`, they do `train_test_split(X_train, y_train_onehot, test_size=0.2)` on the **already standardized** session-1 data. So train and val are both standardized using the **same** scaler fit on **100% of session 1**.

### Our version

- **Path:** We added flat-folder support: if `A01T.mat` exists in `path`, we use `path` directly; else `path + 's{:}/'.format(subject+1)`. So we support both layouts. **Compatible.**
- **AxxT / AxxE:** Same: AxxT for train, AxxE for test. **Compatible.**
- **Standardization (difference):**
  - We load with `isStandard=False`, then split session 1 into train/val (80/20), then:
  - `standardize_fit_train_transform_train_val_test()`: fit scaler on **80% train only**, then transform train, val, and test.
- So we use a **different** normalization: scaler fit on 80% of session 1 instead of 100%. Val and test are then scaled with this 80%-based scaler. The original uses 100% of session 1 for the scaler. This can change scaling and may explain different (e.g. worse) accuracy.

### Original `load_BCI2a_data()`

```python
data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
class_return[NO_valid_trial] = int(a_y[trial])
```

- Uses `int(a_trial[trial])` and `int(a_y[trial])` directly. With some NumPy versions, `a_trial[trial]` is 0-d and `int()` raises. We use `.item()` for compatibility; numerically it should be the same.

---

## 2. Training loop – Original vs ours

### Original

- `get_data(data_path, sub, dataset, LOSO=..., isStandard=True)` → standardized X_train (session 1), X_test (session 2).
- `train_test_split(X_train, y_train_onehot, test_size=0.2, random_state=42)` → 80% train, 20% val.
- Model: `getModel(...)` with **no** GaussianNoise, **no** extra dropout (e.g. 0.3 for ATCNet).
- Loss: `CategoricalCrossentropy(from_logits=from_logits)` — **no label_smoothing**.
- Callbacks: ModelCheckpoint (val_loss), ReduceLROnPlateau. **No** EarlyStopping in the default run.
- Trains 500 epochs, loads best weights by val_loss, evaluates on val for logging; test is in `test()`.

### Ours (run_fixed_subject_atcnet)

- We fit scaler on **train only** (80% of session 1) then transform train/val/test → **different from original.**
- We add **GaussianNoise(0.2)** at input and **higher dropout** (0.45) for single-subject.
- We use **label_smoothing=0.1** in CategoricalCrossentropy.
- So we changed: standardization, regularization (noise + dropout + label smoothing). Any of these can hurt or help; for **matching paper numbers** we should match the original.

---

## 3. Summary of differences that can affect accuracy

| Item | Original (GitHub) | Our version |
|------|-------------------|------------|
| **Scaler fit** | On **full session 1** (AxxT), then transform train/val/test | On **80% of session 1** only, then transform train/val/test |
| **Input layer** | No GaussianNoise | GaussianNoise(0.2) |
| **ATCNet dropout** | 0.3 (eegn, tcn) | 0.45 (single-subject) |
| **Loss** | CategoricalCrossentropy() | CategoricalCrossentropy(label_smoothing=0.1) |
| **Data path** | `path + 's{:}/'.format(subject+1)` | Same + flat-folder support |
| **load_BCI2a indexing** | int(a_trial[trial]) | .item() for compatibility |

---

## 4. Recommendation

To match the **reported accuracy** (e.g. ~81% test on BCI2a) and your cosupervisor’s expectation:

1. **Preprocessing:** Use the **same standardization as the original**: fit the scaler on **full session 1** (AxxT), then transform train, val, and test. So either:
   - Call `get_data(..., isStandard=True)` and then do `train_test_split` on the returned X_train/y (no extra “fit on train only” step), or
   - Keep loading with isStandard=False, then fit one scaler on **full** X_train (before split), then transform train, val, and test.
2. **Model and training:** For a “paper reproduction” run, remove GaussianNoise, use dropout=0.3, and no label_smoothing so the setup matches the original.

---

## 5. Changes applied (to match original)

- **Fixed-subject pipeline** now:
  - Uses `get_data(..., isStandard=True)` so the scaler is fit on **full session 1** (AxxT) and both train and test (AxxE) are transformed the same way as in the original repo. Then we split the standardized session-1 data into train/val (80/20).
  - Uses the **original model**: no GaussianNoise layer, dropout=0.3 (eegn and tcn), and **no label_smoothing** in the loss.

Re-run fixed-subject training (e.g. `python train_cli.py --mode fixed --subject 1`) to compare with the reported ~81% test accuracy. Cross-subject was left as-is since the original repo only does per-subject training.
