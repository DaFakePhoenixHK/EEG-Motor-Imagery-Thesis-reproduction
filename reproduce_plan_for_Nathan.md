# Reproduction-Only Benchmark Plan 

**Timestamp: 27 Feb 2026**

This task brief is **ONLY** for reproducing **other people’s baselines/models** on **BCI Competition IV‑2a (4‑class motor imagery)**.  
It explicitly separates **strict inductive protocols** from **Streaming Test-Time Adaptation (TTA)** so that online, causal, unsupervised adaptation (e.g., causal EA) is *not accidentally disallowed*.

## 0) Fixed Dataset & Input Definitions (must be identical for all models)

### 0.1 Dataset
- Dataset: **BCI Competition IV‑2a**
- Subjects: **A01–A09**
- Classes: **4** (Left Hand / Right Hand / Feet / Tongue)
- Sessions: **2** (different days; session1 and session2)
- Sampling rate: **250 Hz**
- Dataset facts to keep in documentation: **6 runs per session, 48 trials per run (12 per class), 288 trials per session**.

### 0.2 Channel configurations (two lines, both required)
**(A) 8‑channel subset**
- Use exactly the 8 channels provided.

**(B) 22‑channel full EEG**
- Use all **22 EEG channels** (exclude EOG).

---

## 1) Models to Reproduce (others’ work only)

### 1.1 Traditional baseline (required)
1) **FBCSP + LDA (multi-class)**
- Must document multi-class strategy: **One-vs-Rest** or **Pairwise**.
- CSP filters must be fit **only** on the training data for the current split.

### 1.2 Deep CNN baselines (required)
Use standard implementations (preferably **Braindecode**) for reproducibility:
2) **EEGNetv4**  
3) **ShallowFBCSPNet** (or ShallowConvNet)  
4) **Deep4Net (DeepConvNet/Deep4)** *(optional if compute is tight; at least run Protocol W/L on 8ch)*

### 1.3 Transformer/Conformer baseline (required)
5) **EEGConformer**
- Prefer Braindecode `EEGConformer` for consistent parameterisation (`n_chans`, `n_times`) and easier complexity measurement.
- Record the original repo’s *reported* IV‑2a “hold‑out” accuracy as **reference only** (not protocol-equivalent).

---

## 2) Protocol Families (must be implemented and reported separately)

**Important:** Do **NOT** mix inductive and TTA results in the same “SOTA” comparison table. They are different evaluation rules.

### 2.1 Protocol W — Within-subject cross-session (strict inductive, no adaptation)
For each subject:
- Train: **session1**
- Test: **session2**
- Test-time: **pure inference only** (no updates from session2 statistics)

### 2.2 Protocol L — Strict inductive LOSO cross-subject + cross-session (no target data)
For target subject `i`:
- Train: all other subjects `j≠i` **session1**
- Validation: split from the training pool only (source subjects session1)
- Test: target subject `i` **session2**
- **No target-subject data (labelled or unlabelled) may be used** during train/val.

### 2.3 Protocol F — Few-shot labelled calibration (per-class K-shot)
For target subject `i`:
- Train: all other subjects `j≠i` session1
- Calibration (labelled): target subject `i` session1, **K trials per class**
  - Mandatory K grid: **K ∈ {1, 5, 10, 20}**
  - Also keep legacy points if needed: K≈7 and K≈18 (from prior frac-based experiments), but main reporting must be {1,5,10,20}.
- Test: target subject `i` session2 (all trials)


### 2.4 Protocol TTA — Streaming Test-Time Adaptation (Causal, Unsupervised)
Rules:
- Test: session2 is processed **in strict chronological order**.
- Allowed: **unsupervised** updates using *only current/past* test samples (no future peeking), e.g.:
  - **Causal EA** updates of covariance / whitening statistics
  - **Causal BN** running statistics updates (if used)
- Not allowed:
  - Using **labels** from session2
  - Using future test samples to compute current statistics (no look-ahead)
- Output: trial-level metrics as usual.

**Fairness requirement:** If a TTA module (e.g., causal EA) is allowed for one method, the same TTA module must be allowed for baseline models in Protocol TTA runs (e.g., “apply causal EA at the input, then feed into EEGNet/Shallow/Conformer”).

---

## 3) Leakage-Safety Rules (protocol-specific)

### 3.1 For Protocol W/L/F (strict inductive family)
- Any fitted transform (standardisation, whitening, EA, CSP filters, etc.) must be fit on **training data only** within the split.
- No using session2/test to estimate statistics.

### 3.2 For Protocol TTA (streaming adaptation family)
- Test-time updates are allowed **only if causal and unsupervised**.
- Must document the exact update rule (e.g., EMA rate, windowing, warm-up).
- Must guarantee **no future data peeking**.

---

## 4) Exact Experiment Matrix (channel × protocol × model)

### 4.1 8-channel (must fully cover)
Run the following for **each** model:
- **Protocol W**: FBCSP+LDA, EEGNetv4, ShallowFBCSPNet, (Deep4Net if included), EEGConformer
- **Protocol L**: same as Protocol W
- **Protocol F (K={1,5,10,20})**: at minimum EEGNetv4, ShallowFBCSPNet, EEGConformer
- **Protocol TTA**: at minimum EEGNetv4, ShallowFBCSPNet, EEGConformer (FBCSP optional)

### 4.2 22-channel (strictly minimal; reference upper-bound only)
Do **NOT** run the full 22ch matrix.

Run only:
- **Protocol L**: ShallowFBCSPNet + EEGConformer
- **Protocol F (K={1,5,10,20})**: ShallowFBCSPNet + EEGConformer
- **Protocol TTA**: optional, EEGConformer only (if time permits)

---

## 5) Training & Reporting Rules (deep models)

### 5.1 Seeds / repeats
- Use **5 seeds**: `{0,1,2,3,4}` for each deep model configuration.
- Report mean ± std over seeds (and subject-wise).

### 5.2 Early stopping / validation
- Validation split must come only from training pool per protocol.
- No test peeking for tuning.

### 5.3 Metrics (must output)
Per subject:
- **trialAcc**
- **macro-F1**
- **Kappa**
- **Information Transfer Rate**
Aggregates:
- mean ± std across subjects
- **median and IQR** across subjects
Recommended:
- Save confusion_matrix array/CSV for each subject to allow per-class failure analysis.

---

## 6) Complexity / Resource Metrics (ALL models; computed on PC)

Compute these for every reproduced model in both 8ch and 22ch settings using the same fixed input window (`n_times`) and batch=1:

1) **#Parameters**  
2) **MACs (or FLOPs)** — document the convention  
3) **Peak activation memory estimate (bytes)**  
4) **Host (PC) inference latency (ms/window)** — fixed CPU settings; log environment

Deliverables:
- `complexity_8ch.csv`
- `complexity_22ch.csv`

Required columns:
- `model, n_chans, n_times, params, macs_or_flops, peak_act_mem_bytes, pc_latency_ms, notes`

---

## 8) Deliverables (files, structure, and acceptance)

### 8.1 Folder structure (required)
- `results/bci2a/accuracy/{8ch|22ch}/{W|L|F|TTA}/{model}/seed_{s}/`
  - `subjectwise.csv` (one row per subject)
  - `summary.csv` (aggregated)
  - `confusion_{subject}.csv` (recommended)
- `results/bci2a/complexity/`
  - `complexity_8ch.csv`
  - `complexity_22ch.csv`

### 8.2 Global rollups (required)
- `results_summary_acc.csv`  
  Columns: `channels, protocol, model, seed, mean_trialAcc, std_trialAcc, median_trialAcc, iqr_trialAcc, mean_macroF1, std_macroF1, ...`
- `results_subjectwise.csv`  
  Columns: `subject, channels, protocol, model, seed, trialAcc, macroF1, ...`

