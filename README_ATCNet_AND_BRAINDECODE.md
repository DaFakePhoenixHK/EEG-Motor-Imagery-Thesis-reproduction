# EEG-ATCNet and Braindecode: What They Do and Why They Achieve Peak Results

This folder contains a **replication of the EEG-ATCNet repository** ([Altaheri/EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet)) for reference and comparison with your thesis code in `../files`. Below is a concise overview of both sources and why ATCNet reaches state-of-the-art performance.

---

## 1. EEG-ATCNet (Altaheri) — SOTA EEG motor imagery

**Repository:** https://github.com/Altaheri/EEG-ATCNet  
**Paper:** *Physics-informed attention temporal convolutional network for EEG-based motor imagery classification* (IEEE Trans. Industrial Informatics, 2023) — https://ieeexplore.ieee.org/document/9852687

### What it does

- **Goal:** Subject-specific (or LOSO) motor imagery classification on BCI Competition IV-2a (and optionally HGD, CS2R).
- **Recommended pipeline:** **Train–Val–Test** via `main_TrainValTest.py` (train/val split from session 1, test on session 2). Session 1 is split 80% train / 20% val; best model is chosen by **validation loss**; final metrics are reported on the **test** set (session 2).
- **Models:** Implements several baselines (EEGNet, EEG-TCNet, TCNet_Fusion, MBEEG_SENet, EEGNeX, DeepConvNet, ShallowConvNet) and the proposed **ATCNet**.

### ATCNet architecture (why it achieves peak results)

1. **Convolutional (CV) block**  
   EEGNet-style block: temporal conv → depthwise spatial conv → separable conv → pooling/dropout. Produces a **sequence of temporal feature maps** (not a single vector), which preserves temporal structure for the next blocks.

2. **Sliding-window augmentation**  
   The CV block output is treated as a sequence. Multiple **shifted windows** (e.g. 5) are taken along the time dimension. Each window is processed independently by the same Attention + TCN blocks. Outputs are then **averaged** (or concatenated). This acts as data augmentation and multi-view fusion, improving robustness and accuracy.

3. **Attention (AT) block**  
   Multi-head self-attention (MHA) over the temporal dimension highlights **when** in the trial the discriminative activity occurs. Options in the repo: `mha`, `mhla`, `se`, `cbam`. MHA is the default and helps the model focus on informative time segments.

4. **Temporal convolutional (TC) block**  
   A **TCN** (dilated causal convolutions, residual connections) further refines temporal features. TCN is good at long-range dependencies with a small number of parameters.

5. **Regularization**  
   L2 weight decay, max-norm constraints, and dropout (conv block and TCN) reduce overfitting on small EEG datasets.

6. **Training setup**  
   Adam, learning-rate reduction on plateau, **ModelCheckpoint on validation loss** (save best only). No early stopping in the reported 500-epoch setup; best checkpoint is loaded for validation and test.

On **BCI Competition IV-2a**, ATCNet reaches **~81.1%** test accuracy (vs ~68.7% EEGNet, ~65.4% EEG-TCNet in the same codebase). The combination of **conv → attention → TCN** plus **sliding windows** and **strong regularization** is why it achieves this peak.

---

## 2. Braindecode — EEG/MEG deep learning toolbox

**Repository:** https://github.com/braindecode/braindecode  
**Docs:** https://braindecode.org

### What it does

- **Goal:** General-purpose toolbox for decoding **EEG, MEG, ECoG** with deep learning (PyTorch).
- **Features:** Dataset loaders (including MOABB), preprocessing, **data augmentation** (e.g. frequency shift, time reversal, Mixup), and many **model implementations** (EEGNet, ShallowConvNet, DeepConvNet, **ATCNet**, EEGInception, etc.).
- **Train–val–test:** Braindecode recommends and supports a proper **train / validation / test** split and tuning on validation (see [how to train, test and tune](https://braindecode.org/stable/auto_examples/model_building/plot_how_to_train_test_and_tune.html)).

### Why it is useful for peak results

- **Reproducibility:** Same datasets and splits as in the literature.
- **Best practice:** Validation-based model selection and reporting on a held-out test set.
- **PyTorch ATCNet:** ATCNet is implemented in PyTorch in Braindecode (and in the newer [Altaheri/TCFormer](https://github.com/Altaheri/EEG-ATCNet) repo). So you can get **similar SOTA performance** in PyTorch without reimplementing from the TensorFlow ATCNet code.
- **Augmentations and training utilities:** Augmentations and training loops help stabilize training and improve generalization (similar in spirit to what you do in `../files` with Rommel-style augmentation and SOTA scripts).

---

## 3. Summary

| Aspect              | EEG-ATCNet (this replication)     | Braindecode                    |
|---------------------|------------------------------------|---------------------------------|
| Framework           | TensorFlow/Keras                   | PyTorch                         |
| Main model          | ATCNet (conv + attention + TCN)    | Many (including ATCNet)         |
| Data                | BCI2a, HGD, CS2R (in preprocess)   | MOABB, MNE, custom              |
| Why peak results    | Sliding windows + MHA + TCN + reg.  | SOTA models + train/val/test   |

**Replication in this folder:** The files here are a copy of the **TensorFlow** EEG-ATCNet code (models, attention, main_TrainValTest, preprocess) so you can run and compare ATCNet on BCI2a (or adapt paths for your data). For a **PyTorch** version that fits a thesis pipeline, using **Braindecode’s ATCNet** inside `../files` (or calling Braindecode from your scripts) is the most straightforward way to get the same architecture and similar peak results without porting TensorFlow yourself.

---

## 4. How to run this replication (files2)

- **Data:** BCI Competition IV-2a, in the structure expected by `preprocess.py` (e.g. `data_path/s1/A01T.mat`, `data_path/s1/A01E.mat` for subject 1).
- **Config:** In `main_TrainValTest.py`, set `dataset = 'BCI2a'`, `data_path` to your BCI2a folder, and `train_conf['model'] = 'ATCNet'`.
- **Train:** Call `train(dataset_conf, train_conf, results_path)` (and ensure `results_path` exists or is created).
- **Test:** Call `test(model, dataset_conf, results_path)` after training to evaluate on the test set (session 2).
- **Dependencies:** TensorFlow 2.x, numpy, scipy, scikit-learn, matplotlib; for CS2R or HGD, MNE and other deps as in the original repo.

See the original [EEG-ATCNet README](https://github.com/Altaheri/EEG-ATCNet?tab=readme-ov-file) and paper for full details and citation.
