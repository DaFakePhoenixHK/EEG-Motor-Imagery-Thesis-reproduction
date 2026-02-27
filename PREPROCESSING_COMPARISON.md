# Preprocessing Comparison: Cosupervisor vs Original Source Code

## Summary

**Cosupervisor's preprocessing** produces **~50% accuracy** (random chance for 4 classes = 25%), while **original preprocessing** achieves **70-85%+**. This document explains the key differences.

---

## 1. Sampling Rate (Downsampling)

| Aspect | Cosupervisor | Original Source Code |
|--------|-------------|---------------------|
| **Sampling Rate** | **128 Hz** (downsampled from 250 Hz) | **250 Hz** (no downsampling) |
| **Samples per trial** | **448** (0.5-4 s × 128 Hz = 3.5 s × 128 = 448) | **1125** (1.5-6 s × 250 Hz = 4.5 s × 250 = 1125) |
| **Temporal resolution** | Lower (less fine-grained) | Higher (more detail) |

**Impact:** Downsampling to 128 Hz loses high-frequency detail and reduces temporal resolution. The model sees **2.5× fewer samples per trial** (448 vs 1125), which may hurt learning temporal patterns.

---

## 2. Motor Imagery Time Window

| Aspect | Cosupervisor | Original Source Code |
|--------|-------------|---------------------|
| **Epoch window** | **0.5 s to 4.0 s** (relative to cue/trial start) | **1.5 s to 6.0 s** (relative to trial start) |
| **Duration** | **3.5 seconds** | **4.5 seconds** |
| **Start offset** | 0.5 s after cue | 1.5 s after trial start |
| **End offset** | 4.0 s after cue | 6.0 s after trial start |

**Impact:** 
- **Cosupervisor:** Starts earlier (0.5 s) and ends earlier (4.0 s). This may include more baseline/preparation activity and less of the actual MI response.
- **Original:** Starts later (1.5 s) to skip initial preparation, captures more of the sustained MI period (up to 6.0 s).

**Note:** The original code extracts a **7-second window** from the trial, then crops to **1.5-6.0 s** (the MI period). The cosupervisor likely extracts **0.5-4.0 s** directly.

---

## 3. Bandpass Filtering

| Aspect | Cosupervisor | Original Source Code |
|--------|-------------|---------------------|
| **Bandpass filter** | **4-40 Hz** (order 4) | **None** (raw EEG segments used) |
| **Frequency range** | Removes < 4 Hz (DC, drift) and > 40 Hz (muscle artifacts, noise) | Keeps all frequencies (0-125 Hz at 250 Hz, or 0-64 Hz at 128 Hz) |

**Impact:**
- **Cosupervisor:** Filters out low-frequency drift and high-frequency noise. This is **standard practice** in EEG preprocessing and should help.
- **Original:** Uses raw, unfiltered data. This may include more noise but preserves all frequency content.

**Why this matters:** The original model was trained on **unfiltered** data. If you train on **filtered** data (cosupervisor), the model may learn different frequency patterns. However, filtering typically **improves** performance, so this alone shouldn't cause 50% accuracy.

---

## 4. Standardization

| Aspect | Cosupervisor | Original Source Code |
|--------|-------------|---------------------|
| **Standardization** | **Not applied** (data saved as-is) | **Per-channel StandardScaler** (fit on train, transform train/val/test) |
| **When applied** | None (we add it in our code) | After train/val split (fit on train only) |

**Impact:** 
- **Cosupervisor data:** Raw voltage values (likely in microvolts). We now apply standardization in our code (fit on train, transform all).
- **Original:** Standardized per channel (zero mean, unit variance). This is critical for neural networks.

**Note:** We fixed this by adding standardization to the cosupervisor data path, so this shouldn't be the issue anymore.

---

## 5. Data Shape and Model Input

| Aspect | Cosupervisor | Original Source Code |
|--------|-------------|---------------------|
| **Input shape** | **(N, 1, 22, 448)** | **(N, 1, 22, 1125)** |
| **Model expects** | Must set `dataset_conf['in_samples'] = 448` | `dataset_conf['in_samples'] = 1125` (default) |

**Impact:** The model architecture (ATCNet) was designed for **1125 samples**. When we use **448 samples**, the temporal convolution layers see much shorter sequences. This is a **major architectural mismatch**.

---

## 6. Trial Extraction Method

| Aspect | Cosupervisor | Original Source Code |
|--------|-------------|---------------------|
| **Method** | Uses `make_epochs()` with cue-based extraction | Extracts 7-second window, then crops to 1.5-6.0 s |
| **Cue handling** | May apply `cue_offset_s` (e.g., +2.0 s if trial markers are at trial start) | Uses trial markers directly as start |

**Impact:** Different alignment to the cue event could shift the MI period.

---

## 7. Channel Selection

| Aspect | Cosupervisor | Original Source Code |
|--------|-------------|---------------------|
| **Channels** | Uses `channel_subset` from config (likely all 22) | Uses all 22 EEG channels |
| **Channel order** | Uses `STANDARD_CHANNELS` order | Uses channels as stored in .mat (may differ) |

**Impact:** If channel order differs, spatial patterns learned by the model won't match.

---

## Root Cause Analysis

The **most likely causes** of poor performance (<50%):

1. **Temporal resolution mismatch (CRITICAL):**
   - Model trained on **1125 samples** (4.5 s at 250 Hz)
   - Cosupervisor data has **448 samples** (3.5 s at 128 Hz)
   - The model's temporal convolutions expect **2.5× longer sequences**
   - **Solution:** Retrain the model with `in_samples=448` OR use original preprocessing

2. **Time window mismatch:**
   - Cosupervisor: **0.5-4.0 s** (may include more baseline, less MI)
   - Original: **1.5-6.0 s** (focused MI period)
   - **Solution:** Use original window (1.5-6.0 s) or verify cosupervisor's window captures the MI response

3. **Downsampling loss:**
   - 128 Hz loses high-frequency detail compared to 250 Hz
   - May miss rapid temporal dynamics
   - **Solution:** Use 250 Hz (original) or verify 128 Hz is sufficient

4. **Filtering mismatch:**
   - Original model trained on **unfiltered** data
   - Cosupervisor uses **filtered** data (4-40 Hz)
   - Model may have learned to use frequencies outside 4-40 Hz
   - **Solution:** Train on filtered data OR use unfiltered data

---

## Recommendations

### Option 1: Use Original Preprocessing (Verify Code Works)
```bash
python train_cli.py --mode fixed --subject 1 --data "C:\Users\User\Desktop\Thesis\files2\Four class motor imagery (001-2014)"
```
If this gives **70-85%+ accuracy**, the code is fine and the issue is the preprocessing mismatch.

### Option 2: Match Cosupervisor's Preprocessing Exactly
If you must use cosupervisor data:
- Ensure the model is built with `in_samples=448` (we do this)
- Verify the **0.5-4.0 s window** captures the MI response (check with your cosupervisor)
- Consider retraining from scratch with cosupervisor's preprocessing (don't use pretrained weights)

### Option 3: Hybrid Approach
- Use cosupervisor's **filtering** (4-40 Hz) but keep **250 Hz** sampling
- Use original **time window** (1.5-6.0 s) but apply bandpass filter
- This gives: 250 Hz, 4-40 Hz filter, 1.5-6.0 s window → **1125 samples** (matches model)

---

## Conclusion

The **primary issue** is likely the **temporal resolution mismatch** (448 vs 1125 samples) combined with the **time window difference** (0.5-4.0 s vs 1.5-6.0 s). The model architecture expects longer sequences with a different temporal structure. Even if we set `in_samples=448`, the model was likely trained on 1125-sample sequences, so the learned temporal patterns don't match.

**Next step:** Run with original preprocessing (`--data`) to confirm the code works, then decide whether to:
1. Use original preprocessing (recommended for now)
2. Retrain the model with cosupervisor's preprocessing from scratch
3. Modify cosupervisor's preprocessing to match the original (250 Hz, 1.5-6.0 s window)
