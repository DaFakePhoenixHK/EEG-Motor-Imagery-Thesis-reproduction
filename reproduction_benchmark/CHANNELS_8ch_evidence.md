# 8-channel subset: rationale and evidence

## Why 8 channels?

The reproduction plan (Section 0.2) requires an **8-channel subset** in addition to full 22-channel EEG. The benchmark uses a **motor-cortex-focused** subset so that the 8ch condition is both comparable to literature and physiologically justified for motor imagery.

---

## Evidence for motor-cortex channels

1. **BCI Competition IV Dataset 2a (official)**  
   - 22 EEG channels, international 10–20 system (see *desc_2a.pdf*, Figure 3).  
   - Official description: https://www.bbci.de/competition/iv/ (Dataset 2a).  
   - Electrode montage is shown in the description; channel order in your data depends on the loader (GDF vs .mat).

2. **Motor imagery and sensorimotor cortex**  
   - Motor imagery modulates **sensorimotor** rhythms (mu, beta) over **FC, C, CP** regions.  
   - References:  
     - Ang et al. (2012), *Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b*, Frontiers in Neuroscience.  
     - Channel selection over motor cortex is a standard, knowledge-based approach for motor imagery BCI and is used to reduce channels while keeping task-relevant signals.

3. **Channel selection over motor cortex**  
   - "Channel selection focusing on motor cortex regions (FC, C, CP) is optimal" for motor imagery; "reduces computational complexity while optimizing accuracy" (e.g. channel selection / source localization literature).  
   - FBCSP and similar methods on BCI IV 2a typically use either all 22 channels or a subject/channel-selected subset over sensorimotor areas.

4. **C3, Cz, C4 as core MI channels (mandatory)**  
   - **C3, Cz, and C4** are the standard electrodes for motor imagery BCI: C3 (right motor cortex, left-hand MI), C4 (left motor cortex, right-hand MI), Cz (midline, feet/bilateral). Omitting C3 can reduce discriminative power.  
   - Channel selection literature (e.g. Biomedical Engineering Online 2015; IEEE papers on EEG channel selection for MI) treats C3, Cz, C4 as essential; 8-channel sets often add FCz, C1, C2, CP1, CP2.

---

## What this benchmark uses

- **Indices (0-based)**: `EIGHT_CH_INDICES = [3, 7, 8, 9, 10, 11, 14, 16]`  
  These index into the **22-channel array** produced by your loader (e.g. `preprocess.load_BCI2a_data` / BNCI .mat), using the same order as cosupervisor's `STANDARD_CHANNELS` in `bci2a.py`.
- **Electrode names (BNCI order)**: **FCz, C3, C1, Cz, C2, C4, CP1, CP2**.
- **Requirement**: The set **must include C3, Cz, and C4**—the three electrodes over primary motor cortex that are standard for decoding left-hand (C3), right-hand (C4), and feet/midline (Cz) motor imagery. The previous selection omitted C3 (it used C5, C1, Cz, C2, C4), which can hurt classification.

---

## Comparison with cosupervisor's pipeline (files2/bci2a.py)

The cosupervisor's loader uses the **same BNCI .mat** and defines the canonical 22-channel order as `STANDARD_CHANNELS` in `bci2a.py`:

```text
Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
 0    1    2    3    4    5   6   7   8   9  10  11  12   13  14  15  16  17  18  19  20  21
```

Our reproduction benchmark uses the same .mat (first 22 columns), so **the 22-channel order is the same**. Our 8-channel indices (with **C3, Cz, C4** required) map to:

| Index | Electrode (cosupervisor order) |
|-------|---------------------------------|
| 3     | FCz                             |
| 7     | **C3**                          |
| 8     | C1                              |
| 9     | **Cz**                          |
| 10    | C2                              |
| 11    | **C4**                          |
| 14    | CP1                             |
| 16    | CP2                             |

**Our 8ch set (by name):** FCz, C3, C1, Cz, C2, C4, CP1, CP2.

---

## Summary

- The **8ch set** is **FCz, C3, C1, Cz, C2, C4, CP1, CP2** (indices `[3, 7, 8, 9, 10, 11, 14, 16]` in BNCI order). It **must include C3, Cz, C4**; the remaining five (FCz, C1, C2, CP1, CP2) complete the motor/sensorimotor coverage.
