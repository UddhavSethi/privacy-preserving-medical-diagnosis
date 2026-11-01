# RSNA label ambiguity: provenance, audit and re-analysis (2026-10-26)

Scope: analysis only. No model, architecture, label, partition or existing result was changed; no training was run;
nothing was committed, pushed or deployed. New files only (listed at the end).

## 1. Label provenance (verified from local files)

| Source | Label meaning | Basis |
|---|---|---|
| **RSNA** (Hospitals B, C) | `Target = 1` exactly for patients of class **Lung Opacity** (6,012). `Target = 0` for **Normal** (8,851) *and* **No Lung Opacity / Not Normal** (11,821, ≈ 57 % of RSNA negatives). | Radiologist annotation of the image (bounding boxes around opacities, per the challenge's own definition). Checked: `stage_2_train_labels.csv` Target = 1 ⇔ `stage_2_detailed_class_info.csv` class = Lung Opacity, for all 26,684 patients. |
| **Kermany** (Hospital A) | Pneumonia (bacterial or viral) vs Normal, a *clinical diagnosis* of paediatric patients. | Dataset documentation (not re-verifiable locally). |

So "Pneumonia" in B/C means "image shows a radiologist-marked opacity", in A it means "clinically diagnosed pneumonia". The
"No Lung Opacity / Not Normal" images are abnormal on the radiograph in some other way (or equivocal) and are negative **by
definition** in RSNA's labelling, not by error. Project decision DG-2 (CLAUDE.md resolved decision 2) keeps this grouping.

Frozen natural partition, test split (n = 4,838, unchanged): Kermany Normal 235, Kermany Pneumonia 600, RSNA Normal 1,340,
RSNA No Lung Opacity / Not Normal 1,761, RSNA Lung Opacity 902. All 32,540 images: the manifest gives per-split counts.

## 2. Versioned ambiguity manifest

`data/manifests/rsna_ambiguity_manifest_v1.json` (`manifest_version 1.0.0`, partition `hospitals_natural`). One row per
partition record with `original_class` (three-way RSNA class, or kermany_normal / kermany_pneumonia) and
`original_binary_label` copied unchanged from the partition. It contains **tags only; no corrected label exists in it**.
`build_manifest` fails if a three-way class contradicts the binary label. Code: `src/evaluation/rsna_ambiguity.py`.

## 3. Exact-image-hash leakage audit (`outputs/results/rsna_ambiguity/leakage_audit.json`)

All 32,540 images hashed twice: SHA-256 of the file bytes and SHA-256 of the decoded pixel array (+shape/dtype).

| | Duplicate groups | Images in them | Cross-split groups | Label-conflict groups |
|---|---|---|---|---|
| Identical file bytes | 27 | 56 | **0** | 0 |
| Identical pixels | 28 | 58 | **0** | 0 |

**No exact duplicate crosses train / validation / test.** What the duplicates are (investigated, nothing changed):
- 27 groups (56 images) are Kermany images of the **same patient** (same accession id, e.g. `NORMAL-3780378-0001/0002`), in the
  same split, same label. Split-safe. They do slightly over-weight those patients within training (hypothesis: negligible effect).
- 1 extra pixel-identical pair (not byte-identical): two different RSNA patients (`8f96ca8e…` in Hospital B, `08d88a60…` in
  Hospital C), both "No Lung Opacity / Not Normal", both train. It is a cross-**hospital** duplicate in the same split, with the same
  label: no split leakage; it does mean two simulated clients hold the same image.
Limit: this catches exact duplicates only. Near-duplicates (re-encoded, cropped, resized) were not searched.

## 4. Subgroup-aware evaluation (`subgroup_reanalysis.json`)

Primary number is the full test set (n = 4,838, asserted unchanged in code); subgroup numbers are complementary. Models: the
existing head-only FedAvg + DP ε = 4 checkpoints, trained on all data ("all data") and on the filtered partition ("filtered"), 3 seeds
(42, 123, 2024), mean ± sample SD. Scoring was done on the GPU in float64 (see note 1).

Threshold-free (no operating point involved):

| | all data | filtered |
|---|---|---|
| AUROC, full test (primary) | 0.8085 ± 0.0119 | 0.7803 ± 0.0197 |
| AUROC, hard negatives removed from scoring | 0.9068 ± 0.0074 | 0.9296 ± 0.0064 |
| Lung Opacity vs Normal | 0.9054 ± 0.0043 | 0.9350 ± 0.0025 |
| Lung Opacity vs hard negative | 0.7151 ± 0.0009 | 0.7041 ± 0.0024 |
| Scores separate Normal from hard negative (AUROC) | 0.7543 ± 0.0138 | 0.8281 ± 0.0039 |

Frozen operating points (all thresholds chosen on validation data only; specificities are on the **test** subgroups):

| Threshold (source) | Model | Sens. (full) | Spec. (full) | Spec. Normal | Spec. hard-neg | Sens. Lung Opacity |
|---|---|---|---|---|---|---|
| 0.5 (fixed protocol) | all data | 0.476 | 0.879 | 0.985 | 0.789 | 0.499 |
| | filtered | 0.844 | 0.616 | 0.869 | 0.383 | 0.856 |
| spec ≥ 90 % on full validation | all data | 0.417 ± 0.056 | 0.901 | 0.989 | 0.825 | 0.452 |
| | filtered | 0.295 ± 0.019 | 0.905 | 0.992 | 0.826 | 0.428 |
| Youden J on full validation | all data | 0.747 | 0.734 | 0.919 | 0.583 | 0.727 |
| | filtered | 0.818 | 0.644 | 0.887 | 0.418 | 0.834 |
| spec ≥ 90 % on clear-label validation only | all data | 0.755 | 0.728 | 0.919 | 0.576 | 0.739 |
| | filtered | 0.794 | 0.654 | 0.900 | 0.429 | 0.825 |

## 5. Why the filtered retrain behaved as it did (decomposition; no new training)

1. **Filtering effect on ranking (threshold-free).** Filtering improves ranking on clear-label cases (+0.023 AUROC) and slightly worsens
   ranking against hard negatives; the full-test AUROC falls 0.028. Verified.
2. **The sensitivity gain at 0.5 is an operating-point shift, not better discrimination.** The validation-fitted logit intercept is
   −0.02 ± 0.08 for the all-data models but **−1.12 ± 0.02** for the filtered models (`platt_recalibration.json`). After a
   validation-only slope+intercept recalibration, the filtered model at 0.5 has sensitivity 0.408 and specificity 0.859 (was 0.844 / 0.616)
   — i.e. its high sensitivity / low specificity came from a bias, which validation data removes. At a matched 90 % validation
   specificity the filtered model is *less* sensitive than the all-data model (0.295 vs 0.417). Verified.
3. **Threshold choice** moves each model's sensitivity/specificity by 0.3–0.4 along its own curve (table above), larger than the
   difference between the models at matched operating points. Verified.
4. **Calibration.** Raw ECE (test): all data 0.196, filtered 0.285. Validation-fitted temperature scaling fixes the all-data model
   (0.029 ± 0.008) but not the filtered one (0.209 ± 0.012), because a single temperature cannot remove a bias; slope+intercept
   recalibration brings the filtered model to 0.052 ± 0.012. Verified.
5. **Class balance.** The filtered training set is 49.9 % pneumonia vs 31.8 % in the full one, while the test set stays at 31.0 %.
   Whether that prior shift *causes* the −1.12 intercept is a **hypothesis**: training used per-client class weights
   (`class-weights = "local"`) and DP noise, so the effective prior is not simply the raw prevalence. The second change (removing
   hard negatives from training) is confounded with the first and was not separated, since doing so would need new training.

## 6. Why label correction is not justified, and what would justify it

- The 1,761 test images are negative **under the dataset's own definition**; they are not shown to be mislabelled. A model flagging
  them as pneumonia is wrong against RSNA's label even if a radiologist might consider some equivocal.
- No available metadata can reveal a wrong label: no radiology reports, clinical records, cultures or follow-up, and only one read per
  image in the files we hold. Any "correction" would be our own guess, i.e. invented labels, and would make results incomparable with
  RSNA and with Hospital A's clinical-diagnosis labels.
- Models disagreeing with a label is not evidence the label is wrong (the model is the thing under test).
- Evidence that would justify relabelling: blinded re-adjudication of a sampled set by multiple radiologists (with inter-reader
  agreement), or linked clinical records; plus a pre-specified protocol applied identically to train/val/test, and reporting both
  original-label and adjudicated-label results.

## Notes
1. The documented filtered full-test AUROC (0.7799 in `docs/results.md`) is a float32 probability computation; float64 gives 0.7803.
   Verified: float32 probabilities saturate and create ties (CPU float32 mean 0.7799, float64 0.7803). Conclusions are unaffected.
2. Existing results, partitions and `docs/results.md` were not modified; this document does not supersede any of them.

## Files added
`src/evaluation/rsna_ambiguity.py`; `scripts/rsna_ambiguity_manifest_and_leakage.py`, `scripts/rsna_ambiguity_reanalysis.py`,
`scripts/rsna_ambiguity_recalibration.py`; `tests/test_rsna_ambiguity.py`; `data/manifests/rsna_ambiguity_manifest_v1.json`;
`outputs/results/rsna_ambiguity/{leakage_audit,subgroup_reanalysis,platt_recalibration}.json`; this document.
