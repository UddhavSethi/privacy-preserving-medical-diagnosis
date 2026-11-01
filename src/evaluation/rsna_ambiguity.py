"""RSNA label-ambiguity analysis helpers (no model code, no training, no label changes).

RSNA's `Target` is 1 only for "Lung Opacity" patients; "Normal" and "No Lung Opacity / Not
Normal" (a hard negative: abnormal but without a pneumonia-like opacity) are both Target 0.
This module tags every image with its ORIGINAL three-way class, audits exact-duplicate
images across splits, computes subgroup-aware metrics, and applies validation-only
threshold selection. Original labels are never altered.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MANIFEST_VERSION = "1.0.0"
NORMAL = "Normal"
HARD_NEGATIVE = "No Lung Opacity / Not Normal"
LUNG_OPACITY = "Lung Opacity"
KERMANY_NORMAL = "kermany_normal"
KERMANY_PNEUMONIA = "kermany_pneumonia"
RSNA_CLASSES = (NORMAL, HARD_NEGATIVE, LUNG_OPACITY)
SPLITS = ("train", "val", "test")


def load_rsna_class_info(path: Path) -> dict[str, str]:
    """patientId -> original RSNA three-way class (duplicate rows collapsed, conflicts rejected)."""
    df = pd.read_csv(path)
    if (df.groupby("patientId")["class"].nunique() > 1).any():
        raise ValueError("patientId with conflicting classes in class info csv")
    return df.drop_duplicates("patientId").set_index("patientId")["class"].to_dict()


def subgroup_of(record: dict, rsna_class: dict[str, str]) -> str:
    """Original-class subgroup for one partition record. Binary label is never consulted for RSNA."""
    if record["source"] == "kermany":
        return KERMANY_PNEUMONIA if record["label"] == "Pneumonia" else KERMANY_NORMAL
    cls = rsna_class[record["patient_id"].removeprefix("rsna-")]
    if cls not in RSNA_CLASSES:
        raise ValueError(f"unknown RSNA class {cls!r}")
    return cls


def check_binary_consistency(record: dict, subgroup: str) -> None:
    """The three-way class must agree with the partition's binary label (Target=1 <=> Lung Opacity)."""
    expect_pos = subgroup in (LUNG_OPACITY, KERMANY_PNEUMONIA)
    if (record["label"] == "Pneumonia") != expect_pos:
        raise ValueError(f"binary label {record['label']!r} inconsistent with subgroup {subgroup!r}")


def build_manifest(partition: dict, rsna_class: dict[str, str], partition_name: str) -> dict:
    """Versioned ambiguity manifest: one row per partition record, original label preserved."""
    rows = []
    for h, recs in sorted(partition["hospitals"].items()):
        for r in recs:
            sg = subgroup_of(r, rsna_class)
            check_binary_consistency(r, sg)
            rows.append({
                "hospital": h, "source": r["source"], "patient_id": r["patient_id"],
                "relative_path": r["relative_path"], "frozen_split": r["frozen_split"],
                "original_binary_label": r["label"], "original_class": sg,
            })
    counts: dict = defaultdict(lambda: defaultdict(int))
    for x in rows:
        counts[x["frozen_split"]][x["original_class"]] += 1
    return {
        "manifest_version": MANIFEST_VERSION, "partition": partition_name,
        "note": "Tags only. original_binary_label is copied unchanged from the partition; no label is corrected.",
        "counts_by_split": {s: dict(v) for s, v in counts.items()}, "records": rows,
    }


def test_subgroups_in_pooled_order(partition: dict, rsna_class: dict[str, str], split: str = "test",
                                   hospitals=("A", "B", "C")) -> np.ndarray:
    """Subgroup per sample in the exact order `load_pooled_features` yields `split` rows:
    hospitals in order, within a hospital grouped by sorted source, records in partition order."""
    out = []
    for h in hospitals:
        by = defaultdict(list)
        for r in partition["hospitals"][h]:
            if r["frozen_split"] == split:
                by[r["source"]].append(r)
        for _, rs in sorted(by.items()):
            out.extend(subgroup_of(r, rsna_class) for r in rs)
    return np.array(out)


# --------------------------------------------------------------------------- leakage audit
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pixel_hash(path: Path) -> str:
    """Hash of decoded pixels (+shape) so identical images with different file metadata still match."""
    if path.suffix.lower() == ".dcm":
        import pydicom
        arr = pydicom.dcmread(str(path)).pixel_array
    else:
        from PIL import Image
        arr = np.asarray(Image.open(path).convert("L"))
    a = np.ascontiguousarray(arr)
    return sha256_bytes(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes())


def find_duplicate_groups(records: list[dict], hash_key: str) -> list[list[dict]]:
    """Groups of >=2 records sharing `hash_key` (records need that key plus patient_id/frozen_split)."""
    g: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        g[r[hash_key]].append(r)
    return [v for v in g.values() if len(v) > 1]


def classify_duplicate_group(group: list[dict]) -> dict:
    splits = sorted({r["frozen_split"] for r in group})
    pats = sorted({r["patient_id"] for r in group})
    labels = sorted({r["original_binary_label"] for r in group})
    return {
        "size": len(group), "splits": splits, "cross_split": len(splits) > 1,
        "patients": pats, "same_patient": len(pats) == 1, "binary_labels": labels,
        "label_conflict": len(labels) > 1,
        "members": [{k: r[k] for k in ("hospital", "patient_id", "relative_path", "frozen_split",
                                       "original_class", "original_binary_label")} for r in group],
    }


# --------------------------------------------------------------------------- subgroup metrics
def _rate(mask_pred: np.ndarray) -> float:
    return float(mask_pred.mean()) if len(mask_pred) else float("nan")


def subgroup_metrics(y: np.ndarray, score: np.ndarray, sub: np.ndarray, threshold: float) -> dict:
    """Primary = full test set. Complementary = per-subgroup rates and pairwise AUROCs.
    `threshold` must come from validation data; nothing here selects anything."""
    y, score, sub = map(np.asarray, (y, score, sub))
    pred = score >= threshold
    out = {"full": _binary_block(y, score, threshold)}
    out["by_subgroup"] = {}
    for g in sorted(set(sub)):
        k = sub == g
        d = {"n": int(k.sum())}
        if y[k].max() == 0:      # negatives only: false-positive rate = 1 - specificity
            d["fpr"], d["specificity"] = _rate(pred[k]), 1 - _rate(pred[k])
        elif y[k].min() == 1:    # positives only: sensitivity
            d["sensitivity"] = _rate(pred[k])
        out["by_subgroup"][g] = d
    pos = sub == LUNG_OPACITY
    for name, neg in (("lung_opacity_vs_normal", sub == NORMAL), ("lung_opacity_vs_hard_negative", sub == HARD_NEGATIVE),
                      ("lung_opacity_vs_all_rsna_negatives", np.isin(sub, [NORMAL, HARD_NEGATIVE]))):
        k = pos | neg
        if pos.any() and neg.any():
            yy = pos[k].astype(int)
            out[name] = {"n": int(k.sum()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                         "auroc": float(roc_auc_score(yy, score[k])), "auprc": float(average_precision_score(yy, score[k])),
                         **_binary_block(yy, score[k], threshold, auc=False)}
    nh = sub == NORMAL
    hn = sub == HARD_NEGATIVE
    if nh.any() and hn.any():  # can the model tell the two negative types apart? (diagnostic only)
        out["normal_vs_hard_negative_score_auroc"] = float(
            roc_auc_score(hn[nh | hn].astype(int), score[nh | hn]))
    return out


def _binary_block(y, score, thr, auc=True) -> dict:
    pred = score >= thr
    tp, fn = int(((y == 1) & pred).sum()), int(((y == 1) & ~pred).sum())
    tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    d = {"n": int(len(y)), "prevalence": float(np.mean(y)), "threshold": float(thr), "sensitivity": sens,
         "specificity": spec, "balanced_accuracy": (sens + spec) / 2, "tp": tp, "fn": fn, "tn": tn, "fp": fp}
    if auc:
        d["auroc"] = float(roc_auc_score(y, score))
        d["auprc"] = float(average_precision_score(y, score))
    return d


# ------------------------------------------------------- validation-only threshold selection
def threshold_at_val_specificity(val_y: np.ndarray, val_score: np.ndarray, target: float = 0.9) -> float:
    """Lowest threshold with validation specificity >= target (same rule as evaluation_protocol)."""
    neg = np.sort(val_score[val_y == 0])
    k = int(np.ceil(target * len(neg))) - 1
    return float(np.nextafter(neg[min(max(k, 0), len(neg) - 1)], np.inf))


def threshold_at_val_youden(val_y: np.ndarray, val_score: np.ndarray) -> float:
    """Threshold maximising validation sensitivity+specificity-1 (candidates = observed scores)."""
    best, best_t = -2.0, 0.5
    for t in np.unique(val_score):
        pr = val_score >= t
        j = pr[val_y == 1].mean() + (~pr[val_y == 0]).mean() - 1
        if j > best:
            best, best_t = j, float(t)
    return best_t


def fit_temperature(val_y: np.ndarray, val_logits: np.ndarray) -> float:
    """Single temperature minimising validation NLL (grid on log T; validation data only)."""
    ts = np.exp(np.linspace(np.log(0.25), np.log(8.0), 400))
    lp = lambda t: _log_softmax(val_logits / t)[np.arange(len(val_y)), val_y].mean()
    return float(ts[int(np.argmax([lp(t) for t in ts]))])


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if k.any():
            ece += k.mean() * abs(y[k].mean() - p[k].mean())
    return float(ece)


def manifest_to_json(m: dict, path: Path) -> None:
    path.write_text(json.dumps(m, indent=1))
