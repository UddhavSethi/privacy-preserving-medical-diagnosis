import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import rsna_ambiguity as ra

ROOT = Path(__file__).resolve().parents[1]
PART = ROOT / "data/partitions/hospitals_natural.json"
CSV = ROOT / "data/raw/rsna/stage_2_detailed_class_info.csv"
MAN = ROOT / "data/manifests/rsna_ambiguity_manifest_v1.json"
AUD = ROOT / "outputs/results/rsna_ambiguity/leakage_audit.json"
needs_data = pytest.mark.skipif(not (PART.exists() and CSV.exists()), reason="local data not present")


def _tiny():
    part = {"hospitals": {"A": [
        {"source": "kermany", "patient_id": "kermany-1", "label": "Pneumonia", "relative_path": "x", "frozen_split": "test"},
        {"source": "rsna", "patient_id": "rsna-a", "label": "Normal", "relative_path": "a.dcm", "frozen_split": "test"},
        {"source": "rsna", "patient_id": "rsna-b", "label": "Normal", "relative_path": "b.dcm", "frozen_split": "train"},
        {"source": "rsna", "patient_id": "rsna-c", "label": "Pneumonia", "relative_path": "c.dcm", "frozen_split": "val"}]}}
    cls = {"a": ra.HARD_NEGATIVE, "b": ra.NORMAL, "c": ra.LUNG_OPACITY}
    return part, cls


def test_manifest_preserves_original_labels_and_tags_three_way_class():
    part, cls = _tiny()
    m = ra.build_manifest(part, cls, "t")
    assert [r["original_binary_label"] for r in m["records"]] == [r["label"] for r in part["hospitals"]["A"]]
    assert [r["original_class"] for r in m["records"]] == [ra.KERMANY_PNEUMONIA, ra.HARD_NEGATIVE, ra.NORMAL, ra.LUNG_OPACITY]
    assert m["manifest_version"] == ra.MANIFEST_VERSION and "corrected" not in json.dumps(m).lower().replace("no label is corrected", "")


def test_manifest_rejects_class_inconsistent_with_binary_label():
    part, cls = _tiny()
    cls["c"] = ra.NORMAL  # Target=1 patient tagged Normal -> must fail loudly, never silently relabel
    with pytest.raises(ValueError):
        ra.build_manifest(part, cls, "t")


@needs_data
def test_real_manifest_matches_partition_and_frozen_test_counts():
    part = json.load(open(PART)); cls = ra.load_rsna_class_info(CSV)
    m = ra.build_manifest(part, cls, "hospitals_natural")
    assert len(m["records"]) == sum(len(v) for v in part["hospitals"].values()) == 32540
    assert m["counts_by_split"]["test"] == {ra.KERMANY_NORMAL: 235, ra.KERMANY_PNEUMONIA: 600, ra.LUNG_OPACITY: 902,
                                            ra.HARD_NEGATIVE: 1761, ra.NORMAL: 1340}
    assert sum(m["counts_by_split"]["test"].values()) == 4838  # full test set unchanged
    if MAN.exists():  # committed/generated artefact must equal a fresh rebuild
        assert json.load(open(MAN))["counts_by_split"] == m["counts_by_split"]


@needs_data
def test_subgroup_order_aligns_with_pooled_test_labels():
    from src.training.trainer import FEATURE_KEY, load_pooled_features
    part = json.load(open(PART)); cls = ra.load_rsna_class_info(CSV)
    p = load_pooled_features(PART, list("ABC"), ROOT / "data/feature_cache", FEATURE_KEY)
    sub = ra.test_subgroups_in_pooled_order(part, cls, "test")
    y = p.test_labels.numpy()
    assert len(sub) == len(y) == 4838
    assert np.array_equal(y, np.isin(sub, [ra.LUNG_OPACITY, ra.KERMANY_PNEUMONIA]).astype(int))


def test_duplicate_detection_flags_cross_split_and_label_conflict():
    base = dict(hospital="A", original_class="c")
    recs = [dict(base, patient_id="p1", frozen_split="train", original_binary_label="Normal", relative_path="1", h="x"),
            dict(base, patient_id="p2", frozen_split="test", original_binary_label="Pneumonia", relative_path="2", h="x"),
            dict(base, patient_id="p3", frozen_split="test", original_binary_label="Normal", relative_path="3", h="y")]
    g = ra.find_duplicate_groups(recs, "h")
    assert len(g) == 1
    c = ra.classify_duplicate_group(g[0])
    assert c["cross_split"] and c["label_conflict"] and not c["same_patient"]


def test_pixel_hash_ignores_file_metadata_but_not_pixels(tmp_path):
    from PIL import Image
    a = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    Image.fromarray(a).save(tmp_path / "a.png"); Image.fromarray(a).save(tmp_path / "b.png", compress_level=0)
    Image.fromarray(a[::-1].copy()).save(tmp_path / "c.png")
    assert (tmp_path / "a.png").read_bytes() != (tmp_path / "b.png").read_bytes()
    assert ra.pixel_hash(tmp_path / "a.png") == ra.pixel_hash(tmp_path / "b.png") != ra.pixel_hash(tmp_path / "c.png")


@needs_data
@pytest.mark.skipif(not AUD.exists(), reason="leakage audit not generated")
def test_real_leakage_audit_reports_no_cross_split_duplicates():
    a = json.load(open(AUD))
    assert a["n_images"] == 32540
    for k in ("file_sha256", "pixel_sha256"):
        assert a[k]["n_cross_split_groups"] == 0 and a[k]["n_label_conflict_groups"] == 0


def test_subgroup_metrics_known_values():
    sub = np.array([ra.NORMAL] * 4 + [ra.HARD_NEGATIVE] * 4 + [ra.LUNG_OPACITY] * 4)
    y = np.array([0] * 8 + [1] * 4)
    s = np.array([.1, .1, .1, .6] + [.6, .6, .2, .2] + [.9, .9, .9, .3])
    m = ra.subgroup_metrics(y, s, sub, 0.5)
    assert m["by_subgroup"][ra.NORMAL]["specificity"] == pytest.approx(.75)
    assert m["by_subgroup"][ra.HARD_NEGATIVE]["specificity"] == pytest.approx(.5)
    assert m["by_subgroup"][ra.LUNG_OPACITY]["sensitivity"] == pytest.approx(.75)
    assert m["full"]["n"] == 12 and (m["full"]["tp"], m["full"]["fp"], m["full"]["fn"], m["full"]["tn"]) == (3, 3, 1, 5)
    assert m["lung_opacity_vs_normal"]["auroc"] > m["lung_opacity_vs_hard_negative"]["auroc"]


def test_validation_only_thresholds_and_temperature_ignore_test_data():
    rng = np.random.default_rng(0)
    vy = rng.integers(0, 2, 3000); vs = rng.random(3000) * .5 + .4 * vy
    t1, j1 = ra.threshold_at_val_specificity(vy, vs, .9), ra.threshold_at_val_youden(vy, vs)
    # the helpers' signatures take no test arguments; changing any test array cannot move them
    import inspect
    for f in (ra.threshold_at_val_specificity, ra.threshold_at_val_youden, ra.fit_temperature):
        assert not any("test" in n for n in inspect.signature(f).parameters)
    assert 0.9 <= (vs[vy == 0] < t1).mean() <= 0.905 and t1 == ra.threshold_at_val_specificity(vy.copy(), vs.copy(), .9)
    z = np.stack([np.zeros(3000), rng.normal(size=3000) * 4 + 2 * (2 * vy - 1)], 1)  # overconfident logits
    assert ra.fit_temperature(vy, z) > 1.0


def test_reanalysis_script_selects_only_from_validation_arrays():
    src = (ROOT / "scripts/rsna_ambiguity_reanalysis.py").read_text()
    calls = re.findall(r"(?:threshold_at_val_\w+|fit_temperature)\(([^)]*)\)", src)
    assert calls and all("ty" not in re.findall(r"\w+", c) and "pt" not in re.findall(r"\w+", c) for c in calls), calls
    assert "assert {g: int((tsub == g).sum()) for g in EXPECT} == EXPECT" in src  # test set frozen-count guard
