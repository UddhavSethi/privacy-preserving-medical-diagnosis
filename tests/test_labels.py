import csv

import pytest

from src.data.labels import load_kermany_records, load_rsna_records


@pytest.fixture
def kermany_raw_root(tmp_path):
    root = tmp_path / "kermany"
    chest_xray = root / "CellData" / "chest_xray"
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            (chest_xray / split / cls).mkdir(parents=True)

    # Same patient (accession 1000001) contributes to both NORMAL and PNEUMONIA train dirs
    # only in this synthetic fixture to exercise the conflicting-label path elsewhere; here
    # we keep it realistic: one patient, two images, same class.
    (chest_xray / "train" / "NORMAL" / "NORMAL-1000001-0001.jpeg").write_bytes(b"x")
    (chest_xray / "train" / "NORMAL" / "NORMAL-1000001-0002.jpeg").write_bytes(b"x")
    (chest_xray / "train" / "NORMAL" / "NORMAL-1000002-0001.jpeg").write_bytes(b"x")
    (chest_xray / "train" / "PNEUMONIA" / "BACTERIA-2000001-0001.jpeg").write_bytes(b"x")
    (chest_xray / "test" / "PNEUMONIA" / "VIRUS-3000001-0001.jpeg").write_bytes(b"x")
    return root


@pytest.fixture
def rsna_raw_root(tmp_path):
    root = tmp_path / "rsna"
    root.mkdir()
    with open(root / "stage_2_train_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patientId", "x", "y", "width", "height", "Target"])
        writer.writerow(["patient-a", "", "", "", "", "0"])
        writer.writerow(["patient-b", "10", "10", "5", "5", "1"])
        writer.writerow(["patient-b", "20", "20", "5", "5", "1"])  # 2nd bbox, same patient
        writer.writerow(["patient-c", "", "", "", "", "0"])
    return root


def test_kermany_patient_grouping_and_label_mapping(kermany_raw_root):
    records = load_kermany_records(kermany_raw_root)
    assert len(records) == 5

    by_patient = {}
    for r in records:
        by_patient.setdefault(r["patient_id"], []).append(r)

    # accession 1000001 groups both of its images under one patient_id.
    assert len(by_patient["kermany-1000001"]) == 2
    assert all(r["label"] == "Normal" for r in by_patient["kermany-1000001"])

    labels = {r["patient_id"]: r["label"] for r in records}
    assert labels["kermany-2000001"] == "Pneumonia"
    assert labels["kermany-3000001"] == "Pneumonia"


def test_kermany_unrecognized_filename_raises(tmp_path):
    root = tmp_path / "kermany"
    d = root / "CellData" / "chest_xray" / "train" / "NORMAL"
    d.mkdir(parents=True)
    (root / "CellData" / "chest_xray" / "train" / "PNEUMONIA").mkdir(parents=True)
    (root / "CellData" / "chest_xray" / "test" / "NORMAL").mkdir(parents=True)
    (root / "CellData" / "chest_xray" / "test" / "PNEUMONIA").mkdir(parents=True)
    (d / "not-a-recognized-pattern.jpeg").write_bytes(b"x")

    with pytest.raises(ValueError, match="Unrecognized Kermany filename"):
        load_kermany_records(root)


def test_rsna_label_mapping_and_bbox_dedup(rsna_raw_root):
    records = load_rsna_records(rsna_raw_root)
    assert len(records) == 3  # 4 CSV rows, patient-b's 2 bboxes dedupe to 1 record

    by_patient = {r["patient_id"]: r for r in records}
    assert by_patient["rsna-patient-a"]["label"] == "Normal"
    assert by_patient["rsna-patient-b"]["label"] == "Pneumonia"
    assert by_patient["rsna-patient-c"]["label"] == "Normal"
    assert by_patient["rsna-patient-b"]["relative_path"] == "stage_2_train_images/patient-b.dcm"
