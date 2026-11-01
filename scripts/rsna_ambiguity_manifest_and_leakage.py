#!/usr/bin/env python3
"""Builds the versioned ambiguity manifest and runs the exact-image-hash leakage audit.
Read-only on data/partitions and raw images. Writes NEW files only:
  data/manifests/rsna_ambiguity_manifest_v1.json
  outputs/results/rsna_ambiguity/leakage_audit.json
Refuses to overwrite existing outputs.
Usage: PYTHONPATH=. uv run python scripts/rsna_ambiguity_manifest_and_leakage.py"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
from src.evaluation.rsna_ambiguity import (MANIFEST_VERSION, build_manifest, classify_duplicate_group,
                                           find_duplicate_groups, load_rsna_class_info, pixel_hash, sha256_bytes)

R = Path('.'); PART = 'hospitals_natural'
MAN = R / f'data/manifests/rsna_ambiguity_manifest_v{MANIFEST_VERSION.split(".")[0]}.json'
AUD = R / 'outputs/results/rsna_ambiguity/leakage_audit.json'
KROOT = R / 'data/raw/kermany/CellData/chest_xray'


def abspath(rec):
    return (R / 'data/raw/rsna' / rec['relative_path']) if rec['source'] == 'rsna' else KROOT / rec['relative_path']


def hash_one(p):
    p = Path(p); return sha256_bytes(p.read_bytes()), pixel_hash(p)


if __name__ == '__main__':
    for f in (MAN, AUD):
        assert not f.exists(), f'refusing to overwrite {f}'
    part = json.load(open(R / f'data/partitions/{PART}.json'))
    cls = load_rsna_class_info(R / 'data/raw/rsna/stage_2_detailed_class_info.csv')
    man = build_manifest(part, cls, PART)
    print(json.dumps(man['counts_by_split'], indent=1))
    recs = man['records']
    # B and C are disjoint RSNA shards; each image appears once in the partition. Verify.
    keys = [(r['source'], r['relative_path']) for r in recs]
    assert len(keys) == len(set(keys)), 'a record appears in more than one hospital/split'
    with ProcessPoolExecutor(16) as ex:
        hs = list(tqdm(ex.map(hash_one, [str(abspath(r)) for r in recs], chunksize=64), total=len(recs)))
    for r, (fh, ph) in zip(recs, hs):
        r['file_sha256'], r['pixel_sha256'] = fh, ph
    audit = {'manifest_version': MANIFEST_VERSION, 'n_images': len(recs)}
    for key in ('file_sha256', 'pixel_sha256'):
        gs = [classify_duplicate_group(g) for g in find_duplicate_groups(recs, key)]
        audit[key] = {'n_duplicate_groups': len(gs), 'n_cross_split_groups': sum(g['cross_split'] for g in gs),
                      'n_label_conflict_groups': sum(g['label_conflict'] for g in gs),
                      'n_images_in_duplicate_groups': sum(g['size'] for g in gs), 'groups': gs}
        print(key, {k: v for k, v in audit[key].items() if k != 'groups'})
    # hashes live in the audit file, manifest keeps tags only (keeps the manifest small and label-focused)
    MAN.write_text(json.dumps(man, indent=1)); AUD.write_text(json.dumps(audit, indent=1))
