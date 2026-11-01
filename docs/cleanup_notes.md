# Deferred cleanup notes

Disk-space candidates reviewed on 2026-11-01. **Nothing here has been deleted.** The
disk had 222 GB free (51% used), so none of this is urgent. Revisit after the paper is
accepted, or after the first review round if no new experiments are requested.

All paths below are under `data/` and are gitignored.

| Path | Size | Safe to delete? | Notes |
|---|---|---|---|
| `data/_downloads` | 12 GB | **First candidate.** Yes, once `data/raw` is verified against the published checksums. | Original downloaded archives (e.g. the Mendeley Kermany bundle). |
| `data/clahe_cache` | 46 GB | Optional. Deleting costs a slow rebuild. | Cached CLAHE-enhanced images. Rebuild with `uv run python scripts/build_clahe_cache.py` (needs `data/raw`; deterministic). |
| `data/raw` | 5.0 GB | Not until the paper is accepted. | Original Kermany and RSNA images. Needed for any new experiment. |
| `data/feature_cache`, `data/feature_cache_finetuned`, `data/deployment_shards` | about 1.3 GB | Keep. | Needed by the demo app and the Docker deployment. Rebuildable but not worth the effort. |
| `data/partitions`, `data/manifests` | about 106 MB | **Never delete.** | Frozen splits and the ambiguity manifest, committed to git. This is the reproducibility record. |

## What deleting `data/clahe_cache` breaks

Anything that reads images instead of the cached features:

- fine-tuning: `scripts/train_centralized_finetune.py`, `src/federated/client_app_finetune.py`,
  `client_app_finetune_dp.py`, `server_app_finetune.py`
- Grad-CAM: `scripts/generate_explanations.py`, `scripts/run_gradcam_evaluation.py`
- cache builders: `scripts/build_feature_cache.py`, `scripts/build_feature_cache_finetuned.py`
- tests: `tests/test_gradcam.py`, `tests/test_app_inference.py`

Head-only runs, the Streamlit app, paper generation and the committed splits are not affected.
Rebuild the cache before running any of the above again.

## Before deleting anything

1. Confirm no further reviewer-driven experiments are expected (external validation set,
   gradient-inversion attack, extra seeds, re-running an ablation row).
2. Check that `data/raw` matches the published checksums (`scripts/download_kermany.py` verifies Kermany).
3. Check that the committed partition and manifest files and the download scripts still work.
4. Get the owner's explicit approval for the exact path.
