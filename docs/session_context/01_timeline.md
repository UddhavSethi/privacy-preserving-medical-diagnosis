# Timeline of the session (chronological)

Project: PneumoFL — privacy-preserving pneumonia detection with federated learning (B.Tech BCSE497J, VIT).
Governing document `CLAUDE.md` (not to be edited without owner approval; no pushes/dependency changes without asking).

## 1. Orientation
- Loaded context: stages 0–23 and OPT-1..6 complete; ablation campaign done (27 runs); deployed demo app
  `pneumonia-fl-demo.streamlit.app` on `fedavg_dp_eps4`; 255 tests passing; pending decisions listed in `CLAUDE.md` §14.

## 2. Project report check and update
- Located `docs/report/PneumoFL_VIT_Project_Report.pdf` (47 pages, built 2026-10-21 by gitignored generators in
  `docs/report/build/*.py`) and `docs/report/metrics_final.pdf`.
- Found it stale: §7.4 said a filtered-data retrain "was not built"; the 3-seed filtered retrain (2026-10-23) exists.
- Rewrote that passage (clear-label AUROC 0.907→0.930, full-test 0.809→0.780, specificity 0.88→0.62; decision not to adopt),
  rebuilt the report, also updated `metrics_final.pdf`'s limitation sentence.
- Commit `becf2a3`, pushed. The commit message was then changed on request ("Update filtered-retraining experiments")
  by amend + `--force-with-lease` push → `082c8be`.
- Later report fixes (committed and pushed): blank page 17 removed by dropping trailing spacers before chapter breaks
  (`da8ff1e`, report now 46 pages); cover page lists Uddhav Sethi first (`e3b2a77`).

## 3. Removing Claude attribution from the GitHub history
- Search found `Co-Authored-By: Claude Sonnet 5` on 55 commits (not only the latest).
- First `filter-branch` rewrite (with backup branch `backup-before-trailer-rewrite`) removed the co-author lines but
  its regex also swallowed the newline after them, gluing a second trailer (`Claude-Session: https://claude.ai/...`,
  51 commits) onto the end of message bodies. Found when the user's pasted check showed session links still on GitHub.
- Second rewrite stripped the session links (backup `backup-v1-after-first-rewrite`), trees verified identical,
  force-pushed with lease. Final remote history had 0 matches for `claude-session|co-authored-by|claude.ai|noreply@anthropic`;
  every commit authored/committed by Uddhav Sethi.
- ~50 stale short hashes in `CLAUDE.md`, `docs/SESSION_STATE.md`, `docs/pneumoscan_redesign_plan.md` remapped by position
  (commits `b580a0b`, then `7fd3970`). Streamlit app confirmed still loading.
- Process lesson: `pkill -f <pattern>` killed my own shell several times because the pattern was in its own command line.

## 4. IEEE Access manuscript
- Generated a 5-page two-column manuscript `docs/report/IEEE_Access_Manuscript.pdf` (generator `docs/report/build/ieee_access.py`,
  gitignored; the PDF is untracked). Author order Uddhav Sethi, Chirag Gadhyan, Ishaan S Shrivastav.
- Caveats given then: corresponding e-mail and funding line are assumptions; needs the official IEEE template; novelty claim needs a
  literature search. **Later found stale** (see 03): it states the pre-audit pooled-AUROC claims and "mutually authenticated TLS".

## 5. Scientific audit toward a Q2 journal (main body of work)
Instruction: investigate FL-vs-centralized gap and DP sensitivity, complete the ablation (incl. row 6), strengthen/validate privacy
evaluation, reconcile docs, improve reproducibility; no test-set tuning, no fabricated results, no pushing/history rewrite/deletions.
Order of work (details and numbers in `02_tests_and_results.md`):
1. Audit reads + zero-training re-scoring of all checkpoints per hospital (found row 1 not like-for-like, pooled-AUROC artefact).
2. DP-loss batch-dependence demonstration; noise/clipping/accountant validity checks.
3. In-process FedAvg harness (`src/experiments/fl_sim.py`), parity vs Flower; validation-only FedAvg diagnosis.
4. Validation-only DP recipe sweep (48 runs; CPU runs deadlocked three times → `maxtasksperchild=1`; GPU enabled ~15× faster).
5. Selection of DP recipe; report-phase ablation under one protocol on test splits.
6. Re-audit of the original membership-inference attack + new LiRA shadow-model audits (non-DP, ε 1/4/8).
7. Real Flower SecAgg+ campaign (rows 4' and 6, 3 seeds) with communication accounting; found and fixed two instrumentation bugs
   (byte counter crash on empty lists; GPU hidden from Ray actors) and a real privacy-accounting reporting defect (F11).
8. Docs reconciliation, reproducibility script, checksum manifest, tests.
9. Final pre-commit review (diff scan, scipy declared in `pyproject.toml`/`uv.lock`/`requirements.txt`, ε recomputation,
   reproduction-order fixes), then local commit `7187c04` (47 files, no co-author trailer, IEEE PDF excluded).
- Mistakes disclosed to the user: deleted partial scratch output folders (`outputs/mia/dp_eps1`, `dp_eps8`) and smoke files in
  `outputs/comm/` from my own interrupted runs (regenerated); an initial unwanted `uv lock` re-resolution (reverted; scipy added by hand).

## 6. Docker deployment verification (fresh-process-per-round)
- User asked to deploy the full system in Docker from the committed code and check ε accumulates. Ran
  `NUM_ROUNDS=6 scripts/run_deployment_full_system.sh` with an evidence collector. Result: reported ε matched an independent
  recomputation exactly for all six rounds; 30 distinct `flwr-clientapp` processes per hospital (fresh process per message).
- GPU in containers is not possible here (no NVIDIA container toolkit; CPU-only images by owner decision DG-9).

## 7. Follow-up requests (completed)
- Follow-up commit `bf59584`: verification script (`scripts/verify_deployment_epsilon.py`), evidence collector, a test pinning the recomputation, and
  `docs/deployment_privacy_accounting_evidence.md`. The verifier was first tightened so that it reads the recorded configuration (pyproject defaults,
  the deployment wrapper's target ε, the partition file) instead of hardcoded values, and gained a calibration sanity check. Full suite: 293 passed.
  No training code, privacy parameters, existing results, `CLAUDE.md` or the previous commit were changed.
- This directory (`docs/session_context/`) saved at the user's request because the chat context had grown too large, then committed on its own
  ("docs: preserve PneumoFL session context") after review for secrets, personal data and machine-specific paths (none found).
- Nothing was pushed; `origin/main` remains at `e3b2a77`.
