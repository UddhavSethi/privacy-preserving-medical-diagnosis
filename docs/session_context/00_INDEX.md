# Session context archive (2026-10-26) — index

Saved because the chat context grew too large. These files are an ordered record of what was done, tested and measured in the
session, so a new session can resume without the transcript. Nothing here is new evidence: every number comes from raw files under
`outputs/` (gitignored) or from `docs/audit_2026-10.md`, `docs/audit_tables.md` and
`docs/deployment_privacy_accounting_evidence.md` (tracked, regenerated from raw files).

| File | Contents |
|---|---|
| `00_INDEX.md` | this index, repository state, reading guide (current vs superseded figures), how to resume |
| `01_timeline.md` | the session in chronological order: what was asked, what was done, which commit it produced |
| `02_tests_and_results.md` | every test run and measurement, in order, each tagged CURRENT or HISTORICAL |
| `03_state_risks_next_steps.md` | open risks, operational gotchas, pending decisions, next steps |
| `04_publication_updates_and_label_ambiguity.md` | later on 2026-10-26: PDFs regenerated and pushed, reference checks, related-work comparison, the Table II deployed-model answer, and the RSNA label-ambiguity analysis and recommendation (open) |

## Repository state when this was written

| Item | State |
|---|---|
| Branch | `main` |
| Audit commit | `7187c049307dcf8643831d043a6d830355e919bf` — "Audit: like-for-like evaluation, DP validity fixes, privacy-accounting fix, shadow-model privacy audit" (47 files) |
| Follow-up commit | `bf59584d96be9f6e51aaa619f0dbdc41ba096049` — "Add deployment privacy-accounting verification and evidence summary" (4 files), parent `7187c04` |
| This archive | committed on top of `bf59584` as a separate docs commit ("docs: preserve PneumoFL session context"); its hash is not recorded here because a commit cannot contain its own hash — run `git log -3` |
| Pushed? | When this was written: **no** (`origin/main` at `e3b2a77`). **Later the same day all three commits were pushed** (fast-forward, `origin/main` = `e2fe291`); see the addendum |
| Backups (local only, old history, never push) | `backup-before-trailer-rewrite` (= `082c8be`, history with Claude co-author trailers), `backup-v1-after-first-rewrite` (= `bb7e0a5`) |
| Deliberately uncommitted (when written) | `docs/report/IEEE_Access_Manuscript.pdf` (stale then; regenerated and committed later, see the addendum) |
| `CLAUDE.md` | never modified in this session; proposed edits are at the end of `docs/audit_2026-10.md` |
| Tests | 293 passed on the tree at `bf59584` (255 originally + 34 audit + 4 for the deployment verifier) |
| Docker | the committed code was deployed in Docker (6 rounds, CPU, TLS + node auth); reported ε matched an independent recomputation (3.991884 at round 6) — one seed, one run |

## Reading guide: current figures versus superseded ones

Use the CURRENT column. The superseded numbers remain in older documents (README history, the VIT report, the IEEE manuscript, older
`docs/results.md` text) and in the HISTORICAL entries of `02_tests_and_results.md`; they are kept only so earlier statements can be traced.

| Topic | CURRENT (use this) | SUPERSEDED (do not quote) |
|---|---|---|
| Cost of federation | ≈ 0.017 macro (within-hospital) AUROC (0.893 → 0.876) | "≈ 9 AUROC points" (pooled AUROC across hospitals with 3× different prevalence) |
| Federation vs local models | beats every local model on the pooled split; slightly below each hospital's own local model in-domain | "federation did not beat the best single hospital" (compared different test sets) |
| Cost of DP at ε = 4 | ≈ 0.038 macro AUROC and ≈ 0.065 sensitivity vs the DP-matched non-private baseline | AUROC 0.809 vs FedAvg 0.814 (pooled, mismatched baseline) |
| DP sensitivity | partly a threshold artefact; real cost ≈ 0.065 at matched specificity | "misses more than half of pneumonia cases" at threshold 0.5 |
| Reported ε of the SecAgg+/DP client | cumulative: 2.12 → 3.17 → 3.992 over 20 rounds (deployment: 2.94 → 3.99 over 6) | a constant 2.12 per round; also 3.37 / 3.60 in earlier notes (single-round values) |
| Membership inference | attack power ≈ chance; LiRA AUROC 0.517 without DP, ≈ 0.50 with DP | "centralized 0.5084 is a real, repeatable effect" (inside its own negative control, 0.5038) |
| Transport security | server-side TLS + Flower node authentication | "mutually authenticated TLS" |
| Row 6 (full system) | 3 seeds in simulation, macro AUROC 0.830 ± 0.010; Docker demonstration | "single exploratory run" |

## How to resume

1. Read `docs/audit_2026-10.md` (authoritative) and `02_tests_and_results.md`.
2. `git log --oneline -6`, `git status` — confirm the state above.
3. Raw evidence is in gitignored `outputs/`; `docs/audit_raw_outputs_manifest.md` fingerprints 53 audit files and
   `docs/deployment_privacy_accounting_evidence.md` fingerprints the deployment evidence.
4. Reproduce with `scripts/reproduce_audit.sh` (GPU recommended); regenerate tables with `uv run python scripts/make_audit_tables.py`.

## Addendum (later on 2026-10-26, after the archive commit): deliverables regenerated and `CLAUDE.md` updated

- The VIT project report (`docs/report/PneumoFL_VIT_Project_Report.pdf`, now 51 pages), `docs/report/metrics_final.pdf` and the IEEE Access manuscript
  (`docs/report/IEEE_Access_Manuscript.pdf`, 6 pages) were regenerated on the macro-AUROC protocol; their tables and charts are read from the raw
  result files by the generators in `docs/report/build/` (gitignored). They no longer carry the superseded claims listed in the reading guide above, except as
  explicitly labelled corrections. The "stale deliverables" risk in `03_state_risks_next_steps.md` is resolved by this.
- `CLAUDE.md` was updated at the owner's instruction after review of the proposals (new resolved decision 17; §4 SciPy row; §6 transport wording; §11
  evaluation protocol; §14 status/row 6/tests; pending decision 2 resolved; §15 items 5, 9–12). The statement above that `CLAUDE.md` was never modified applies
  to the work up to the archive commit only.
- These changes (the three PDFs, the two regenerated charts, `CLAUDE.md`, the audit document and this addendum) were committed together in one local publication-documentation commit; the IEEE PDF became tracked in it. Whether that commit has been pushed is not recorded here — run `git status` and `git log origin/main..HEAD`.

## Addendum 2 (later on 2026-10-26)

Everything above was pushed: `origin/main` = `9c26cc3` (publication update `641e3cc`, related-work comparison `9c26cc3`). The tests, state and open items after that point are in
`04_publication_updates_and_label_ambiguity.md`; where it differs from the state table above, it is the later and correct one.
