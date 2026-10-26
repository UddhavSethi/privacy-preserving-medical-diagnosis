# Addendum 2 (2026-10-26, later): publication updates, reference checks, related work, and the RSNA label-ambiguity question

Continues `00`–`03` in this directory. Same status convention: **[CURRENT]** may be quoted, **[OPEN]** is undecided or not done.

## 1. Repository state when this was written [CURRENT]

| Item | State |
|---|---|
| `origin/main` = local `main` | `9c26cc3dae3398fdb09791462d0e9799d950c3db` (pushed, fast-forward) |
| Working tree | clean before this addendum (this file and `00_INDEX.md`/memory edits are uncommitted) |
| Commits pushed this session, in order | `7187c04` audit → `bf59584` deployment verification → `e2fe291` context archive → `641e3cc` macro-AUROC publication update (VIT report, `metrics_final.pdf`, IEEE manuscript, `CLAUDE.md`) → `9c26cc3` related-work comparison |
| Backups (local only, never push) | `backup-before-trailer-rewrite` (= `082c8be`), `backup-v1-after-first-rewrite` (= `bb7e0a5`) |
| Tests | 293 passed (last full run before `641e3cc`; later commits changed only documents and PDFs) |
| `CLAUDE.md` | updated at the owner's instruction in `641e3cc` (resolved decision 17, §4 SciPy row, §6 wording, §11 evaluation protocol, §14 status, §15 items 5 and 9–12) |
| PDFs | VIT report 53 pages, IEEE manuscript 6 pages, `metrics_final.pdf` 3 pages; the pushed copies match the local files and contain the macro-AUROC results, the LiRA audit, ε = 3.992 and the Docker verification |

## 2. Regeneration of the three publication PDFs [CURRENT]

Generators live in `docs/report/build/` (gitignored); tables and charts are read from `outputs/results/exp_final_ablation.json`, `audit_per_hospital_eval.json`,
`audit_mia_existing.json`, `outputs/mia/*/analysis.json` and `outputs/deployment_lifecycle/verification.json`, so the documents cannot drift from the raw results.
Headline numbers that all three agree on (macro within-hospital test AUROC, 3 seeds): centralized 0.893; FedAvg 0.876 (cost ≈ 0.017); DP-matched baseline 0.869;
DP ε = 8/4/2/1 (validation-selected recipe) 0.839/0.830/0.817/0.796; row 6 (SecAgg+ and DP, ε = 4) 0.830. Withdrawn: "≈ 9 AUROC points" (pooled metric) and
"federation did not beat the best single hospital" (different test sets). The deployed model is described as the **original-recipe seed-42 checkpoint**; the
tuned recipe is not claimed to be deployed and its calibration is stated as not re-measured.

## 3. Reference verification of the IEEE manuscript [CURRENT]

- All 28 original references exist (title, authors, venue matched by web search). Page ranges confirmed for most; **not independently confirmed**: Shokri et al. (pp. 3–18)
  and Bonawitz et al. (pp. 1175–1191); Carlini et al. (pp. 1897–1914) matched only a secondary BibTeX entry.
- Fixed: Group Normalization [13] was never cited in the text and now is; the DenseNet reference [8] used the open-access pagination (4700–4708) and now uses the IEEE
  pagination (2261–2269) with the DOI. Dwork & Roth: the publisher's own citation page says 2013, most records 2014 (2014 kept).
- Four references added after checking each against its source: PriMIA (Kaissis et al., *Nat. Mach. Intell.* 3(6), 473–484, 2021; 14 authors, cited "et al."),
  Ziller et al. (*Sci. Rep.* 11, 13524, 2021), Adnan et al. (*Sci. Rep.* 12, 1953, 2022), Dayan et al. (EXAM, *Nat. Med.* 27(10), 1735–1743, 2021; 99 authors, cited "Dayan, I. et al.").
  The manuscript now has 32 references, all cited.

## 4. Related-work comparison added to both documents [CURRENT]

What the sources say (checked): PriMIA combines federated learning, differential privacy (journal abstract), secure multi-party aggregation, encrypted inference and a gradient-inversion
evaluation; its paediatric chest-radiography case study trains on the public Kermany data over three computation nodes and is tested on separate clinical sets (details from its arXiv
preprint; the journal page was not accessible). Ziller et al. (centralized DP-SGD on the same paediatric data): ROC-AUC 0.960 without privacy, 0.882 at ε ≈ 2.7–2.8, 0.848 at ε ≈ 0.5–0.6.
Adnan et al.: DP-FL on histopathology, ε = 2.90, δ = 1e-4, no secure aggregation or explanations. EXAM: 20 real institutions, AUC above 0.92, about 16% above single-site models.

Positioning that was written into the manuscript (Section II, Table I, Section VI) and the report (2.1, 2.2, Table 2.2, 7.11): no new algorithm; the contribution is the integration, the
per-layer cost under a within-hospital protocol, audited privacy accounting, and clinical-trust layers in one reproducible pipeline; narrower than the literature on encrypted inference,
inversion attacks and real/external sites; **no numerical cross-paper comparison and no state-of-the-art claim**; the coverage table's "–" means "not reported in the sources read".

## 5. Question: "which row of Table II is the deployed model?" [CURRENT answer]

None. The deployed app model is `fedavg_dp_eps4`, the **original-recipe** seed-42 checkpoint (`outputs/checkpoints/ablation/dp_eps4.0_seed42.pt`; Adam 1e-3, class-weighted loss),
run at threshold 0.35 with temperature T = 2.2534. Table II's "FedAvg + DP, ε = 4" row is the validation-selected recipe (0.830, sensitivity 0.538 over 3 seeds), which was never deployed.
The original recipe's own numbers are in the results file but not in Table II: macro AUROC 0.819 over 3 seeds (0.818 for seed 42), sensitivity at the per-hospital 90%-specificity threshold 0.502.
The deployed operating point is reported in manuscript Table V / report Table 7.4 (pooled AUROC 0.800, sensitivity 0.739, specificity 0.732). **[OPEN]** Offered, not done: add an
"original recipe (deployed checkpoint)" row or footnote to Table II in both documents.

## 6. RSNA label ambiguity: analysis and recommendation (nothing run yet) [OPEN decision]

Request received (summarised): investigate whether the 1,761 test images labelled Normal that show other abnormalities reflect label errors; trace the labels, audit, separate verified errors
from legitimate non-pneumonia abnormalities and uncertain cases, build a versioned correction manifest, check leakage, run a controlled original-vs-corrected retraining comparison across centralized /
FedAvg / DP, evaluate on a frozen protocol, explain the earlier filtered-retrain behaviour, and make no commits or pushes. The owner first asked whether it should be done.

Facts verified from the local metadata (`data/raw/rsna/`):
- `stage_2_detailed_class_info.csv`: 26,684 patients = "No Lung Opacity / Not Normal" 11,821, "Normal" 8,851, "Lung Opacity" 6,012.
- `stage_2_train_labels.csv`: Target = 1 for exactly the 6,012 "Lung Opacity" patients; Target = 0 for all 11,821 + 8,851. The 1,761 test images are part of the "No Lung Opacity / Not Normal"
  group (≈ 57% of RSNA negatives), which RSNA labels negative **by design**. They are not mislabelled under the dataset's own definition.
- The labels are image-based radiologist reads; no clinical records, cultures or reports are available, so pneumonia status cannot be verified. Hospital A's (Kermany) labels have a different
  meaning (paediatric clinical diagnosis). `CLAUDE.md` decision 13 already chose to keep the native labels project-wide.
- The earlier filtered retrain was data filtering, not label correction; its full-test AUROC/specificity drop is the expected result of never seeing those images, and it was compared at a fixed 0.5 threshold,
  which confounds threshold and class balance (not yet separated).

Recommendation given: do **not** retrain on "corrected" labels (no verified corrections exist; it would invent labels or reproduce the filtered set). Do a narrow ambiguity analysis instead:
(1) provenance note, (2) versioned ambiguity manifest tagging each image Normal / hard negative / Lung Opacity from the original `class` field with originals untouched, (3) leakage audit adding
exact-duplicate image hashes across splits (patient-level separation is already tested), (4) a subgroup-aware evaluation protocol (pneumonia vs truly normal, vs hard negatives, full test primary),
(5) re-analysis of the earlier filtered retrain with validation-chosen thresholds and calibration (cheap on the GPU), (6) a note on what evidence would justify label changes (expert re-adjudication or clinical records).
Optional and needing approval: training with the dataset's own 3-way class as an auxiliary target (changes the output head; `CLAUDE.md` §7 covers binary classification only).

Realistic benefit stated to the owner: it improves defensibility, not accuracy — it answers the likely reviewer objection ("wrong labels or weak model?"), turns the limitation into a reported
subgroup result, explains the earlier filtered retrain, and adds a duplicate-leakage check; it will not raise the headline AUROC and cannot show whether label correction helps generalization.
It ranks below external validation, a gradient-inversion attack and per-ε recipe re-tuning. Minimal version suggested (under an hour): items 1, 2 and 4. **No decision from the owner yet.**

## 7. Open items after this point [OPEN]

- Decision on the RSNA ambiguity work (full, minimal, or none) and on the Table II deployed-recipe row/footnote.
- Read PriMIA's full paper and confirm its row in the coverage table; confirm pages for Shokri and Bonawitz.
- IEEE Access needs its official Word/LaTeX template; corresponding e-mail and the "no external funding" line were assumptions.
- Scientific limitations unchanged: simulated hospitals (3 clients), ε per hospital/run and excluding the study's many runs and validation selection, weak privacy-attack evidence, DP recipe tuned once (ε = 4, one seed),
  no external validation, tuned recipe's calibration not measured, Docker verification one six-round CPU run.
- `metrics_final.pdf` has no related-work section.
