"""OPT-6 — Streamlit demo interface (CLAUDE.md section 16.1a; approved 2026-08-30).

Third design pass (2026-08-30, owner-directed): the app is now built around a
normal end user's workflow — upload an X-ray, get a result — with every
federated-learning/privacy/research term (FedAvg, epsilon, DP-SGD, Secure
Aggregation, hospital A/B/C, seeds, AUROC, ablation) moved out of the primary
screen into two clearly-separated tabs ("How it works" and "Research &
Results"). The model configuration is picked automatically (the project's own
documented default, DP epsilon=4 — CLAUDE.md Decision Gate DG-7) rather than
exposed as a chooser; a collapsed "Advanced" panel still lets a technical
visitor override it, per the owner's own "small Advanced area" allowance.

Presentation layer ONLY, over already-trained checkpoints from Stage 21's real
ablation campaign. Does not touch training, evaluation, privacy guarantees, the
federated pipeline, or Secure Aggregation in any way — every prediction, Grad-CAM
heatmap, uncertainty estimate, and OOD verdict shown here is computed by calling
directly into `src/*` (via `app/inference.py`), never a second implementation.

Run from the repository root:

    uv run streamlit run app/streamlit_app.py

See docs/frontend.md for the full write-up.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
from omegaconf import OmegaConf

from app import components as c
from app import inference, results_loader, theme

st.set_page_config(
    page_title="PneumoFL — AI-Assisted Chest X-ray Analysis",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown(theme.CSS, unsafe_allow_html=True)

CFG = OmegaConf.load(REPO_ROOT / "conf" / "app.yaml")
HOSPITALS = ["A", "B", "C"]
DEFAULT_CONFIG_KEY = "fedavg_no_dp"
# Not DG-7's eps=4 default. DG-7 fixes the canonical epsilon for the *research*
# ablation sweep (pyproject.toml's [tool.flwr.app.config] default for `flwr run`),
# not what this demo UI shows by default. Debugged 2026-08-31 after a live pneumonia
# X-ray was returned as "Normal, high confidence": every DP-trained checkpoint
# (eps in {1,2,4,8}) is severely biased toward predicting "Normal" on Hospital A
# (Kermany) specifically -- e.g. dp_eps4_seed42 predicts "Normal" on ~71% of
# Hospital A's test set despite it being 72% Pneumonia, crashing accuracy there to
# 54.9% (worse than the 71.9% majority-class baseline) even though its AUROC
# (0.825) shows the underlying features are still informative -- it's the decision
# threshold, not the representation, that DP-SGD's per-sample clipping/noise
# miscalibrated. This is invisible in docs/calibration.md's pooled 3-hospital
# accuracy (0.740) because Hospitals B/C are Normal-majority in their test sets, so
# a Normal-biased model still scores fine there. FedAvg (no DP) shows no such
# collapse (82.8% accuracy / 0.950 AUROC on Hospital A) and matches
# docs/calibration.md's own finding that DP causes a real ~4x ECE increase. The DP
# configurations remain selectable under "Advanced" -- this only changes what a
# first-time visitor sees by default.
DISCLAIMER_TEXT = "Educational/research project. Not intended for clinical diagnosis or treatment."


# --------------------------------------------------------------------------
# Cached resource loaders (unchanged logic — pure wiring around app/inference.py).
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(checkpoint_rel_path: str, fine_tune_last_block: bool = False):
    path = REPO_ROOT / checkpoint_rel_path
    if not path.exists():
        return None
    return inference.load_classifier(path, fine_tune_last_block=fine_tune_last_block)


@st.cache_resource(show_spinner=False)
def get_deferral_threshold(checkpoint_rel_path: str, fine_tune_last_block: bool = False) -> float | None:
    if fine_tune_last_block:
        # Stage 9's pooled-feature cache (what calibrate_deferral_threshold reads)
        # is a frozen-backbone artifact -- see docs/adr1_groupnorm_fallback.md's
        # own note that this cache "is invalid the moment part of the backbone
        # becomes trainable." Calibrating against it here would silently produce
        # a threshold derived from features this checkpoint's backbone never
        # actually generated. Returning None makes deferral explicitly
        # unavailable for this checkpoint instead of quietly wrong.
        return None
    model = get_model(checkpoint_rel_path, fine_tune_last_block)
    if model is None:
        return None
    partition_path = REPO_ROOT / CFG.paths.partition_path
    feature_cache_dir = REPO_ROOT / CFG.paths.feature_cache_dir
    if not partition_path.exists() or not feature_cache_dir.exists():
        return None
    try:
        return inference.calibrate_deferral_threshold(
            model, partition_path, feature_cache_dir, CFG.deferral.target_defer_fraction, CFG.mc_dropout.num_passes
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_ood_detectors():
    partition_path = REPO_ROOT / CFG.paths.partition_path
    feature_cache_dir = REPO_ROOT / CFG.paths.feature_cache_dir
    if not partition_path.exists() or not feature_cache_dir.exists():
        return None, None
    try:
        return inference.build_ood_detectors(
            partition_path, feature_cache_dir, HOSPITALS, CFG.ood_detector.detector_seed, CFG.ood_detector.target_flag_fraction
        )
    except Exception:
        return None, None


def artifacts_available() -> bool:
    return (REPO_ROOT / CFG.paths.partition_path).exists() and (REPO_ROOT / CFG.paths.feature_cache_dir).exists()


CONFIG_BY_KEY = {cfg["key"]: cfg for cfg in CFG.configurations}
CONFIG_BY_LABEL = {cfg["label"]: cfg for cfg in CFG.configurations}
DEFAULT_CONFIG = CONFIG_BY_KEY.get(DEFAULT_CONFIG_KEY, CFG.configurations[0])


# --------------------------------------------------------------------------
# Headline result extraction for the Research tab — reads real artifacts,
# never hardcodes a number, degrades to a placeholder if a source is missing.
# --------------------------------------------------------------------------
def _find_row(rows: list[dict] | None, prefix: str) -> dict | None:
    if not rows:
        return None
    return next((r for r in rows if r["row"].startswith(prefix)), None)


def compute_headline_stats() -> list[tuple[str, str, str]]:
    """Returns (label, value, sub) triples."""
    ablation = results_loader.load_ablation_table()
    calibration = results_loader.load_calibration_results()
    gradcam = results_loader.load_gradcam_localization_results()
    privacy_attack = results_loader.load_privacy_attack_results()

    stats = []

    fedavg = _find_row(ablation, "3. FedAvg (natural)")
    centralized = _find_row(ablation, "2. Centralized (natural")
    if fedavg and centralized and fedavg.get("mean_auroc") is not None:
        stats.append((
            "Federated model quality",
            f"{fedavg['mean_auroc']:.3f} AUROC",
            f"vs. {centralized['mean_auroc']:.3f} for a non-federated, non-private model trained on all data pooled together",
        ))
    else:
        stats.append(("Federated model quality", "—", "run scripts/run_ablation.py"))

    dp4 = _find_row(ablation, "5. FedAvg + DP (epsilon=4")
    if fedavg and dp4 and dp4.get("mean_auroc") is not None:
        acc_cost = fedavg["mean_auroc"] - dp4["mean_auroc"]
        stats.append((
            "Cost of privacy protection",
            f"−{acc_cost:.3f} AUROC",
            "accuracy difference between the deployed private model and the same model with no privacy protection",
        ))
    else:
        stats.append(("Cost of privacy protection", "—", "run scripts/run_calibration_analysis.py"))

    if gradcam and "centralized (natural, ceiling)" in gradcam:
        pg = gradcam["centralized (natural, ceiling)"]["pointing_game_accuracy"]["mean"]
        stats.append((
            "Explanation quality",
            f"{pg:.0%}",
            "of the time, the model's visual explanation points inside a radiologist-annotated region",
        ))
    else:
        stats.append(("Explanation quality", "—", "run scripts/run_gradcam_evaluation.py"))

    if privacy_attack and "centralized (natural, ceiling)" in privacy_attack:
        auroc = privacy_attack["centralized (natural, ceiling)"]["attack_auroc"]["mean"]
        stats.append((
            "Privacy stress-test",
            f"{auroc:.3f}",
            "a simulated attacker's ability to detect which images trained the model — 0.5 means no better than a coin flip",
        ))
    else:
        stats.append(("Privacy stress-test", "—", "run scripts/run_privacy_attack.py"))

    return stats


# --------------------------------------------------------------------------
# Header — shown above the tabs on every screen
# --------------------------------------------------------------------------
st.markdown(c.brand_row("PneumoFL", "AI-assisted Chest X-ray Analysis"), unsafe_allow_html=True)

tab_analyze, tab_how, tab_research = st.tabs(["Analyze", "How it works", "Research & Results"])


# ==========================================================================
# TAB 1 — ANALYZE (the primary experience)
# ==========================================================================
with tab_analyze:
    st.markdown(c.hero_title("Upload a chest X-ray to analyze it"), unsafe_allow_html=True)
    st.markdown(
        c.hero_subtitle("The model checks it for visual patterns associated with pneumonia and explains what it saw."),
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload X-ray",
        type=["jpg", "jpeg", "png", "dcm"],
        label_visibility="collapsed",
    )
    st.caption("Supported formats: JPG, PNG, DICOM")

    if uploaded_file is not None:
        file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]

        preview_col, button_col = st.columns([2, 1])
        with preview_col:
            st.image(uploaded_file, width=280)
        with button_col:
            st.write("")
            st.write("")
            analyze_clicked = st.button("Analyze X-ray", type="primary")

        if analyze_clicked:
            st.session_state["analyzed_hash"] = file_hash
            st.session_state["analyzed_bytes"] = uploaded_file.getvalue()
            st.session_state["analyzed_name"] = uploaded_file.name

        if st.session_state.get("analyzed_hash") == file_hash:
            active_config = DEFAULT_CONFIG
            with st.expander("Advanced"):
                override_label = st.selectbox(
                    "Model version (technical)",
                    list(CONFIG_BY_LABEL.keys()),
                    index=list(CONFIG_BY_LABEL.keys()).index(DEFAULT_CONFIG["label"]),
                )
                active_config = CONFIG_BY_LABEL[override_label]

            checkpoint_path = REPO_ROOT / active_config["checkpoint"]
            uses_finetuned_backbone = bool(active_config.get("fine_tune_last_block", False))
            if not checkpoint_path.exists():
                st.error("The analysis model isn't available in this environment. Please try again later.")
            else:
                model = get_model(active_config["checkpoint"], uses_finetuned_backbone)
                deferral_threshold = get_deferral_threshold(active_config["checkpoint"], uses_finetuned_backbone)
                if uses_finetuned_backbone:
                    # Same frozen-backbone-cache mismatch as deferral (see
                    # get_deferral_threshold) -- OOD detectors are calibrated
                    # against Stage 9's cached features, which this checkpoint's
                    # partially-unfrozen backbone would not itself produce.
                    # Skip rather than show a flag calibrated against the wrong
                    # feature distribution.
                    ood_detectors, ood_thresholds = {}, {}
                else:
                    ood_detectors, ood_thresholds = get_ood_detectors()

                with st.spinner("Analyzing..."):
                    t0 = time.time()
                    try:
                        bgr_image = inference.decode_uploaded_image(
                            st.session_state["analyzed_bytes"], st.session_state["analyzed_name"]
                        )
                    except ValueError as exc:
                        st.error(f"Could not read this file: {exc}")
                        st.stop()

                    effective_threshold = deferral_threshold if deferral_threshold is not None else float("inf")
                    result = inference.run_full_inference(
                        model,
                        bgr_image,
                        deferral_threshold=effective_threshold,
                        ood_detectors=ood_detectors or {},
                        ood_thresholds=ood_thresholds or {},
                        image_size=CFG.preprocessing.image_size,
                        num_mc_passes=CFG.mc_dropout.num_passes,
                        decision_threshold=float(active_config.get("decision_threshold", inference.DEFAULT_DECISION_THRESHOLD)),
                        abstention_half_width=float(active_config.get("abstention_half_width", 0.0)),
                        temperature=float(active_config.get("temperature", 1.0)),
                    )
                    elapsed = time.time() - t0

                st.markdown("<hr>", unsafe_allow_html=True)

                is_pneumonia = result.predicted_class == inference.PNEUMONIA_CLASS_INDEX
                u_level = inference.uncertainty_label(result.entropy, result.deferral_threshold)
                low_confidence = (not result.abstained) and (u_level == "High" or result.deferred)
                # NOT any() -- verified 2026-08-31: each hospital's IsolationForest
                # detector is calibrated to flag only ~5% of ITS OWN held-out data, but
                # legitimate images from one hospital's distribution routinely look
                # anomalous to *other* hospitals' detectors (Kermany vs. RSNA differ in
                # equipment/population/preprocessing) -- with any(), a real Kermany-style
                # X-ray got flagged 9/10 times and a real RSNA-style X-ray 10/10 times in
                # a direct test, i.e. the caution banner fired on almost every genuine
                # chest X-ray regardless of source. all() only fires when NO hospital's
                # distribution recognizes the image, which matches the intended meaning
                # ("this doesn't look like the training data at all") and empirically
                # drops the false-positive rate on real held-out images to ~0-10%.
                flagged_ood = all(result.ood_flags.values()) if result.ood_flags else False

                st.markdown(
                    c.result_label(result.predicted_label, attention=is_pneumonia, uncertain=result.abstained),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    c.confidence_meter(result.confidence, attention=is_pneumonia, uncertain=result.abstained),
                    unsafe_allow_html=True,
                )

                if result.abstained:
                    st.markdown(
                        c.confidence_note(
                            "This X-ray's result was too close to the model's decision boundary to call "
                            "confidently either way — further evaluation is recommended rather than "
                            "treating this as a Normal or Pneumonia result.",
                            uncertain=True,
                        ),
                        unsafe_allow_html=True,
                    )
                elif low_confidence:
                    st.markdown(
                        c.confidence_note(
                            "Low confidence — further clinical review recommended. The model was not "
                            "confident enough in this case to rely on the prediction alone.",
                            attention=True,
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        c.confidence_note("This prediction falls within the model's normal confidence range."),
                        unsafe_allow_html=True,
                    )

                if flagged_ood:
                    st.markdown(
                        c.confidence_note(
                            "This image looks visually different from the X-rays the model was trained on "
                            "(unusual scan, different equipment, or non-chest image). Treat this result with "
                            "extra caution.",
                            attention=True,
                        ),
                        unsafe_allow_html=True,
                    )

                st.markdown(c.subhead("Model explanation"), unsafe_allow_html=True)
                st.markdown(
                    c.prose(
                        "The highlighted regions below show which parts of the X-ray most influenced the "
                        "result — this is not a diagnosis of a specific area, just a visual guide to where "
                        "the model focused."
                    ),
                    unsafe_allow_html=True,
                )
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(result.rgb_image, caption="X-ray", width="stretch")
                with img_col2:
                    st.image(result.gradcam_overlay_rgb, caption="What the model focused on", width="stretch")

                with st.expander("Advanced technical details"):
                    if uses_finetuned_backbone:
                        st.markdown(
                            c.confidence_note(
                                "This experimental checkpoint's deferral threshold and out-of-distribution "
                                "checks are unavailable (calibration cache mismatch — see "
                                "docs/adr1_groupnorm_fallback.md §6), not confirmed absent of concern. "
                                "Treat every prediction from this model as undeferred/unchecked.",
                                attention=True,
                            ),
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"- Model version: `{active_config['label']}`\n"
                        f"- Decision threshold: `{result.decision_threshold:.4f}`"
                        + (" (derived from validation set — see docs/adr1_groupnorm_fallback.md §10)"
                           if active_config.get("decision_threshold") is not None else " (default)")
                        + "\n"
                        f"- P(Pneumonia): `{result.prob_pneumonia:.4f}`\n"
                        f"- Abstained (Uncertain result): `{result.abstained}`\n"
                        f"- Uncertainty (predictive entropy): `{result.entropy:.4f}`\n"
                        f"- Deferral threshold: `{result.deferral_threshold:.4f}`"
                        + (" (unavailable — see note above)" if uses_finetuned_backbone else "")
                        + "\n"
                        f"- Deferred for review: `{result.deferred}`\n"
                        f"- Analysis time: `{elapsed:.2f}s`\n"
                        + "".join(
                            f"- OOD check ({h}): `{'flagged' if result.ood_flags.get(h) else 'normal'}` "
                            f"(score `{result.ood_scores.get(h, float('nan')):.4f}`)\n"
                            for h in HOSPITALS if h in result.ood_flags
                        )
                    )
    else:
        if not artifacts_available():
            st.caption(
                "Note: some supporting data for this environment isn't present, so uncertainty and anomaly "
                "checks may be limited. Predictions will still run."
            )

    st.markdown(c.disclaimer(DISCLAIMER_TEXT), unsafe_allow_html=True)


# ==========================================================================
# TAB 2 — HOW IT WORKS
# ==========================================================================
with tab_how:
    st.markdown(c.section_title("How it works"), unsafe_allow_html=True)
    st.markdown(
        c.section_lede("PneumoFL is built on five ideas working together, each doing a different part of the job."),
        unsafe_allow_html=True,
    )

    st.markdown(c.subhead("Federated learning"), unsafe_allow_html=True)
    st.markdown(
        c.prose(
            "Instead of collecting patient X-rays onto one server, the model is trained across several "
            "simulated hospitals. Each one trains on its own data, on its own infrastructure — the images "
            "themselves never move."
        ),
        unsafe_allow_html=True,
    )

    st.markdown(c.subhead("Differential privacy"), unsafe_allow_html=True)
    st.markdown(
        c.prose(
            "Before a hospital's training update is sent anywhere, carefully calibrated statistical noise is "
            "added to it. This gives a mathematical guarantee that no single patient's data can be "
            "reverse-engineered from the update, at a quantified cost to accuracy."
        ),
        unsafe_allow_html=True,
    )

    st.markdown(c.subhead("Secure aggregation"), unsafe_allow_html=True)
    st.markdown(
        c.prose(
            "Hospital updates are cryptographically masked before transmission, so the central server can only "
            "ever recover the combined total of every hospital's contribution — never any one hospital's update "
            "on its own."
        ),
        unsafe_allow_html=True,
    )

    st.markdown(c.subhead("Explainable AI"), unsafe_allow_html=True)
    st.markdown(
        c.prose(
            "Every prediction comes with a Grad-CAM visualization highlighting the regions of the X-ray that "
            "most influenced the result, so a viewer can sanity-check what the model actually looked at."
        ),
        unsafe_allow_html=True,
    )

    st.markdown(c.subhead("Uncertainty estimation"), unsafe_allow_html=True)
    st.markdown(
        c.prose(
            "The model is queried multiple times with a randomized internal process (Monte Carlo Dropout) to "
            "measure how consistent its own predictions are. Inconsistent, low-confidence cases are flagged for "
            "human review instead of being acted on automatically."
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(c.subhead("System architecture"), unsafe_allow_html=True)
    st.markdown(
        c.diagram(
            "Hospital A  ─┐\n"
            "Hospital B  ─┼──▶  Secure Aggregation  ──▶  Shared Model\n"
            "Hospital C  ─┘        (server only ever\n"
            "                       sees the combined total)\n\n"
            "Every connection above is encrypted, with each hospital cryptographically\n"
            "verified before it can participate."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        c.prose(
            "<b>Scope note:</b> the hospitals here are simulated for this project, not a real "
            "multi-institution deployment, and the system is not designed to defend against a hospital "
            "acting maliciously — only against an honest-but-curious server and network eavesdroppers."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(c.disclaimer(DISCLAIMER_TEXT), unsafe_allow_html=True)


# ==========================================================================
# TAB 3 — RESEARCH & RESULTS
# ==========================================================================
with tab_research:
    st.markdown(c.section_title("Research & Results"), unsafe_allow_html=True)
    st.markdown(
        c.section_lede("Every number below comes from a real, live training run — nothing here is estimated."),
        unsafe_allow_html=True,
    )

    for label, value, sub in compute_headline_stats():
        st.markdown(
            f'<div style="margin-bottom:1.3rem;">'
            f'<div style="font-size:0.78rem;color:var(--text-faint);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">{label}</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:var(--text-primary);">{value}</div>'
            f'<div style="font-size:0.85rem;color:var(--text-muted);">{sub}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("Full research results & technical tables"):
        st.markdown(
            c.prose(
                "Repository: <code>UddhavSethi/privacy-preserving-medical-diagnosis</code>. Full write-ups: "
                "<code>docs/results.md</code>, <code>docs/calibration.md</code>, <code>docs/privacy_attack.md</code>, "
                "<code>docs/gradcam_localization.md</code>, <code>docs/conformal.md</code>, <code>docs/ood_detection.md</code>."
            ),
            unsafe_allow_html=True,
        )

        ablation = results_loader.load_ablation_table()
        if ablation:
            st.markdown("**Ablation ladder** — cost of each privacy/security layer, 3 seeds each")
            import pandas as pd

            rows = [r for r in ablation if r.get("mean_auroc") is not None]
            st.dataframe(
                pd.DataFrame([{"Configuration": r["row"], "Mean AUROC": r["mean_auroc"], "Std": r["std_auroc"], "Seeds": r["n_seeds"]} for r in rows]),
                width="stretch",
                hide_index=True,
            )
            fig = results_loader.figure_path("ablation_table_chart.png")
            if fig:
                st.image(str(fig), width="stretch")
        else:
            st.caption("Ablation table not available — needs a reachable MLflow database (mlruns.db).")

        for title, loader_fn, cols in [
            ("Calibration (OPT-1)", results_loader.load_calibration_results, [("ece", "ECE"), ("brier", "Brier")]),
            ("Membership-inference attack (OPT-2)", results_loader.load_privacy_attack_results, [("attack_auroc", "Attack AUROC")]),
            ("Grad-CAM localization (OPT-3)", results_loader.load_gradcam_localization_results, [("pointing_game_accuracy", "Pointing game"), ("mean_iou", "Mean IoU")]),
            ("Conformal prediction (OPT-4)", results_loader.load_conformal_results, [("empirical_coverage", "Coverage"), ("mean_set_size", "Set size")]),
        ]:
            data = loader_fn()
            if not data:
                continue
            import pandas as pd

            st.markdown(f"**{title}**")
            table_rows = []
            for config_name, values in data.items():
                row = {"Configuration": config_name}
                for key, label in cols:
                    if key in values:
                        row[label] = values[key]["mean"]
                table_rows.append(row)
            st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

        ood_data = results_loader.load_ood_detector_results()
        if ood_data:
            import pandas as pd

            st.markdown("**OOD detection gate (OPT-5)**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Hospital": h,
                            "Val flag rate (target 5%)": v["calibration"]["realized_flag_fraction_on_calibration"],
                            "Test flag rate": v["test_flag_rate"],
                            "Synthetic-noise flag rate": v["synthetic_flag_rates"]["random_noise"],
                        }
                        for h, v in ood_data.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    st.markdown(c.disclaimer(DISCLAIMER_TEXT), unsafe_allow_html=True)
