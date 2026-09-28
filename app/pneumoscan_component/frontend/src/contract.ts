// PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Phase 1C/1D) —
// TypeScript mirror of `app/presentation.py::build_props`'s real output
// shape. Every nullable field here is nullable because the real backend
// genuinely may not have produced a value (missing calibration artifacts, a
// non-Pneumonia label, no OOD detectors in this environment, ...) — render
// "Not available" for `null`, never substitute a default. Do not invent a
// field that isn't listed here; if the UI needs something the backend
// doesn't send, that's a backend change, not a client-side default.
export const PROTOCOL_VERSION = 1;

export type Phase = "upload" | "quality" | "analyzing" | "review" | "rejected" | "error";

export interface StudyProps {
  study_id: string;
  filename: string;
  format: string; // "JPEG" | "PNG" | "DICOM" | ...
  width: number;
  height: number;
  file_size_bytes: number | null;
  projection: string | null;
  loaded_at: string; // ISO datetime
  patient: null; // always null — never collected (app/presentation.py::study_props)
  is_sample: boolean;
}

export interface XrayGateProps {
  status: "unavailable" | "passed" | "failed";
  p_xray: number | null;
}

export interface QualityProps {
  decoded: boolean;
  format_supported: boolean;
  min_side_px: number;
  resolution_ok: boolean;
  grayscale: boolean;
  xray_gate: XrayGateProps;
}

export type JobStageId = "gate" | "preprocess" | "mc_dropout" | "ood" | "gradcam";
export type JobStageStatus = "pending" | "running" | "done";

export interface ProgressProps {
  job_id: number;
  queued: boolean;
  stages: Array<{ id: JobStageId; status: JobStageStatus }>;
  fraction: number; // 0..1, done-stage count / total stages
}

export interface OodSite {
  flagged: boolean;
  score: number;
}

export interface Focus {
  side: "left" | "right";
  zone: "upper" | "middle" | "lower";
  u: number; // 0..1, image-space x
  v: number; // 0..1, image-space y
}

export interface ResultProps {
  label: "Normal" | "Pneumonia" | "Uncertain";
  confidence: number; // 0..1
  prob_pneumonia: number; // 0..1
  entropy: number;
  abstained: boolean;
  certainty: "High" | "Medium" | "Low" | null;
  deferred: boolean | null;
  deferral_threshold: number | null;
  image_check: "Typical" | "Unusual" | null;
  ood_sites: Record<string, OodSite> | null; // keyed by hospital id, e.g. "A"/"B"/"C"
  gradcam_target: "Normal" | "Pneumonia";
  focus: Focus | null;
  focus_spread: boolean;
  decision_threshold: number;
  decision_threshold_source: "validation" | "default";
  temperature: number | null;
  abstention_half_width: number | null;
}

export interface PneumoScanData {
  protocol_version: number;
  phase: Phase;
  study: StudyProps | null;
  quality: QualityProps | null;
  progress: ProgressProps | null;
  result: ResultProps | null;
  error: string | null;
}

// The full `args` object a Streamlit v1 component receives
// (`RenderData.args`, i.e. `ComponentProps["args"]`). `display_image` and
// `heatmap` arrive as Streamlit "special args" (real binary, not
// base64/JSON — see `custom_component.py`'s `is_bytes_like` branch) whenever
// the backend has one; Streamlit's component bridge may hand these to us as
// either `Uint8Array` or a plain `ArrayBuffer` depending on version, so
// callers must go through `toUint8Array` (bridge.ts) rather than assuming
// one or the other.
export interface PneumoScanArgs {
  data: PneumoScanData;
  display_image: Uint8Array | ArrayBuffer | null;
  heatmap: Uint8Array | ArrayBuffer | null;
}

// JS -> Python events (Phase 1D). `event_id` must be unique per event (the
// Python side de-dupes on it, since a v1 component's last value persists
// across unrelated reruns/polls) — bridge.ts's `sendEvent` generates this.
export type OutboundEventType = "upload" | "use_sample" | "analyze" | "new_screening" | "dismiss_error";

export interface UploadPayload {
  name: string;
  mime: string;
  size: number;
  data_b64: string;
}

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

export const JOB_STAGE_LABELS: Record<JobStageId, string> = {
  gate: "Checking image",
  preprocess: "Preprocessing",
  mc_dropout: "AI analysis",
  ood: "Image comparison",
  gradcam: "AI highlight",
};
