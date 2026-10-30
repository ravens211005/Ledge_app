"""Semantic-depth hazard analysis: YOLO floor mask x Depth Anything V2 relative depth.

Flags regions inside an (eroded) floor/path mask whose depth shows:
  1. "void"  - values far beyond the local floor level (unbounded drop-off)
  2. "hole"  - NaN / non-finite / zero depth (model failed to compute a surface)
  3. "step"  - abnormally large downward step along the image Y axis (Sobel-Y)

Flagged regions are then given a tier:
  CRITICAL  - a depth-level anomaly (void/hole) AND a vertical step agree, and the region is
              large or reaches low in the frame (near the camera).
  POSSIBLE  - everything else that was flagged.

IMPORTANT: Depth Anything V2 (standard checkpoint) outputs RELATIVE INVERSE depth
(larger = closer, arbitrary scale per image). All thresholds below are therefore expressed in
robust-sigma units of the floor's own depth statistics, never in metres. Research prototype:
not validated as a safety-critical system.

Run standalone:  python hazard_analysis.py path\\to\\image.jpg [--out result.png]
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

import config

BoolMap = NDArray[np.bool_]
FloatMap = NDArray[np.float32]
Frame = NDArray[np.uint8]

# Which YOLO class names count as the "floor / walkable path" mask. Configurable on purpose:
# the trained model only has 'edges' and 'stairs'. Using 'stairs' makes check 3 fire on real steps.
FLOOR_CLASS_NAMES: tuple[str, ...] = (config.STAIR_CLASS_NAME,)

COLOR_CRITICAL = (0, 0, 255)      # BGR bright red
COLOR_POSSIBLE = (0, 165, 255)    # BGR orange


# =============================================================================================
# CONFIGURATION  (tune here; every threshold is documented)
# =============================================================================================
@dataclass(frozen=True)
class HazardConfig:
    """Tunable settings. Defaults are untested starting points, NOT validated values.

    How to tune for close-range walking (roughly 1-3 m ahead):
      * Relative depth cannot be converted to metres, so "1-3 m" is approximated geometrically.
        Place markers 1 m and 3 m ahead on flat floor, capture a frame from the real camera
        mounting, and note which image rows they fall on. Set `near_field_row_start_frac` to the
        row of the 3 m marker divided by frame height (rows above it are ignored), and
        `critical_min_bottom_frac` to the row of roughly the 2 m marker.
      * Collect frames of (a) flat safe floor and (b) real ledges / down-steps. Raise a *_k value
        until (a) produces no hazard pixels, then check (b) is still caught. Record the false
        positives and misses you measure; do not quote accuracy you have not measured.
      * If it flags the far end of ordinary floors, raise `void_k`; if it misses shallow drops,
        lower `step_k`; if edges of walls trigger alarms, raise `erosion_px`.
      * Tiers: if too many regions are CRITICAL, raise `critical_min_area_frac` or
        `critical_min_bottom_frac`; if real drops stay POSSIBLE, lower them.
    """

    # --- Alignment / speed -------------------------------------------------------------------
    analysis_max_side: int = 640        # analysis runs at this size; results are scaled back up

    # --- Anti-noise preprocessing (units: pixels at analysis resolution) ----------------------
    erosion_px: int = 5                 # shrink floor mask to drop edge bleed / misalignment.
                                        # Trade-off: also removes a ledge sitting on the mask boundary
    near_field_row_start_frac: float = 0.30   # ignore rows above this fraction of frame height
    min_floor_area_frac: float = 0.01   # below this much eroded floor area -> "uncertain"

    # --- Depth convention --------------------------------------------------------------------
    depth_is_inverse: bool = True       # True for Depth Anything V2 relative output (larger = closer)

    # --- Floor baseline: robust polynomial fit of floor depth vs image row (v-disparity) ------
    # For a flat floor, inverse depth varies smoothly (roughly linearly) with image row. The fit is
    # robust (Tukey bisquare), so a drop-off occupying a minority of the rows is treated as an
    # outlier instead of bending the baseline. A per-row median cannot do this: a ledge that spans
    # the full width of a row would simply become that row's baseline and be invisible.
    baseline_degree: int = 2
    baseline_iters: int = 6

    # --- Check 1: unbounded void (far-direction residual vs the fitted floor baseline) --------
    void_k: float = 6.0                 # flag if farther than baseline by > void_k robust sigmas
    void_scale_floor_frac: float = 0.02 # sigma never below this fraction of the floor depth range

    # --- Check 2: data holes -----------------------------------------------------------------
    zero_eps: float = 1e-6              # depth <= eps counts as a hole (ReLU zeros = "very far")

    # --- Check 3: downward step via Sobel-Y --------------------------------------------------
    step_k: float = 8.0                 # flag if drop-gradient exceeds floor median by > step_k sigmas
    step_scale_floor_frac: float = 0.004  # gradient sigma floor, as fraction of depth range per pixel
    pre_blur_ksize: int = 5             # median blur before Sobel (3 or 5; 0 disables)

    # --- Clean-up ----------------------------------------------------------------------------
    open_ksize: int = 0                 # opening kernel (0 disables). Opening erased thin 2-3 px step
                                        # bands, so it is off by default; min_blob_area_frac removes speckle
    min_blob_area_frac: float = 0.002   # discard blobs smaller than this fraction of floor area

    # --- Tiering -----------------------------------------------------------------------------
    # Evidence groups: group A = void or hole (depth-level anomaly), group B = step (gradient).
    # void and hole are correlated on Depth Anything V2 (ReLU zeros trigger both), so agreement
    # is counted between groups, not between individual checks.
    critical_min_groups: int = 2        # groups that must overlap on the region for CRITICAL
    critical_min_area_frac: float = 0.01      # region area as fraction of the whole frame ("large")
    critical_min_bottom_frac: float = 0.60    # region reaches this far down the frame ("near")

    # --- Image-quality guards (cheap uncertainty indicators) ---------------------------------
    min_mean_luma: float = 40.0         # darker than this -> uncertain
    min_laplacian_var: float = 30.0     # blurrier than this (at analysis size) -> uncertain

    # --- Rendering ---------------------------------------------------------------------------
    label: str = "CRITICAL FALL HAZARD"
    label_possible: str = "POSSIBLE DROP-OFF"
    overlay_alpha: float = 0.45


# =============================================================================================
# RESULT TYPES
# =============================================================================================
CHECK_DESCRIPTIONS = {
    "void": "unbounded void (depth far beyond local floor level)",
    "hole": "depth data hole (zero / NaN)",
    "step": "downward step-change (Sobel-Y)",
}


@dataclass
class HazardRegion:
    bbox: tuple[int, int, int, int]     # x, y, w, h in frame pixels
    area_px: int
    checks: tuple[str, ...]             # which of void / hole / step contributed
    tier: str = "possible"              # "critical" | "possible"
    tier_reason: str = ""


@dataclass
class HazardResult:
    status: str                         # "hazard" | "no_anomaly" | "uncertain"
    message: str
    hazard_map: BoolMap                 # merged binary hazard map (both tiers), frame size
    critical_map: BoolMap               # pixels belonging to CRITICAL regions, frame size
    check_maps: dict[str, BoolMap]      # per-check maps, frame size (for debugging)
    regions: list[HazardRegion]
    overlay_bgr: Frame                  # original frame with overlay + labels
    warnings: list[str] = field(default_factory=list)
    timing_ms: float = 0.0
    diag: dict = field(default_factory=dict)   # pixel counts at analysis resolution (for tuning)

    def summary_markdown(self) -> str:
        head = {"hazard": "🔴", "no_anomaly": "🟢", "uncertain": "🟡"}[self.status]
        parts = [f"{head} **{self.message}**"]
        if self.regions:
            bullets = []
            for i, r in enumerate(self.regions, 1):
                why = ", ".join(CHECK_DESCRIPTIONS[c] for c in r.checks)
                x, y, w, h = r.bbox
                tier = "CRITICAL" if r.tier == "critical" else "POSSIBLE"
                bullets.append(f"- Region {i} **[{tier}]**: box x={x}, y={y}, {w}x{h}px, "
                               f"area {r.area_px}px. Evidence: {why}. {r.tier_reason}")
            parts.append("\n".join(bullets))
        if self.warnings:
            parts.append("\n".join(f"- ⚠️ {w_}" for w_ in self.warnings))
        if self.diag:
            d = self.diag
            parts.append(f"_Diagnostics (analysis resolution): floor ROI {d.get('roi_px', 0)} px; raw anomaly "
                         f"pixels void {d.get('void_px', 0)} / hole {d.get('hole_px', 0)} / step "
                         f"{d.get('step_px', 0)}; kept after clean-up {d.get('kept_px', 0)}._")
        parts.append(f"_Hazard analysis took {self.timing_ms:.0f} ms. Research prototype: unvalidated, "
                     "depth is relative, not a safety guarantee._")
        return "\n\n".join(parts)


# =============================================================================================
# INPUT HELPERS
# =============================================================================================
def floor_mask_from_detections(detections: Iterable, class_names: Sequence[str],
                               frame_shape: tuple[int, ...]) -> BoolMap:
    """Union of the masks of all detections whose class name is in `class_names`."""
    mask = np.zeros(frame_shape[:2], dtype=bool)
    for d in detections:
        if d.class_name in class_names:
            mask |= d.mask
    return mask


def _validate(frame_bgr: np.ndarray, floor_mask: np.ndarray, depth: np.ndarray) -> None:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("frame_bgr must be an HxWx3 uint8 BGR array")
    if floor_mask.ndim != 2 or depth.ndim != 2:
        raise ValueError("floor_mask and depth must be 2-D arrays")


def _align(hw: tuple[int, int], mask: np.ndarray, depth: np.ndarray) -> tuple[BoolMap, FloatMap]:
    """Bring mask and depth to the frame's (H, W) if they come in at another resolution."""
    h, w = hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if depth.shape != (h, w):
        depth = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return mask.astype(bool), np.ascontiguousarray(depth, dtype=np.float32)


# =============================================================================================
# CHECKS (all vectorised; no per-pixel Python loops)
# =============================================================================================
def _robust_sigma(x: FloatMap) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _erode(mask: BoolMap, px: int) -> BoolMap:
    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def _row_median_profile(depth: FloatMap, roi: BoolMap, finite: BoolMap, min_px: int = 8) -> FloatMap:
    """Median floor depth per image row (NaN for rows with too few floor pixels)."""
    vals = np.where(roi & finite, depth, np.nan).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are expected
        prof = np.nanmedian(vals, axis=1)
    prof[(roi & finite).sum(axis=1) < min_px] = np.nan
    return prof.astype(np.float32)


def _robust_polyfit(x: np.ndarray, y: np.ndarray, degree: int, iters: int) -> np.ndarray:
    """Polynomial fit with Tukey-bisquare reweighting (tolerates a minority of outlier rows)."""
    coef = np.polyfit(x, y, degree)
    for _ in range(iters):
        r = y - np.polyval(coef, x)
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if s < 1e-9:
            break
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        keep = w > 0
        if int(keep.sum()) < degree + 2:
            break
        coef = np.polyfit(x[keep], y[keep], degree, w=np.sqrt(w[keep]))
    return coef


def _floor_baseline(depth: FloatMap, roi: BoolMap, finite: BoolMap, cfg: HazardConfig) -> FloatMap:
    """Expected floor depth per image row from a robust polynomial fit (NaN where no data)."""
    prof = _row_median_profile(depth, roi, finite)
    ok = np.isfinite(prof)
    base = np.full(depth.shape[0], np.nan, dtype=np.float32)
    if int(ok.sum()) < max(10, cfg.baseline_degree + 3):
        return base
    ys = np.nonzero(ok)[0].astype(np.float64)
    x = ys / max(1, depth.shape[0] - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # polyfit conditioning warnings
        coef = _robust_polyfit(x, prof[ok].astype(np.float64), cfg.baseline_degree, cfg.baseline_iters)
    base[ok] = np.polyval(coef, x).astype(np.float32)
    return base


def check_void(depth: FloatMap, roi: BoolMap, finite: BoolMap, depth_range: float,
               cfg: HazardConfig) -> BoolMap:
    """Check 1: pixels far beyond the fitted floor baseline (drop-off into empty space)."""
    base = _floor_baseline(depth, roi, finite, cfg)
    resid = depth - base[:, None]
    far = -resid if cfg.depth_is_inverse else resid        # positive = farther than the floor
    valid = roi & finite & np.isfinite(far)
    if int(valid.sum()) < 50:
        return np.zeros_like(roi)
    sigma = max(_robust_sigma(far[valid]), cfg.void_scale_floor_frac * depth_range)
    with np.errstate(invalid="ignore"):
        out = valid & (far > cfg.void_k * sigma)
    if not cfg.depth_is_inverse:                           # true-depth maps: +inf = infinite distance
        out |= roi & np.isposinf(depth)
    return out


def check_hole(depth: FloatMap, roi: BoolMap, finite: BoolMap, cfg: HazardConfig) -> BoolMap:
    """Check 2: zero / NaN / non-finite depth inside the floor mask."""
    with np.errstate(invalid="ignore"):
        return roi & (~finite | (depth <= cfg.zero_eps))


def check_step(depth_filled: FloatMap, roi: BoolMap, depth_range: float, cfg: HazardConfig) -> BoolMap:
    """Check 3: abnormally large downward step along Y (Sobel-Y on the full map).

    Moving down the image, a normal floor gets nearer at a steady rate. A ledge makes the surface
    beyond it suddenly much farther, i.e. an unusually large jump in the 'nearer going down'
    direction. Sobel runs on the whole depth map so values at the mask boundary are correct.
    """
    d = depth_filled
    if cfg.pre_blur_ksize in (3, 5):
        d = cv2.medianBlur(d, cfg.pre_blur_ksize)
    dy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)   # depth change per pixel
    drop = dy if cfg.depth_is_inverse else -dy
    ref = drop[roi]
    if ref.size < 50:
        return np.zeros_like(roi)
    med = float(np.median(ref))
    sigma = max(_robust_sigma(ref), cfg.step_scale_floor_frac * depth_range)
    return roi & ((drop - med) > cfg.step_k * sigma)


# =============================================================================================
# CLEAN-UP, TIERING, QUALITY GUARDS, RENDERING
# =============================================================================================
def _clean(anomaly: BoolMap, floor_area_px: int, cfg: HazardConfig) -> BoolMap:
    u8 = anomaly.astype(np.uint8)
    if cfg.open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_ksize, cfg.open_ksize))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    min_area = max(1, int(cfg.min_blob_area_frac * floor_area_px))
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False                                        # background
    return keep[labels]


def _assign_tier(checks: tuple[str, ...], area_px: int, bottom_y: int, h: int, w: int,
                 cfg: HazardConfig) -> tuple[str, str]:
    """Return (tier, plain-language reason) for one flagged region."""
    level = ("void" in checks) or ("hole" in checks)       # group A: depth-level anomaly
    step = "step" in checks                                # group B: vertical gradient step
    groups = int(level) + int(step)
    big = area_px >= cfg.critical_min_area_frac * h * w
    near = bottom_y >= cfg.critical_min_bottom_frac * h

    if groups < cfg.critical_min_groups:
        have = "depth-level anomaly only" if level else "vertical step only"
        return "possible", f"Only one kind of evidence ({have})."
    if not (big or near):
        return "possible", "Evidence agrees, but the region is small and high in the frame (far)."
    why = "large" if big else "reaches low in the frame (near)"
    return "critical", f"Depth-level anomaly and vertical step agree; region is {why}."


def _quality_warnings(frame_small_bgr: Frame, cfg: HazardConfig) -> list[str]:
    gray = cv2.cvtColor(frame_small_bgr, cv2.COLOR_BGR2GRAY)
    out: list[str] = []
    if float(gray.mean()) < cfg.min_mean_luma:
        out.append("Image is very dark; depth and segmentation may be unreliable.")
    if float(cv2.Laplacian(gray, cv2.CV_32F).var()) < cfg.min_laplacian_var:
        out.append("Image looks blurry or low-detail; depth and segmentation may be unreliable.")
    return out


def _resize_bool(m: BoolMap, w: int, h: int) -> BoolMap:
    if m.shape == (h, w):
        return m
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _draw_label(out: Frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    h, w = out.shape[:2]
    scale = max(0.5, w / 1000.0 * 0.8)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(np.clip(x, 0, max(0, w - tw - 6)))
    y_top = y - th - base - 6
    y_top = y_top if y_top >= 0 else min(h - th - base - 6, y + 4)
    cv2.rectangle(out, (x, y_top), (x + tw + 6, y_top + th + base + 6), color, -1)
    cv2.putText(out, text, (x + 3, y_top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


def render_overlay(frame_bgr: Frame, critical: BoolMap, possible: BoolMap,
                   regions: list[HazardRegion], cfg: HazardConfig) -> Frame:
    """Red overlay for CRITICAL regions, orange for POSSIBLE ones, each labelled."""
    out = frame_bgr.copy()
    for mask, color in ((possible, COLOR_POSSIBLE), (critical, COLOR_CRITICAL)):
        if not mask.any():
            continue
        tint = np.zeros_like(out)
        tint[:] = color
        blended = cv2.addWeighted(out, 1.0 - cfg.overlay_alpha, tint, cfg.overlay_alpha, 0)
        out = np.where(mask[..., None], blended, out).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 3, cv2.LINE_AA)
    for r in regions:
        crit = r.tier == "critical"
        _draw_label(out, cfg.label if crit else cfg.label_possible, r.bbox[0], r.bbox[1],
                    COLOR_CRITICAL if crit else COLOR_POSSIBLE)
    return out


# =============================================================================================
# MAIN ENTRY POINT
# =============================================================================================
def detect_fall_hazard(frame_bgr: Frame, floor_mask: np.ndarray, depth: np.ndarray,
                       cfg: HazardConfig | None = None) -> HazardResult:
    """Detect drop-off anomalies inside a floor mask using aligned relative depth.

    Args:
        frame_bgr: HxWx3 uint8 BGR camera frame.
        floor_mask: binary mask (any bool/uint8 array) of the floor / walkable path.
        depth: raw float32 Depth Anything V2 map (relative inverse depth). Resized to the frame
            if its shape differs.
        cfg: thresholds; see HazardConfig for tuning guidance.

    Returns:
        HazardResult. status is "uncertain" when the evidence is insufficient; "no_anomaly" is
        NOT a safety guarantee. Each flagged region is tiered "critical" or "possible".
    """
    t0 = time.perf_counter()
    cfg = cfg or HazardConfig()
    _validate(frame_bgr, floor_mask, depth)
    h, w = frame_bgr.shape[:2]
    mask, depth = _align((h, w), floor_mask, depth)

    # --- downscale for speed ---------------------------------------------------------------
    s = min(1.0, cfg.analysis_max_side / max(h, w))
    if s < 1.0:
        sw, sh = max(1, round(w * s)), max(1, round(h * s))
        m_s = _resize_bool(mask, sw, sh)
        d_s = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_AREA)
        f_s = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sw, sh, m_s, d_s, f_s = w, h, mask, depth, frame_bgr

    warns = _quality_warnings(f_s, cfg)
    diag: dict = {}

    def _finish(status: str, message: str | None = None, hazard_s: BoolMap | None = None,
                checks_s: dict[str, BoolMap] | None = None) -> HazardResult:
        empty = np.zeros((h, w), dtype=bool)
        hazard_full = _resize_bool(hazard_s, w, h) if hazard_s is not None else empty
        checks_full = {k: _resize_bool(v, w, h) for k, v in (checks_s or {}).items()}
        regions: list[HazardRegion] = []
        critical_full = empty
        if hazard_full.any():
            n, lab, stats, _ = cv2.connectedComponentsWithStats(hazard_full.astype(np.uint8), connectivity=8)
            hits = {k: np.bincount(lab[v & hazard_full], minlength=n) > 0 for k, v in checks_full.items()}
            is_critical = np.zeros(n, dtype=bool)
            for i in range(1, n):
                x, y, bw, bh, area = (int(v) for v in stats[i])
                checks = tuple(k for k in ("void", "hole", "step") if hits.get(k, np.zeros(n, bool))[i])
                tier, reason = _assign_tier(checks, area, y + bh, h, w, cfg)
                regions.append(HazardRegion((x, y, bw, bh), area, checks, tier, reason))
                is_critical[i] = tier == "critical"
            critical_full = is_critical[lab]
        possible_full = hazard_full & ~critical_full
        if status == "hazard":
            n_crit = sum(r.tier == "critical" for r in regions)
            message = f"{n_crit} critical and {len(regions) - n_crit} possible fall-hazard region(s) flagged"
        overlay = render_overlay(frame_bgr, critical_full, possible_full, regions, cfg)
        return HazardResult(status, message or "", hazard_full, critical_full, checks_full, regions,
                            overlay, warns, (time.perf_counter() - t0) * 1000.0, dict(diag))

    # --- anti-noise ROI: eroded floor mask, near-field rows only ---------------------------
    roi = _erode(m_s, cfg.erosion_px)
    roi[: int(sh * cfg.near_field_row_start_frac)] = False
    floor_px = int(roi.sum())
    diag["roi_px"] = floor_px
    if floor_px < cfg.min_floor_area_frac * sh * sw:
        return _finish("uncertain", "Uncertain — unable to assess reliably. "
                       "Floor region too small after erosion / near-field crop.")

    finite = np.isfinite(d_s)
    if int((roi & finite).sum()) < 0.5 * floor_px:
        return _finish("uncertain", "Uncertain — unable to assess reliably. "
                       "Depth map is invalid over most of the floor region.")

    fv = d_s[roi & finite]
    lo, hi = np.percentile(fv, (1, 99))
    depth_range = max(float(hi - lo), 1e-6)
    d_fill = np.where(finite, d_s, np.float32(np.median(fv))).astype(np.float32)

    # --- three vectorised checks -----------------------------------------------------------
    checks = {
        "void": check_void(d_s, roi, finite, depth_range, cfg),
        "hole": check_hole(d_s, roi, finite, cfg),
        "step": check_step(d_fill, roi, depth_range, cfg),
    }
    anomaly = checks["void"] | checks["hole"] | checks["step"]
    hazard_s = _clean(anomaly, floor_px, cfg)
    diag.update(void_px=int(checks["void"].sum()), hole_px=int(checks["hole"].sum()),
                step_px=int(checks["step"].sum()), kept_px=int(hazard_s.sum()))

    if hazard_s.any():
        return _finish("hazard", None, hazard_s, checks)
    if warns:
        return _finish("uncertain", "Uncertain — unable to assess reliably (image quality).",
                       hazard_s, checks)
    return _finish("no_anomaly", "No drop-off anomaly found by the three checks "
                   "(this is not a safety guarantee)", hazard_s, checks)


# =============================================================================================
# Stage 3 helpers kept for the results table
# =============================================================================================
TABLE_HEADERS = ["ID", "Class", "Conf", "Area %", "Rel. depth median", "p10", "p90",
                 "Box-band median (bottom 30%)"]


def region_stats(detections, depth_rel):
    """One row per detection. Depth values are RELATIVE (0..1 per image, larger = closer)."""
    h, w = depth_rel.shape
    rows = []
    for d in sorted(detections, key=lambda d: (d.class_name != config.STAIR_CLASS_NAME, -d.conf)):
        area_pct = 100.0 * d.mask.sum() / (h * w)
        vals = depth_rel[d.mask]
        if vals.size >= 20:
            p10, med, p90 = np.percentile(vals, [10, 50, 90])
            stats = [f"{med:.3f}", f"{p10:.3f}", f"{p90:.3f}"]
        else:
            stats = ["n/a", "n/a", "n/a"]
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        y_start = int(y2 - config.EDGE_DEPTH_BOX_BAND_FRACTION * (y2 - y1))
        band = depth_rel[max(0, y_start):max(0, y2), max(0, x1):max(0, x2)]
        band_med = f"{np.median(band):.3f}" if band.size >= 20 else "n/a"
        rows.append([d.idx, d.class_name, f"{d.conf:.2f}", f"{area_pct:.1f}", *stats, band_med])
    return rows


# =============================================================================================
# Standalone demo:  python hazard_analysis.py image.jpg [--out result.png]
# =============================================================================================
def _demo() -> None:
    from PIL import Image
    import inference

    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="hazard_result.png")
    ap.add_argument("--conf", type=float, default=None)
    args = ap.parse_args()

    rgb = inference.prepare_image(Image.open(args.image))
    res = inference.load_models().run(rgb, conf=args.conf)
    frame_bgr = np.ascontiguousarray(rgb[..., ::-1])
    floor = floor_mask_from_detections(res.detections, FLOOR_CLASS_NAMES, rgb.shape)
    hz = detect_fall_hazard(frame_bgr, floor, res.depth_raw)
    cv2.imwrite(args.out, hz.overlay_bgr)
    print(hz.status, "|", hz.message, f"| {hz.timing_ms:.0f} ms")
    for r in hz.regions:
        print("  region", r.bbox, r.area_px, r.checks, r.tier, "-", r.tier_reason)
    print("saved", args.out)


if __name__ == "__main__":
    _demo()