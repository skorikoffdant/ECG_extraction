from pathlib import Path

import cv2
import numpy as np
import pandas as pd


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

DEFAULT_BIG_SQUARE_PX = 46.5
MS_PER_BIG_SQUARE = 200.0
SIGNAL_GRAY_LIMITS = (135, 140, 145)

LEFT_LEADS = LEADS[:6]
RIGHT_LEADS = LEADS[6:]


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def gaussian_smooth(values, sigma):
    radius = max(1, int(4 * sigma))
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def red_grid_enhancement(img):
    b, g, r = cv2.split(img)
    redness = 2.0 * r.astype(np.float32) - g.astype(np.float32) - b.astype(np.float32)
    redness = cv2.GaussianBlur(redness, (3, 3), 0)
    background = cv2.GaussianBlur(redness, (0, 0), 25)
    return redness - background


def make_red_grid_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = red_grid_enhancement(img)
    threshold = max(6.0, float(np.percentile(enhanced, 90)))

    mask = ((enhanced > threshold) & (gray > 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))

    return mask


def normalize_line_angle(angle):
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def detect_grid_line_angles(img):
    grid_mask = make_red_grid_mask(img)
    h, w = grid_mask.shape
    min_side = min(h, w)

    lines = cv2.HoughLinesP(
        grid_mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(30, int(round(0.03 * min_side))),
        minLineLength=max(25, int(round(0.03 * min_side))),
        maxLineGap=8,
    )

    horizontal_angles = []
    vertical_deviations = []

    if lines is None:
        return horizontal_angles, vertical_deviations

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))

        if length < max(25.0, 0.03 * float(min_side)):
            continue

        angle = normalize_line_angle(float(np.degrees(np.arctan2(dy, dx))))

        if abs(angle) <= 25.0:
            horizontal_angles.append(angle)
        elif abs(abs(angle) - 90.0) <= 25.0:
            if angle >= 0.0:
                vertical_deviations.append(angle - 90.0)
            else:
                vertical_deviations.append(angle + 90.0)

    return horizontal_angles, vertical_deviations


def grid_alignment_score(img):
    horizontal_angles, vertical_deviations = detect_grid_line_angles(img)

    if len(horizontal_angles) < 6 or len(vertical_deviations) < 6:
        return np.inf, {
            "horizontal_count": len(horizontal_angles),
            "vertical_count": len(vertical_deviations),
            "median_angle": np.nan,
        }

    horizontal_angles = np.asarray(horizontal_angles, dtype=float)
    vertical_deviations = np.asarray(vertical_deviations, dtype=float)

    median_h = float(np.median(horizontal_angles))
    median_v = float(np.median(vertical_deviations))
    median_angle = float(np.median([median_h, median_v]))

    mad_h = float(np.median(np.abs(horizontal_angles - median_h)))
    mad_v = float(np.median(np.abs(vertical_deviations - median_v)))

    score = 0.5 * (abs(median_h) + abs(median_v)) + 0.35 * (mad_h + mad_v)

    return float(score), {
        "horizontal_count": int(len(horizontal_angles)),
        "vertical_count": int(len(vertical_deviations)),
        "median_angle": median_angle,
        "horizontal_angle": median_h,
        "vertical_deviation": median_v,
    }


def rotate_keep_size(img, angle):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )



def affine_grid_align(img, horizontal_angle, vertical_deviation):
    h, w = img.shape[:2]
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    h_rad = np.deg2rad(float(horizontal_angle))
    v_rad = np.deg2rad(90.0 + float(vertical_deviation))

    horizontal_axis = np.array([np.cos(h_rad), np.sin(h_rad)], dtype=np.float32)
    vertical_axis = np.array([np.cos(v_rad), np.sin(v_rad)], dtype=np.float32)
    basis = np.column_stack([horizontal_axis, vertical_axis]).astype(np.float32)

    determinant = float(np.linalg.det(basis))
    if abs(determinant) < 0.75:
        return None

    linear = np.linalg.inv(basis).astype(np.float32)
    offset = center - linear @ center
    matrix = np.column_stack([linear, offset]).astype(np.float32)

    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def choose_grid_rectified_image(img):
    before_score, before_info = grid_alignment_score(img)

    info = {
        "used": False,
        "before_score": before_score,
        "after_score": before_score,
        "angle": before_info.get("median_angle", np.nan),
        "reason": "grid_not_detected",
    }

    if not np.isfinite(before_score):
        return img, info

    angle = float(before_info.get("median_angle", np.nan))

    if not np.isfinite(angle):
        return img, info

    horizontal_angle = float(before_info.get("horizontal_angle", angle))
    vertical_deviation = float(before_info.get("vertical_deviation", angle))

    candidates = []

    if abs(angle) >= 0.25:
        for correction_angle in (-angle, angle):
            corrected = rotate_keep_size(img, correction_angle)
            after_score, _ = grid_alignment_score(corrected)
            candidates.append((after_score, correction_angle, corrected, "rotation"))

    if abs(horizontal_angle) >= 0.25 or abs(vertical_deviation) >= 0.25:
        corrected = affine_grid_align(img, horizontal_angle, vertical_deviation)

        if corrected is not None:
            after_score, _ = grid_alignment_score(corrected)
            candidates.append((after_score, 0.0, corrected, "affine"))

    if not candidates:
        info["reason"] = "grid_already_aligned"
        return img, info

    best_score, best_angle, best_img, best_method = min(candidates, key=lambda item: item[0])
    info["after_score"] = float(best_score)
    info["angle"] = float(best_angle)
    info["method"] = best_method

    if np.isfinite(best_score) and best_score < 0.90 * before_score:
        info["used"] = True
        info["reason"] = "accepted"
        return best_img, info

    info["reason"] = "no_improvement"
    return img, info


def estimate_period_from_profile(profile, min_lag=30, max_lag=80):
    profile = profile.astype(np.float32)
    profile -= float(profile.mean())

    if float(profile.std()) < 1e-3:
        return np.nan

    scores = []

    for lag in range(min_lag, max_lag + 1):
        a = profile[:-lag]
        b = profile[lag:]
        denom = float(a.std() * b.std()) + 1e-6
        scores.append((lag, float(np.mean(a * b) / denom)))

    best_lag, best_score = max(scores, key=lambda p: p[1])

    if best_score < 0.2:
        return np.nan

    return float(best_lag)


def estimate_big_square_px(img):
    enhanced = red_grid_enhancement(img)
    h, w = enhanced.shape

    x0 = int(0.05 * w)
    x1 = int(0.95 * w)
    y0 = int(0.05 * h)
    y1 = int(0.88 * h)

    roi = enhanced[y0:y1, x0:x1]
    col_period = estimate_period_from_profile(roi.mean(axis=0))
    row_period = estimate_period_from_profile(roi.mean(axis=1))

    periods = [p for p in (col_period, row_period) if np.isfinite(p)]

    if not periods:
        return DEFAULT_BIG_SQUARE_PX

    return float(np.median(periods))


def make_photo_signal_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 25)
    dark = cv2.subtract(background, gray)

    threshold = max(16.0, float(np.percentile(dark, 90)))
    red_enhanced = red_grid_enhancement(img)
    red_grid = (red_enhanced > 10.0) & (gray > 115) & (dark < 70)

    for gray_limit in SIGNAL_GRAY_LIMITS:
        mask = (dark > threshold) & (gray < gray_limit) & ~red_grid

        if mask.mean() >= 0.003:
            return mask.astype(bool), dark

    return mask.astype(bool), dark


def contiguous_runs(indices, max_gap=1):
    indices = np.asarray(indices, dtype=int)

    if len(indices) == 0:
        return []

    indices.sort()
    runs = []
    start = int(indices[0])
    prev = int(indices[0])

    for value in indices[1:]:
        value = int(value)

        if value <= prev + max_gap:
            prev = value
        else:
            runs.append((start, prev))
            start = value
            prev = value

    runs.append((start, prev))
    return runs


def estimate_flat_baseline(mask, rect):
    x0, x1, y0, y1 = rect
    width = x1 - x0
    sx0 = x0 + max(20, int(round(width * 0.06)))
    sx1 = x1 - max(10, int(round(width * 0.03)))
    sx1 = max(sx0 + 2, sx1)
    crop = mask[y0:y1, sx0:sx1]

    profile = crop.sum(axis=1).astype(float)
    profile = gaussian_smooth(profile, sigma=3.0)

    if profile.max() <= 0:
        return float((y0 + y1) / 2.0)

    candidates = []

    for i in range(2, len(profile) - 2):
        if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1]:
            candidates.append((i + y0, profile[i]))

    threshold = max(float(np.percentile(profile, 85)), float(profile.max()) * 0.15)
    candidates = [(y, score) for y, score in candidates if score >= threshold]

    if not candidates:
        return float(y0 + int(np.argmax(profile)))

    mid_y = (y0 + y1) / 2.0
    return float(max(candidates, key=lambda item: (item[1], -abs(item[0] - mid_y)))[0])


def detect_divider_x(mask):
    h, w = mask.shape
    y0 = int(round(0.06 * h))
    y1 = int(round(0.88 * h))
    x0 = int(round(0.45 * w))
    x1 = int(round(0.55 * w))

    profile = mask[y0:y1, :].sum(axis=0).astype(float)
    profile = gaussian_smooth(profile, sigma=6.0)

    if x1 <= x0:
        return int(round(0.50 * w))

    return int(x0 + np.argmin(profile[x0:x1]))


def pick_strong_row_peaks(profile, y_offset, expected_count, min_spacing):
    profile = gaussian_smooth(profile.astype(float), sigma=3.0)

    if profile.max() <= 0:
        return []

    threshold = max(float(np.percentile(profile, 80)), float(profile.max()) * 0.18)
    peaks = []

    for idx in range(2, len(profile) - 2):
        if profile[idx] >= profile[idx - 1] and profile[idx] > profile[idx + 1] and profile[idx] >= threshold:
            peaks.append((idx + y_offset, float(profile[idx])))

    selected = []

    for y, score in sorted(peaks, key=lambda item: item[1], reverse=True):
        if all(abs(float(y) - float(existing_y)) >= min_spacing for existing_y, _ in selected):
            selected.append((int(y), float(score)))

        if len(selected) >= expected_count:
            break

    return sorted(y for y, _ in selected)


def detect_side_row_centers(mask, x0, x1, expected_count=6):
    h, _ = mask.shape
    y0 = int(round(0.05 * h))
    y1 = int(round(0.90 * h))
    x0 = int(np.clip(x0, 0, mask.shape[1] - 2))
    x1 = int(np.clip(x1, x0 + 2, mask.shape[1]))

    profile = mask[y0:y1, x0:x1].sum(axis=1).astype(float)
    min_spacing = max(35, int(round(0.07 * h)))
    centers = pick_strong_row_peaks(profile, y0, expected_count, min_spacing)

    if len(centers) != expected_count:
        raise ValueError("Cannot detect row centers")

    return [float(y) for y in centers]


def build_row_windows(centers, image_h):
    centers = [float(y) for y in centers]
    row_step = float(np.median(np.diff(centers))) if len(centers) > 1 else float(image_h) / 8.0
    windows = []

    for center_y in centers:
        y0 = int(max(0, round(center_y - 0.75 * row_step)))
        y1 = int(min(image_h, round(center_y + 0.60 * row_step)))

        if y1 <= y0 + 8:
            y0 = int(max(0, round(center_y - 0.65 * row_step)))
            y1 = int(min(image_h, round(center_y + 0.65 * row_step)))

        windows.append((y0, y1))

    return windows


def detect_photo_layout(img_shape, mask):
    h, w = img_shape[:2]
    divider_x = detect_divider_x(mask)
    side_gap = max(24, int(round(0.022 * w)))

    left_x0 = int(round(0.035 * w))
    left_x1 = max(left_x0 + 80, divider_x - side_gap)
    right_x0 = min(w - 80, divider_x + side_gap)
    right_x1 = int(round(0.972 * w))

    left_centers = detect_side_row_centers(mask, left_x0, left_x1, expected_count=6)
    right_centers = detect_side_row_centers(mask, right_x0, right_x1, expected_count=6)
    left_windows = build_row_windows(left_centers, h)
    right_windows = build_row_windows(right_centers, h)

    rects = {}

    for lead, (y0, y1) in zip(LEFT_LEADS, left_windows):
        rects[lead] = (left_x0, left_x1, y0, y1)

    for lead, (y0, y1) in zip(RIGHT_LEADS, right_windows):
        rects[lead] = (right_x0, right_x1, y0, y1)

    layout_debug = {
        "divider_x": divider_x,
        "left_centers": left_centers,
        "right_centers": right_centers,
        "left_x": (left_x0, left_x1),
        "right_x": (right_x0, right_x1),
    }

    return rects, layout_debug


def vertical_segments(ys):
    return contiguous_runs(ys, max_gap=2)


def build_column_groups(mask_col, dark_col):
    groups = []
    ys = np.where(mask_col)[0]

    for top, bottom in vertical_segments(ys):
        rows = np.arange(top, bottom + 1)
        ink = np.maximum(dark_col[rows].astype(float), 0.0)
        support = float(ink.sum())

        if support > 1e-6:
            center = float(np.sum(rows * ink) / support)
        else:
            center = float((top + bottom) / 2.0)

        span = float(bottom - top + 1)
        groups.append(
            {
                "center": center,
                "top": float(top),
                "bottom": float(bottom),
                "span": span,
                "support": support,
                "density": support / max(1.0, span),
            }
        )

    return groups


def choose_column_group(groups, target_y, anchor_y, band_height, has_track, x_idx, gap_count):
    if not groups:
        return None

    startup_width = max(10, int(round(0.12 * band_height)))
    startup_radius = max(14.0, 0.36 * float(band_height))
    continuity_scale = max(5.0, 0.08 * float(band_height))
    anchor_scale = max(12.0, 0.50 * float(band_height))
    support_scale = max(10.0, 0.95 * float(band_height))

    best_group = None
    best_cost = 1.15 + 0.04 * min(int(gap_count), 10)
    tall_fallback = None
    tall_fallback_score = None

    for group in groups:
        center = float(group["center"])

        if not has_track and x_idx < startup_width and abs(center - anchor_y) > startup_radius:
            continue

        continuity = abs(center - target_y) / continuity_scale
        anchor_pull = abs(center - anchor_y) / anchor_scale
        distance_from_anchor = abs(center - anchor_y)

        if distance_from_anchor > 0.72 * float(band_height):
            continue

        if float(group["span"]) >= max(8.0, 0.11 * float(band_height)):
            excursion = max(abs(float(group["top"]) - anchor_y), abs(float(group["bottom"]) - anchor_y))
            fallback_score = excursion + 0.002 * float(group["support"]) - 0.15 * abs(center - target_y)

            if tall_fallback is None or fallback_score > tall_fallback_score:
                tall_fallback = group
                tall_fallback_score = fallback_score

        support_bonus = min(float(group["support"]) / support_scale, 2.0)
        density_bonus = min(float(group["density"]) / max(3.0, 0.07 * band_height), 1.3)
        wide_penalty = max(0.0, float(group["span"]) - 0.34 * band_height) / max(8.0, 0.18 * band_height)

        if has_track:
            cost = continuity + 0.18 * anchor_pull + 0.08 * wide_penalty - 0.12 * support_bonus - 0.04 * density_bonus
        else:
            cost = continuity + 0.38 * anchor_pull + 0.08 * wide_penalty - 0.12 * support_bonus - 0.04 * density_bonus

        if cost < best_cost:
            best_group = group
            best_cost = cost

    if best_group is None and tall_fallback is not None:
        return tall_fallback

    return best_group


def track_column_groups(roi_mask, roi_dark, anchor_y):
    band_height, width = roi_mask.shape
    centers = np.full(width, np.nan, dtype=float)
    tops = np.full(width, np.nan, dtype=float)
    bottoms = np.full(width, np.nan, dtype=float)
    spans = np.full(width, np.nan, dtype=float)
    supports = np.full(width, np.nan, dtype=float)
    valid = np.zeros(width, dtype=bool)

    previous = float(anchor_y)
    has_track = False
    gap_count = 0

    for x_idx in range(width):
        groups = build_column_groups(roi_mask[:, x_idx], roi_dark[:, x_idx])
        target_y = previous if has_track else float(anchor_y)
        group = choose_column_group(
            groups=groups,
            target_y=target_y,
            anchor_y=float(anchor_y),
            band_height=band_height,
            has_track=has_track,
            x_idx=x_idx,
            gap_count=gap_count,
        )

        if group is None:
            gap_count += 1
            if has_track and gap_count >= 3:
                previous = 0.96 * previous + 0.04 * float(anchor_y)
            continue

        centers[x_idx] = float(group["center"])
        tops[x_idx] = float(group["top"])
        bottoms[x_idx] = float(group["bottom"])
        spans[x_idx] = float(group["span"])
        supports[x_idx] = float(group["support"])
        valid[x_idx] = True
        previous = float(group["center"])
        has_track = True
        gap_count = 0

    return centers, tops, bottoms, spans, supports, valid


def choose_group_point(center_y, top_y, bottom_y, span, prev_y, next_y, anchor_y, band_height):
    if not np.isfinite(center_y):
        return np.nan

    if span <= 2.0:
        return float(center_y)

    slope_limit = max(1.8, 0.018 * float(band_height))
    tall_group = span >= max(5.0, 0.035 * float(band_height))
    top_excursion = abs(float(top_y) - float(anchor_y))
    bottom_excursion = abs(float(bottom_y) - float(anchor_y))

    if tall_group:
        edge_margin = max(1.0, 0.10 * float(span))

        if top_excursion >= bottom_excursion + edge_margin:
            return float(top_y)
        if bottom_excursion >= top_excursion + edge_margin:
            return float(bottom_y)

    if np.isfinite(prev_y) and np.isfinite(next_y):
        expected = 0.5 * (float(prev_y) + float(next_y))
        excursion = max(1.3, min(0.24 * float(span), 0.055 * float(band_height)))

        if float(center_y) < expected - excursion:
            return float(top_y)
        if float(center_y) > expected + excursion:
            return float(bottom_y)

        trend = float(next_y) - float(prev_y)
    elif np.isfinite(prev_y):
        trend = float(center_y) - float(prev_y)
    elif np.isfinite(next_y):
        trend = float(next_y) - float(center_y)
    else:
        trend = np.nan

    if np.isfinite(trend):
        if trend < -slope_limit:
            return float(top_y)
        if trend > slope_limit:
            return float(bottom_y)

    if tall_group:
        if top_excursion >= bottom_excursion + 1.0:
            return float(top_y)
        if bottom_excursion >= top_excursion + 1.0:
            return float(bottom_y)

        if float(center_y) < float(anchor_y):
            return float(top_y)
        if float(center_y) > float(anchor_y):
            return float(bottom_y)

    return float(center_y)


def refine_group_edges(centers, tops, bottoms, spans, valid, anchor_y, band_height):
    refined = centers.copy()
    usable_indices = np.where(valid & np.isfinite(centers))[0]

    for pos, x_idx in enumerate(usable_indices):
        prev_y = centers[usable_indices[pos - 1]] if pos > 0 else np.nan
        next_y = centers[usable_indices[pos + 1]] if pos + 1 < len(usable_indices) else np.nan
        refined[x_idx] = choose_group_point(
            center_y=centers[x_idx],
            top_y=tops[x_idx],
            bottom_y=bottoms[x_idx],
            span=spans[x_idx],
            prev_y=prev_y,
            next_y=next_y,
            anchor_y=anchor_y,
            band_height=band_height,
        )

    return refined


def fill_short_nan_gaps(values, valid, max_gap):
    filled = values.copy()
    filled_valid = valid.copy()
    width = len(filled)
    x_idx = 0

    while x_idx < width:
        if filled_valid[x_idx] and np.isfinite(filled[x_idx]):
            x_idx += 1
            continue

        start = x_idx

        while x_idx < width and (not filled_valid[x_idx] or not np.isfinite(filled[x_idx])):
            x_idx += 1

        end = x_idx - 1
        gap_len = end - start + 1
        left = start - 1
        right = end + 1

        if gap_len <= max_gap and left >= 0 and right < width and filled_valid[left] and filled_valid[right]:
            left_y = float(filled[left])
            right_y = float(filled[right])

            for j in range(start, end + 1):
                alpha = (j - left) / max(1, right - left)
                filled[j] = (1.0 - alpha) * left_y + alpha * right_y
                filled_valid[j] = True

    return filled, filled_valid


def complete_trace_path(values, valid, anchor_y, band_height):
    completed = values.copy()
    usable = valid & np.isfinite(completed)

    if usable.sum() == 0:
        return np.full(len(completed), float(anchor_y), dtype=float)

    indices = np.where(usable)[0]
    first = int(indices[0])
    last = int(indices[-1])

    completed[:first] = float(anchor_y)
    completed[last + 1:] = float(anchor_y)

    jump_limit = max(7.0, 0.12 * float(band_height))
    flat_limit = max(3.0, 0.035 * float(band_height))
    short_gap_limit = max(4, int(round(0.008 * len(completed))))

    x_idx = first

    while x_idx <= last:
        if usable[x_idx]:
            x_idx += 1
            continue

        start = x_idx

        while x_idx <= last and not usable[x_idx]:
            x_idx += 1

        end = x_idx - 1
        left = start - 1
        right = end + 1

        if left < 0 or right >= len(completed) or not usable[left] or not usable[right]:
            completed[start:end + 1] = float(anchor_y)
            continue

        left_y = float(completed[left])
        right_y = float(completed[right])
        gap_len = end - start + 1
        jump = abs(right_y - left_y)

        if (gap_len <= short_gap_limit and jump <= jump_limit) or jump <= flat_limit:
            for j in range(start, end + 1):
                alpha = (j - left) / max(1, right - left)
                completed[j] = (1.0 - alpha) * left_y + alpha * right_y
            continue

        if jump <= jump_limit and abs(left_y - anchor_y) <= jump_limit and abs(right_y - anchor_y) <= jump_limit:
            for j in range(start, end + 1):
                alpha = (j - left) / max(1, right - left)
                completed[j] = (1.0 - alpha) * left_y + alpha * right_y
            continue

        completed[start:end + 1] = float(anchor_y)

        ramp = min(3, max(1, gap_len // 4))

        for offset in range(ramp):
            alpha = (offset + 1) / float(ramp + 1)
            completed[start + offset] = (1.0 - alpha) * left_y + alpha * float(anchor_y)
            completed[end - offset] = (1.0 - alpha) * right_y + alpha * float(anchor_y)

    return completed


def suppress_narrow_scan_spikes(values, valid, band_height, protected=None):
    cleaned = values.copy()
    usable = valid & np.isfinite(cleaned)

    if protected is None:
        protected = np.zeros(len(cleaned), dtype=bool)
    else:
        protected = protected.astype(bool)

    if usable.sum() < 3:
        return cleaned

    spike_limit = max(3.5, 0.045 * float(band_height))
    flat_limit = max(1.6, 0.020 * float(band_height))

    for _ in range(2):
        for x_idx in range(1, len(cleaned) - 1):
            if protected[x_idx]:
                continue
            if not (usable[x_idx - 1] and usable[x_idx] and usable[x_idx + 1]):
                continue

            left_y = float(cleaned[x_idx - 1])
            mid_y = float(cleaned[x_idx])
            right_y = float(cleaned[x_idx + 1])

            if abs(right_y - left_y) <= flat_limit:
                expected = 0.5 * (left_y + right_y)
                if abs(mid_y - expected) >= spike_limit:
                    cleaned[x_idx] = expected

        for x_idx in range(1, len(cleaned) - 2):
            if protected[x_idx] or protected[x_idx + 1]:
                continue
            if not (usable[x_idx - 1] and usable[x_idx] and usable[x_idx + 1] and usable[x_idx + 2]):
                continue

            left_y = float(cleaned[x_idx - 1])
            right_y = float(cleaned[x_idx + 2])

            if abs(right_y - left_y) > flat_limit:
                continue

            first_expected = left_y + (right_y - left_y) / 3.0
            second_expected = left_y + 2.0 * (right_y - left_y) / 3.0

            if abs(float(cleaned[x_idx]) - first_expected) >= spike_limit and abs(float(cleaned[x_idx + 1]) - second_expected) >= spike_limit:
                cleaned[x_idx] = first_expected
                cleaned[x_idx + 1] = second_expected

    return cleaned


def trace_roi(mask, dark, rect, baseline_y=None):
    x0, x1, y0, y1 = rect
    roi_mask = mask[y0:y1, x0:x1]
    roi_dark = dark[y0:y1, x0:x1]
    height, width = roi_mask.shape

    if baseline_y is None:
        anchor_y = height / 2.0
    else:
        anchor_y = float(np.clip(baseline_y - y0, 0, height - 1))

    centers, tops, bottoms, spans, supports, valid = track_column_groups(roi_mask, roi_dark, anchor_y)
    values = refine_group_edges(centers, tops, bottoms, spans, valid, anchor_y, height)
    values, valid = fill_short_nan_gaps(values, valid, max_gap=6)

    if valid.any():
        support_cutoff = float(np.percentile(supports[valid & np.isfinite(supports)], 45)) if np.any(valid & np.isfinite(supports)) else 0.0
        protected = valid & np.isfinite(spans) & (spans >= max(3.0, 0.045 * height)) & (supports >= support_cutoff)
    else:
        protected = np.zeros(width, dtype=bool)

    values = suppress_narrow_scan_spikes(values, valid, height, protected=protected)
    values = complete_trace_path(values, valid, anchor_y, height)

    if len(values) >= 7:
        smoothed = gaussian_smooth(values, sigma=0.45)
        smoothable = ~protected if len(protected) == len(values) else np.ones(len(values), dtype=bool)
        values[smoothable] = smoothed[smoothable]

    values = values + float(y0)

    if baseline_y is None:
        baseline = float(np.median(values))
    else:
        baseline = float(baseline_y)

    amplitude = baseline - values

    return amplitude, values, baseline


def detect_r_peaks(signal, times_ms):
    signal = np.asarray(signal, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float)

    if len(signal) < 10:
        return []

    centered = signal - np.median(signal)
    score = np.abs(centered)
    score = gaussian_smooth(score, sigma=2.0)

    max_score = float(score.max())

    if max_score <= 0:
        return []

    threshold = max(float(np.percentile(score, 92)), max_score * 0.45)
    min_distance_ms = 350.0

    candidates = []

    for i in range(1, len(score) - 1):
        if score[i] >= score[i - 1] and score[i] > score[i + 1] and score[i] >= threshold:
            candidates.append((i, score[i]))

    selected = []

    for idx, value in sorted(candidates, key=lambda p: p[1], reverse=True):
        t = times_ms[idx]

        if all(abs(t - times_ms[other]) >= min_distance_ms for other in selected):
            selected.append(idx)

    selected.sort()

    return [float(times_ms[i]) for i in selected]


def estimate_bpm(leads_data, ms_per_px):
    best = None

    for lead, signal in leads_data.items():
        raw_time = np.arange(len(signal), dtype=float) * ms_per_px
        peaks = detect_r_peaks(signal, raw_time)

        if len(peaks) < 2:
            continue

        intervals = np.diff(peaks)
        intervals = intervals[(intervals >= 250.0) & (intervals <= 1800.0)]

        if len(intervals) == 0:
            continue

        bpm = 60000.0 / float(np.mean(intervals))
        confidence = len(intervals)
        amplitude = float(np.percentile(np.abs(signal - np.median(signal)), 95))
        candidate = (confidence, amplitude, bpm, lead, peaks)

        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        return np.nan, None, []

    _, _, bpm, lead, peaks = best
    return float(bpm), lead, peaks


def extract_scan_image(img, return_debug=False):
    rectified = img.copy()
    mask, dark = make_photo_signal_mask(rectified)
    layout_rects, layout_debug = detect_photo_layout(rectified.shape, mask)

    big_square_px = estimate_big_square_px(rectified)
    ms_per_px = MS_PER_BIG_SQUARE / big_square_px

    raw_leads = {}
    y_values = {}
    baselines = {}
    rects = {}
    durations = []

    for lead in LEADS:
        rect = layout_rects[lead]
        baseline_y = estimate_flat_baseline(mask, rect)
        amplitude, ys, baseline = trace_roi(
            mask=mask,
            dark=dark,
            rect=rect,
            baseline_y=baseline_y,
        )

        raw_leads[lead] = amplitude
        y_values[lead] = ys
        baselines[lead] = baseline
        rects[lead] = rect
        durations.append((len(amplitude) - 1) * ms_per_px)

    duration_ms = max(1.0, float(np.floor(min(durations))))
    out_time = np.arange(0.0, duration_ms + 1.0, 1.0)

    result = {"time_ms": out_time}

    for lead in LEADS:
        raw_time = np.arange(len(raw_leads[lead]), dtype=float) * ms_per_px
        result[lead] = np.interp(out_time, raw_time, raw_leads[lead])

    df = pd.DataFrame(result)
    df["time_ms"] = df["time_ms"].round(0).astype(float)

    for lead in LEADS:
        df[lead] = df[lead].round(0).astype(int)

    bpm, bpm_lead, bpm_peaks = estimate_bpm(raw_leads, ms_per_px)

    if not return_debug:
        return df, bpm, bpm_lead

    debug = {
        "rectified": rectified,
        "mask": mask,
        "dark": dark,
        "big_square_px": big_square_px,
        "ms_per_px": ms_per_px,
        "rects": rects,
        "y_values": y_values,
        "baselines": baselines,
        "raw_leads": raw_leads,
        "bpm_peaks": bpm_peaks,
        "layout": layout_debug,
    }

    return df, bpm, bpm_lead, debug


def save_debug_images(debug, out_base):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    rectified = debug["rectified"]
    mask = debug["mask"]

    cv2.imwrite(str(out_base.with_suffix(".rectified.jpg")), rectified)
    cv2.imwrite(str(out_base.with_suffix(".signal_mask.png")), mask.astype(np.uint8) * 255)

    overlay = rectified.copy()

    for lead in LEADS:
        x0, x1, y0, y1 = debug["rects"][lead]
        baseline = debug["baselines"][lead]
        ys = debug["y_values"][lead]

        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 180, 255), 1)
        cv2.line(overlay, (x0, int(round(baseline))), (x1, int(round(baseline))), (0, 0, 255), 1)

        xs = np.arange(x0, x0 + len(ys), dtype=float)
        points = np.column_stack([xs, ys]).round().astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], False, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    cv2.imwrite(str(out_base.with_suffix(".overlay.png")), overlay)


def process_image(image_path, output_dir, debug=False):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = read_image(image_path)
    img, rectification_info = choose_grid_rectified_image(img)

    before_score = rectification_info["before_score"]
    after_score = rectification_info["after_score"]

    if rectification_info["used"]:
        print(
            f"{image_path.name}: grid rectification accepted "
            f"(score {before_score:.3f} -> {after_score:.3f}, "
            f"method {rectification_info.get('method', 'unknown')}, "
            f"angle {rectification_info['angle']:.2f} deg)"
        )
    else:
        reason = rectification_info["reason"]
        if np.isfinite(before_score) and np.isfinite(after_score):
            print(
                f"{image_path.name}: grid rectification skipped, original image kept "
                f"(score {before_score:.3f} -> {after_score:.3f}, reason: {reason})"
            )
        else:
            print(
                f"{image_path.name}: grid rectification skipped, original image kept "
                f"(reason: {reason})"
            )

    if debug:
        df, bpm, bpm_lead, debug_data = extract_scan_image(img, return_debug=True)
        debug_data["rectification"] = rectification_info
    else:
        df, bpm, bpm_lead = extract_scan_image(img, return_debug=False)
        debug_data = None

    csv_path = output_dir / f"{image_path.stem}.csv"
    df.to_csv(csv_path, index=False)

    if np.isfinite(bpm):
        print(f"{image_path.name}: BPM={bpm:.1f} (lead {bpm_lead})")
    else:
        print(f"{image_path.name}: BPM=not detected")

    print(f"saved: {csv_path}")

    if debug and debug_data is not None:
        debug_dir = output_dir / "debug"
        save_debug_images(debug_data, debug_dir / image_path.stem)


def process_folder(input_dir, output_dir, debug=False):
    input_dir = Path(input_dir)
    image_paths = []

    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        image_paths.extend(sorted(input_dir.glob(pattern)))

    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    for image_path in sorted(set(image_paths)):
        process_image(image_path, output_dir, debug=debug)