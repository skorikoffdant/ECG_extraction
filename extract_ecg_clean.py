from pathlib import Path

import cv2
import numpy as np
import pandas as pd


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

DEFAULT_MS_PER_PX = 200.0 / 226.0



def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def find_plot_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x_a = int(w * 0.04)
    x_b = int(w * 0.96)

    row_dark_ratio = np.mean(gray[:, x_a:x_b] < 15, axis=1)
    rows = np.where(row_dark_ratio > 0.85)[0]
    rows = rows[rows > int(h * 0.08)]

    if len(rows) == 0:
        y0 = 100
        y1 = h - 45
    else:
        y0 = int(rows.min())
        y1 = int(rows.max())

    col_dark_ratio = np.mean(gray[y0:y1, :] < 15, axis=0)
    cols = np.where(col_dark_ratio > 0.70)[0]

    if len(cols) == 0:
        x0 = 30
        x1 = w - 30
    else:
        x0 = int(cols.min())
        x1 = int(cols.max())

    x0 = max(0, x0 + 2)
    x1 = min(w - 1, x1 - 2)

    return x0, x1, y0, y1


def make_signal_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = (hsv[:, :, 2] > 90) & (hsv[:, :, 1] < 90) & (gray > 70)

    return mask.astype(bool)


def remove_vertical_grid(mask, y0, y1):
    result = mask.copy()
    col_counts = result[y0:y1, :].sum(axis=0)

    bad_cols = np.where(col_counts > 120)[0]

    for x in bad_cols:
        result[y0:y1, max(0, x - 1):x + 2] = False

    return result


def remove_small_components(mask, min_area=4):
    cleaned = mask.copy().astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            cleaned[labels == i] = 0

    return cleaned.astype(bool)


def remove_text_components(mask, x0, x1, y0, y1):
    cleaned = mask.copy()

    label_x1 = min(x1, x0 + 85)
    roi = cleaned[y0:y1, x0:label_x1].astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        aspect = w / max(h, 1)

        is_long_ecg_line = aspect > 4.0 and h <= 6 and w >= 14

        is_text_like = (
            area <= 500
            and w <= 55
            and h <= 40
            and not is_long_ecg_line
        )

        if is_text_like:
            roi[labels == i] = 0

    cleaned[y0:y1, x0:label_x1] = roi.astype(bool)

    return cleaned


def close_small_horizontal_gaps(mask, x0, x1, y0, y1):
    cleaned = mask.copy().astype(np.uint8)

    roi = cleaned[y0:y1, x0:x1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    cleaned[y0:y1, x0:x1] = roi

    return cleaned.astype(bool)


def smooth_1d(values, sigma=3.0):
    radius = max(1, int(4 * sigma))
    xs = np.arange(-radius, radius + 1)

    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    return np.convolve(values, kernel, mode="same")


def find_baselines(mask, x0, x1, y0, y1, n_leads=12):
    crop = mask[y0:y1, x0:x1]

    profile = crop.sum(axis=1).astype(float)
    profile = smooth_1d(profile, sigma=3.0)

    candidates = []

    for i in range(2, len(profile) - 2):
        if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1]:
            candidates.append((i + y0, profile[i]))

    if not candidates:
        step = (y1 - y0) / 14.0
        return [int(y0 + step * (i + 1)) for i in range(n_leads)]

    threshold = max(np.percentile(profile, 85), profile.max() * 0.15)
    candidates = [(y, score) for y, score in candidates if score >= threshold]
    candidates = sorted(candidates, key=lambda p: p[1], reverse=True)

    chosen = []
    min_dist = max(30, int((y1 - y0) / 22))

    for y, score in candidates:
        if all(abs(y - yy) >= min_dist for yy in chosen):
            chosen.append(int(y))

        if len(chosen) >= n_leads:
            break

    chosen = sorted(chosen)

    if len(chosen) < n_leads:
        if len(chosen) >= 2:
            spacing = int(round(np.median(np.diff(chosen))))

            while len(chosen) < n_leads:
                chosen.append(chosen[-1] + spacing)
        else:
            step = (y1 - y0) / 14.0
            chosen = [int(y0 + step * (i + 1)) for i in range(n_leads)]

    return chosen[:n_leads]


def interpolate_nans(arr):
    arr = np.asarray(arr, dtype=float)
    idx = np.arange(len(arr))

    valid = np.isfinite(arr)

    if valid.sum() == 0:
        return np.zeros_like(arr)

    if valid.sum() == 1:
        return np.full_like(arr, arr[valid][0])

    return np.interp(idx, idx[valid], arr[valid])


def weighted_center_from_pixels(gray, x, ys):
    if len(ys) == 0:
        return np.nan

    ys = np.asarray(ys, dtype=int)

    weights = gray[ys, x].astype(float)
    weights = np.maximum(weights - 35.0, 0.0)

    if weights.sum() <= 0:
        return float(np.mean(ys))

    return float(np.sum(ys * weights) / np.sum(weights))


def vertical_segments(ys):
    if len(ys) == 0:
        return []

    ys = np.asarray(ys, dtype=int)
    ys.sort()

    segments = []

    start = int(ys[0])
    prev = int(ys[0])

    for y in ys[1:]:
        y = int(y)

        if y <= prev + 1:
            prev = y
        else:
            segments.append((start, prev))
            start = y
            prev = y

    segments.append((start, prev))

    return segments


def choose_segment_by_previous(segments, prev_y):
    if not segments:
        return None

    best = None
    best_distance = None

    for top, bottom in segments:
        if top <= prev_y <= bottom:
            distance = 0.0
        else:
            distance = min(abs(top - prev_y), abs(bottom - prev_y))

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = (top, bottom)

    return best


def pick_y_near_previous(gray, mask, x, y_top, y_bottom, prev_y, prev_prev_y, baseline, gap_len):
    ys = np.where(mask[y_top:y_bottom, x])[0] + y_top

    if len(ys) == 0:
        return np.nan

    segments = vertical_segments(ys)

    if prev_y is None:
        best_segment = None
        best_score = None

        for top, bottom in segments:
            center = (top + bottom) / 2.0
            score = abs(center - baseline)

            if best_score is None or score < best_score:
                best_score = score
                best_segment = (top, bottom)

        if best_segment is None:
            return np.nan

        top, bottom = best_segment
        return weighted_center_from_pixels(gray, x, np.arange(top, bottom + 1))

    segment = choose_segment_by_previous(segments, prev_y)

    if segment is None:
        return np.nan

    top, bottom = segment
    height = bottom - top + 1

    if prev_prev_y is None or not np.isfinite(prev_prev_y):
        trend = 0.0
    else:
        trend = prev_y - prev_prev_y

    if height >= 3:
        if trend > 0.25:
            return float(bottom)

        if trend < -0.25:
            return float(top)

        top_distance = abs(top - prev_y)
        bottom_distance = abs(bottom - prev_y)

        if bottom - prev_y >= 6.0 and top_distance <= 3.0:
            return float(bottom)

        if prev_y - top >= 6.0 and bottom_distance <= 3.0:
            return float(top)

        if prev_y > baseline + 1.0:
            return float(bottom)

        if prev_y < baseline - 1.0:
            return float(top)

    return weighted_center_from_pixels(gray, x, np.arange(top, bottom + 1))


def find_start_y(gray, mask, x0, x1, y0, y1, baseline):
    start_x0 = min(x1 - 1, x0 + 100)
    start_x1 = min(x1, x0 + 260)

    best = None

    for x in range(start_x0, start_x1):
        ys = np.where(mask[y0:y1, x])[0] + y0

        if len(ys) == 0:
            continue

        close = ys[np.abs(ys - baseline) <= 8]

        if len(close) == 0:
            close = ys[np.abs(ys - baseline) <= 15]

        if len(close) == 0:
            continue

        y = weighted_center_from_pixels(gray, x, close)
        score = abs(y - baseline)

        if best is None or score < best[0]:
            best = (score, x, y)

    if best is not None:
        return int(best[1]), float(best[2])

    return start_x0, float(baseline)


def trace_line(gray, mask, x0, x1, y0, y1, baseline, start_x, start_y, direction):
    values = {}

    prev_prev_y = None
    prev_y = start_y
    gap_len = 0

    if direction > 0:
        x_range = range(start_x, x1)
    else:
        x_range = range(start_x, x0 - 1, -1)

    for x in x_range:
        y = pick_y_near_previous(
            gray=gray,
            mask=mask,
            x=x,
            y_top=y0,
            y_bottom=y1,
            prev_y=prev_y,
            prev_prev_y=prev_prev_y,
            baseline=baseline,
            gap_len=gap_len,
        )

        if np.isfinite(y):
            values[x] = y
            prev_prev_y = prev_y
            prev_y = y
            gap_len = 0
        else:
            values[x] = np.nan
            gap_len += 1

            if gap_len > 20:
                prev_prev_y = prev_y
                prev_y = 0.9 * prev_y + 0.1 * baseline

    return values


def extract_lead(gray, mask, baseline, x0, x1, y0, y1):
    y_values = np.full(x1 - x0, np.nan, dtype=float)

    start_x, start_y = find_start_y(
        gray=gray,
        mask=mask,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        baseline=baseline,
    )

    right_values = trace_line(
        gray=gray,
        mask=mask,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        baseline=baseline,
        start_x=start_x,
        start_y=start_y,
        direction=1,
    )

    left_values = trace_line(
        gray=gray,
        mask=mask,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        baseline=baseline,
        start_x=start_x,
        start_y=start_y,
        direction=-1,
    )

    for x, y in left_values.items():
        y_values[x - x0] = y

    for x, y in right_values.items():
        y_values[x - x0] = y

    y_values = interpolate_nans(y_values)

    amplitude = baseline - y_values

    return amplitude


def extract_image(img, ms_per_px=DEFAULT_MS_PER_PX, return_raw=False):
    x0, x1, y0, y1 = find_plot_box(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = make_signal_mask(img)
    mask = remove_vertical_grid(mask, y0, y1)
    mask = remove_text_components(mask, x0, x1, y0, y1)
    mask = remove_small_components(mask, min_area=4)
    mask = close_small_horizontal_gaps(mask, x0, x1, y0, y1)

    baseline_x0 = min(x1 - 100, x0 + 80)
    baselines = find_baselines(mask, baseline_x0, x1, y0, y1, n_leads=len(LEADS))

    leads_data = {}

    for i, lead in enumerate(LEADS):
        base = baselines[i]

        signal = extract_lead(
            gray=gray,
            mask=mask,
            baseline=base,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
        )

        leads_data[lead] = signal

    raw_time = np.arange(x1 - x0, dtype=float) * ms_per_px
    out_time = np.arange(0.0, np.floor(raw_time[-1]) + 1.0, 1.0)

    result = {"time_ms": out_time}

    for lead in LEADS:
        result[lead] = np.interp(out_time, raw_time, leads_data[lead])

    df = pd.DataFrame(result)

    df["time_ms"] = df["time_ms"].round(0).astype(float)

    for lead in LEADS:
        df[lead] = df[lead].round(0).astype(int)

    if return_raw:
        return df, mask, baselines, (x0, x1, y0, y1), leads_data

    return df, mask, baselines, (x0, x1, y0, y1)




def detect_r_peaks(signal, times_ms):
    signal = np.asarray(signal, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float)

    if len(signal) < 10:
        return []

    centered = signal - np.median(signal)
    score = np.abs(centered)
    score = smooth_1d(score, sigma=2.0)

    max_score = float(score.max())

    if max_score <= 0:
        return []

    threshold = max(float(np.percentile(score, 92)), max_score * 0.45)
    min_distance_ms = 250.0

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


def save_debug(img, mask, baselines, box, leads_data, out_path):
    x0, x1, y0, y1 = box

    baselines_debug = img.copy()
    cv2.rectangle(baselines_debug, (x0, y0), (x1, y1), (0, 255, 255), 2)

    for y in baselines:
        cv2.line(baselines_debug, (x0, int(y)), (x1, int(y)), (0, 0, 255), 1)

    cv2.imwrite(str(out_path.with_suffix(".baselines.png")), baselines_debug)
    cv2.imwrite(str(out_path.with_suffix(".mask.png")), mask.astype(np.uint8) * 255)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)

    for lead, baseline in zip(LEADS, baselines):
        if lead not in leads_data:
            continue

        amplitude = np.asarray(leads_data[lead], dtype=float)
        xs = np.arange(x0, x0 + len(amplitude), dtype=float)
        ys = float(baseline) - amplitude

        valid = np.isfinite(ys)
        if valid.sum() < 2:
            continue

        points = np.column_stack([xs[valid], ys[valid]]).round().astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], False, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (x0, int(round(baseline))), (x1, int(round(baseline))), (0, 0, 255), 1)

    cv2.imwrite(str(out_path.with_suffix(".overlay.png")), overlay)


def iter_image_paths(input_dir):
    input_dir = Path(input_dir)
    image_paths = []

    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        image_paths.extend(sorted(input_dir.glob(pattern)))

    return sorted(set(image_paths))


def process_folder(input_dir, output_dir, debug=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug"

    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(input_dir)

    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    for path in image_paths:
        img = read_image(path)

        df, mask, baselines, box, leads_data = extract_image(img, return_raw=True)
        bpm, bpm_lead, _ = estimate_bpm(leads_data, DEFAULT_MS_PER_PX)

        csv_path = output_dir / f"{path.stem}.csv"
        df.to_csv(csv_path, index=False)

        if np.isfinite(bpm):
            print(f"{path.name}: BPM={bpm:.1f} (lead {bpm_lead})")
        else:
            print(f"{path.name}: BPM=not detected")

        print(f"saved: {csv_path}")

        if debug:
            save_debug(img, mask, baselines, box, leads_data, debug_dir / path.stem)
