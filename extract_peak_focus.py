#!/usr/bin/env python3
"""
Peak Focus Extractor (CSV Event-Driven)
Reads a CSV event log to identify focus sweep ranges (focus_up, focus_down,
first_focus_down), computes focus measure within each range, finds the peak,
and extracts a short clip around that peak using ffmpeg.
"""

import cv2
import numpy as np
import subprocess
import shutil
import argparse
import os
import sys
import csv


# ---------------------------------------------------------------------------
# Focus measures (same as your other scripts)
# ---------------------------------------------------------------------------
class FocusMeasure:
    """Compute various focus measures for images"""

    @staticmethod
    def variance_of_laplacian(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def tenengrad(image, ksize=3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        return np.mean(gx**2 + gy**2)

    @staticmethod
    def normalized_variance(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean = np.mean(gray)
        return np.var(gray) / mean if mean != 0 else 0

    @staticmethod
    def modified_laplacian(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kernel_x = np.array([[-1, 2, -1]], dtype=np.float64)
        kernel_y = np.array([[-1], [2], [-1]], dtype=np.float64)
        dx = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        dy = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        return np.mean(np.abs(dx) + np.abs(dy))

    @staticmethod
    def sum_modified_laplacian(image, threshold=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kernel_x = np.array([[-1, 2, -1]], dtype=np.float64)
        kernel_y = np.array([[-1], [2], [-1]], dtype=np.float64)
        dx = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        dy = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        ml = np.abs(dx) + np.abs(dy)
        if threshold is None:
            threshold = np.mean(ml)
        return np.sum(ml[ml > threshold])


FOCUS_METHODS = {
    'laplacian': FocusMeasure.variance_of_laplacian,
    'tenengrad': FocusMeasure.tenengrad,
    'variance': FocusMeasure.normalized_variance,
    'modified_laplacian': FocusMeasure.modified_laplacian,
    'sum_modified_laplacian': FocusMeasure.sum_modified_laplacian,
}


# ---------------------------------------------------------------------------
# CSV event parsing
# ---------------------------------------------------------------------------
def parse_focus_events(csv_path):
    """
    Parse the event CSV and return a list of focus-sweep ranges.

    Each range is a dict:
        {
            'start_time': float,
            'end_time':   float,
            'direction':  'up' | 'down',
            'label':      str   (e.g. 'line1_step2' from the details column)
            'event_type': str   (e.g. 'focus_up', 'scan1_first_focus_down')
        }

    Matches these event pairs:
        focus_up_start        / focus_up_end
        focus_down_start      / focus_down_end
        scan1_first_focus_down_start / scan1_first_focus_down_end
        scan2_first_focus_down_start / scan2_first_focus_down_end
        (and any scanN_first_focus_down_start/end pattern)
    """
    events = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                'time': float(row['elapsed_time']),
                'event': row['event'].strip(),
                'details': row['details'].strip() if row.get('details') else '',
            })

    # Pair up start/end events
    ranges = []
    pending_starts = {}  # key: base_event_name -> (time, details)

    for ev in events:
        name = ev['event']

        if name.endswith('_start'):
            # Determine the base name (everything before _start)
            base = name[:-len('_start')]

            # Only care about focus events
            if 'focus_up' in base or 'focus_down' in base:
                pending_starts[base] = (ev['time'], ev['details'])

        elif name.endswith('_end'):
            base = name[:-len('_end')]

            if base in pending_starts:
                start_time, label = pending_starts.pop(base)

                # Determine direction
                if 'focus_up' in base:
                    direction = 'up'
                else:
                    direction = 'down'

                ranges.append({
                    'start_time': start_time,
                    'end_time': ev['time'],
                    'direction': direction,
                    'label': label,
                    'event_type': base,
                })

    return ranges


# ---------------------------------------------------------------------------
# Video focus analysis
# ---------------------------------------------------------------------------
def compute_focus_for_ranges(video_path, ranges, method='laplacian',
                             crop_width_factor=1, crop_height_factor=1):
    """
    For each focus-sweep range, compute the focus measure per frame and
    find the peak frame/time.

    Returns the ranges list with added keys:
        'peak_time', 'peak_frame', 'peak_value',
        'frame_nums', 'times', 'focus_values'
    Also returns (fps, total_frames).
    """
    focus_fn = FOCUS_METHODS[method]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crop region (centred)
    crop_w = width // crop_width_factor
    crop_h = height // crop_height_factor
    x1 = (width - crop_w) // 2
    y1 = (height - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Sort ranges by start time so we can process in a single pass
    ranges = sorted(ranges, key=lambda r: r['start_time'])

    # Pre-compute frame boundaries for each range
    for r in ranges:
        r['frame_start'] = int(r['start_time'] * fps)
        r['frame_end'] = min(int(r['end_time'] * fps), total_frames - 1)
        r['frame_nums'] = []
        r['focus_values'] = []
        r['times'] = []

    # Build a lookup: for each frame, which ranges need it?
    # (Since ranges don't overlap, this is efficient.)
    range_idx = 0  # current range we're looking at

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Advance range_idx past any ranges we've already passed
        while range_idx < len(ranges) and frame_idx > ranges[range_idx]['frame_end']:
            range_idx += 1

        if range_idx >= len(ranges):
            break  # No more ranges to process

        # Check all active ranges (in case of very close/overlapping ranges)
        for ri in range(range_idx, len(ranges)):
            r = ranges[ri]
            if frame_idx < r['frame_start']:
                break  # Haven't reached this range yet, and rest are later
            if r['frame_start'] <= frame_idx <= r['frame_end']:
                roi = frame[y1:y2, x1:x2]
                fv = focus_fn(roi)
                time_sec = frame_idx / fps
                r['frame_nums'].append(frame_idx)
                r['focus_values'].append(fv)
                r['times'].append(time_sec)

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ...", end='\r')

    cap.release()
    print(f"  Processed {frame_idx}/{total_frames} frames — done.")

    # Find peak in each range
    for r in ranges:
        if len(r['focus_values']) == 0:
            r['peak_time'] = (r['start_time'] + r['end_time']) / 2.0
            r['peak_frame'] = int(r['peak_time'] * fps)
            r['peak_value'] = 0.0
            continue

        fv = np.array(r['focus_values'])
        peak_local_idx = int(np.argmax(fv))
        r['peak_time'] = r['times'][peak_local_idx]
        r['peak_frame'] = r['frame_nums'][peak_local_idx]
        r['peak_value'] = r['focus_values'][peak_local_idx]

    return ranges, fps, total_frames


# ---------------------------------------------------------------------------
# ffmpeg extraction
# ---------------------------------------------------------------------------
def extract_clip_ffmpeg(input_path, output_path, start_sec, duration_sec):
    """Extract a clip using ffmpeg."""
    if shutil.which('ffmpeg') is None:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")

    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_sec:.3f}',
        '-i', input_path,
        '-t', f'{duration_sec:.3f}',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-c:a', 'copy',
        output_path,
    ]

    print(f"    ffmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  ffmpeg stderr:\n", result.stderr)
        raise RuntimeError("ffmpeg failed — see stderr above.")
    print(f"    Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def save_focus_plot(ranges, fps, total_frames, method, output_path):
    """
    Save a PNG showing all focus-sweep ranges with their peaks annotated.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 6))

    up_color = '#2196F3'    # blue for focus_up
    down_color = '#F44336'  # red for focus_down

    for i, r in enumerate(ranges):
        if len(r['focus_values']) == 0:
            continue

        color = up_color if r['direction'] == 'up' else down_color
        times = r['times']
        vals = r['focus_values']

        # Plot the curve for this range
        ax.plot(times, vals, color=color, linewidth=1.0, alpha=0.7)

        # Mark the peak
        ax.plot(r['peak_time'], r['peak_value'], 'o', color=color,
                markersize=6, zorder=5)

        # Light shading for the range
        ax.axvspan(r['start_time'], r['end_time'], alpha=0.05, color=color)

    # Legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=up_color, linewidth=2, label='Focus Up'),
        Line2D([0], [0], color=down_color, linewidth=2, label='Focus Down'),
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=6, label='Peak'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    video_duration = total_frames / fps
    ax.set_xlim(0, video_duration)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Focus Measure ({method})')
    ax.set_title(f'Focus Sweeps — {len(ranges)} ranges, peaks marked')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Focus plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Extract peak-focus clips from a video using CSV event ranges.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic — 2s clips around each focus peak
  python extract_peak_focus.py video.mp4 --csv events.csv

  # 1s clips with tenengrad method
  python extract_peak_focus.py video.mp4 --csv events.csv --method tenengrad --duration 1.0

  # Centre-crop before measuring focus
  python extract_peak_focus.py video.mp4 --csv events.csv --crop-width-factor 4 --crop-height-factor 4

  # Save everything: clips + plot + CSV data
  python extract_peak_focus.py video.mp4 --csv events.csv --plot --save-csv

  # Custom output directory
  python extract_peak_focus.py video.mp4 --csv events.csv -o my_results/

CSV format expected:
  elapsed_time,event,details
  Events used: focus_up_start/end, focus_down_start/end,
               scan1_first_focus_down_start/end, scan2_first_focus_down_start/end
        """,
    )

    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--csv', required=True,
                        help='Path to CSV event file')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory for clips and data '
                             '(default: <video_name>_peak_clips/)')
    parser.add_argument('--method', '-m',
                        choices=list(FOCUS_METHODS.keys()),
                        default='laplacian',
                        help='Focus measure method (default: laplacian)')
    parser.add_argument('--duration', '-t', type=float, default=2.0,
                        help='Clip duration in seconds around each peak (default: 2.0)')
    parser.add_argument('--crop-width-factor', type=int, default=1,
                        help='Centre-crop width divisor (default: 1 = full width)')
    parser.add_argument('--crop-height-factor', type=int, default=1,
                        help='Centre-crop height divisor (default: 1 = full height)')
    parser.add_argument('--plot', action='store_true',
                        help='Save a PNG plot of all focus sweeps with peaks')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save per-range focus values to CSV files')

    args = parser.parse_args()

    # --- Validate inputs ---
    if not os.path.isfile(args.video):
        print(f"Error: video file not found: {args.video}")
        return 1
    if not os.path.isfile(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return 1

    # --- Output directory ---
    video_base = os.path.splitext(os.path.basename(args.video))[0]
    video_dir = os.path.dirname(os.path.abspath(args.video))

    if args.output_dir is None:
        output_dir = os.path.join(video_dir, f"{video_base}_peak_clips")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Parse CSV events ---
    print(f"\n=== Peak Focus Extractor (CSV Event-Driven) ===")
    print(f"Video    : {args.video}")
    print(f"CSV      : {args.csv}")
    print(f"Method   : {args.method}")
    print(f"Duration : {args.duration:.1f}s clip around each peak")
    print(f"Output   : {output_dir}")
    print()

    print("Step 1/3  Parsing CSV events ...")
    ranges = parse_focus_events(args.csv)

    if not ranges:
        print("  No focus events found in CSV!")
        return 1

    n_up = sum(1 for r in ranges if r['direction'] == 'up')
    n_down = sum(1 for r in ranges if r['direction'] == 'down')
    print(f"  Found {len(ranges)} focus sweep(s): {n_up} up, {n_down} down")

    print(f"\n  {'#':>4}  {'Direction':>10}  {'Start (s)':>10}  {'End (s)':>10}  "
          f"{'Duration':>10}  {'Label'}")
    print(f"  {'—'*4}  {'—'*10}  {'—'*10}  {'—'*10}  {'—'*10}  {'—'*20}")
    for i, r in enumerate(ranges):
        dur = r['end_time'] - r['start_time']
        print(f"  {i+1:4d}  {r['direction']:>10}  {r['start_time']:10.2f}  "
              f"{r['end_time']:10.2f}  {dur:10.2f}s  {r['label']}")

    # --- Step 2: Compute focus and find peaks ---
    print(f"\nStep 2/3  Computing focus measure within each range ...")
    ranges, fps, total_frames = compute_focus_for_ranges(
        args.video, ranges,
        method=args.method,
        crop_width_factor=args.crop_width_factor,
        crop_height_factor=args.crop_height_factor,
    )

    video_duration = total_frames / fps if fps > 0 else total_frames

    print(f"\n  Peak results:")
    print(f"  {'#':>4}  {'Direction':>10}  {'Peak Time (s)':>14}  "
          f"{'Peak Frame':>12}  {'Peak Value':>14}  {'Label'}")
    print(f"  {'—'*4}  {'—'*10}  {'—'*14}  {'—'*12}  {'—'*14}  {'—'*20}")
    for i, r in enumerate(ranges):
        print(f"  {i+1:4d}  {r['direction']:>10}  {r['peak_time']:14.3f}  "
              f"{r['peak_frame']:12d}  {r['peak_value']:14.2f}  {r['label']}")

    # --- Step 3: Extract clips ---
    half_dur = args.duration / 2.0
    print(f"\nStep 3/3  Extracting {len(ranges)} clip(s) with ffmpeg ...")

    for i, r in enumerate(ranges):
        clip_start = max(0.0, r['peak_time'] - half_dur)
        clip_end = min(video_duration, r['peak_time'] + half_dur)
        actual_dur = clip_end - clip_start

        # Build a descriptive filename:  videoname_peak_001_focus_up_line1_step2.mp4
        safe_label = r['label'].replace(' ', '_').replace(',', '')
        clip_name = (f"{video_base}_peak_{i+1:03d}_{r['event_type']}"
                     f"_{safe_label}.mp4")
        clip_path = os.path.join(output_dir, clip_name)

        print(f"\n  [{i+1}/{len(ranges)}] {r['event_type']} — {r['label']}  "
              f"peak@{r['peak_time']:.2f}s  "
              f"[{clip_start:.3f}s — {clip_end:.3f}s]")
        extract_clip_ffmpeg(args.video, clip_path, clip_start, actual_dur)

    # --- Optional: save plot ---
    if args.plot:
        plot_path = os.path.join(output_dir, f"{video_base}_focus_plot.png")
        save_focus_plot(ranges, fps, total_frames, args.method, plot_path)

    # --- Optional: save per-range CSV ---
    if args.save_csv:
        csv_out = os.path.join(output_dir, f"{video_base}_focus_data.csv")
        print(f"\n  Saving focus data to: {csv_out}")
        with open(csv_out, 'w') as f:
            f.write('range_index,event_type,direction,label,'
                    'frame,time_sec,focus_value,is_peak\n')
            for i, r in enumerate(ranges):
                for j in range(len(r['frame_nums'])):
                    is_peak = 1 if r['frame_nums'][j] == r['peak_frame'] else 0
                    f.write(f"{i+1},{r['event_type']},{r['direction']},"
                            f"{r['label']},{r['frame_nums'][j]},"
                            f"{r['times'][j]:.4f},{r['focus_values'][j]:.6f},"
                            f"{is_peak}\n")

    # --- Summary ---
    print(f"\n=== Done! ===")
    print(f"  {len(ranges)} clip(s) saved to: {output_dir}/")
    if args.plot:
        print(f"  Focus plot: {video_base}_focus_plot.png")
    if args.save_csv:
        print(f"  Focus data: {video_base}_focus_data.csv")

    return 0


if __name__ == '__main__':
    sys.exit(main())
