#!/usr/bin/env python3
"""
Simple example: Process a video and display focus measure graph
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_focus_measure(frame):
    """Compute Variance of Laplacian (most common focus measure)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def analyze_video(video_path):
    """Analyze video and plot focus measures"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print("\nProcessing frames...")
    
    frame_numbers = []
    focus_values = []
    times = []
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Compute focus measure
        focus = compute_focus_measure(frame)
        time_sec = frame_num / fps if fps > 0 else frame_num
        
        frame_numbers.append(frame_num)
        focus_values.append(focus)
        times.append(time_sec)
        
        # Progress update
        if frame_num % 50 == 0 or frame_num == total_frames - 1:
            print(f"Frame {frame_num}/{total_frames} - Focus: {focus:.2f}")
        
        frame_num += 1
    
    cap.release()
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Focus Measure Analysis (Variance of Laplacian)', fontsize=14, fontweight='bold')
    
    # Plot vs frame number
    ax1.plot(frame_numbers, focus_values, 'b-', linewidth=1.5)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Focus Measure')
    ax1.set_title('Focus Measure vs Frame Number')
    ax1.grid(True, alpha=0.3)
    
    # Plot vs time
    ax2.plot(times, focus_values, 'r-', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Focus Measure')
    ax2.set_title('Focus Measure vs Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Find peaks (best focus frames)
    focus_array = np.array(focus_values)
    peaks = []
    for i in range(1, len(focus_array) - 1):
        if focus_array[i] > focus_array[i-1] and focus_array[i] > focus_array[i+1]:
            # Check if it's a significant peak
            if focus_array[i] > np.mean(focus_array) + 0.5 * np.std(focus_array):
                peaks.append((i, focus_array[i]))
    
    print(f"\n{len(peaks)} focal stack peaks detected")
    print("Best focus frames:")
    for frame_idx, focus_val in sorted(peaks, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Frame {frame_idx} (t={times[frame_idx]:.2f}s): Focus = {focus_val:.2f}")
    
    # Mark peaks on plots
    if peaks:
        peak_frames, peak_values = zip(*peaks)
        peak_times = [times[i] for i in peak_frames]
        ax1.plot(peak_frames, peak_values, 'ro', markersize=8, label='Focus peaks')
        ax2.plot(peak_times, peak_values, 'ro', markersize=8, label='Focus peaks')
        ax1.legend()
        ax2.legend()
    
    plt.show()
    
    return frame_numbers, times, focus_values

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_example.py <video_file>")
        print("Example: python simple_example.py nematodes.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    analyze_video(video_file)
