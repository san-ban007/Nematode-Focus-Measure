#!/usr/bin/env python3
"""
Focus Measure Analyzer with Live Video Display
Shows the video alongside the focus measure graph in real-time
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import argparse
import time


class FocusMeasure:
    """Compute various focus measures for images"""
    
    @staticmethod
    def variance_of_laplacian(image):
        """Variance of Laplacian - most popular focus measure"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    @staticmethod
    def tenengrad(image, ksize=3):
        """Tenengrad - based on gradient magnitude"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        return np.mean(gx**2 + gy**2)
    
    @staticmethod
    def normalized_variance(image):
        """Normalized variance of grayscale values"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean = np.mean(gray)
        if mean == 0:
            return 0
        return np.var(gray) / mean
    
    @staticmethod
    def modified_laplacian(image):
        """
        Modified Laplacian focus measure
        Uses two 1D second derivative kernels instead of 2D Laplacian
        More sensitive to directional edges
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Second derivative kernels for x and y directions
        kernel_x = np.array([[-1, 2, -1]], dtype=np.float64)
        kernel_y = np.array([[-1], [2], [-1]], dtype=np.float64)
        
        # Compute second derivatives in x and y directions
        dx = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        dy = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        
        # Modified Laplacian: sum of absolute values
        ml = np.abs(dx) + np.abs(dy)
        
        # Return mean of modified Laplacian values
        return np.mean(ml)
    
    @staticmethod
    def sum_modified_laplacian(image, threshold=None):
        """
        Sum Modified Laplacian (SML) focus measure
        Sums ML values above a threshold to focus on sharp edges
        If threshold is None, uses mean of ML as threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Second derivative kernels for x and y directions
        kernel_x = np.array([[-1, 2, -1]], dtype=np.float64)
        kernel_y = np.array([[-1], [2], [-1]], dtype=np.float64)
        
        # Compute second derivatives in x and y directions
        dx = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        dy = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        
        # Modified Laplacian: sum of absolute values
        ml = np.abs(dx) + np.abs(dy)
        
        # Use adaptive threshold if none provided
        if threshold is None:
            threshold = np.mean(ml)
        
        # Sum only values above threshold
        sml = np.sum(ml[ml > threshold])
        
        return sml


class VideoWithFocusGraph:
    """Display video alongside live focus measure graph"""
    
    def __init__(self, video_path, method='laplacian', buffer_size=500, 
                 display_mode='sidebyside', playback_speed=1.0, downsample=1,
                 plot_type='time', save_interval=None, output_dir='focus_plots',
                 marker_interval=1, crop_factor=(1, 1)):
        self.video_path = video_path
        self.method = method
        self.buffer_size = buffer_size
        self.display_mode = display_mode  # 'sidebyside' or 'overlay'
        self.playback_speed = playback_speed
        self.downsample = downsample  # Process every Nth frame for speed
        self.plot_type = plot_type  # 'time' or 'frame'
        self.save_interval = save_interval  # Minutes between saves
        self.marker_interval = marker_interval  # Seconds between markers (1, 2, 5, etc.)
        self.crop_factor = crop_factor  # (m, n) where crop is (width/m, height/n)
        
        # Determine output directory
        # If output_dir is the default 'focus_plots', create folder with video name and timestamp
        # Otherwise use the user-specified directory
        if output_dir == 'focus_plots':
            # Get the directory containing the video and video filename
            import os
            from datetime import datetime
            
            video_dir = os.path.dirname(os.path.abspath(video_path))
            video_name = os.path.splitext(os.path.basename(video_path))[0]  # Remove extension
            
            # Create timestamp in format: YYYY-MM-DD_HH-MM-SS
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            # Create folder name: videoname_timestamp
            folder_name = f"{video_name}_{timestamp}"
            
            self.output_dir = os.path.join(video_dir, 'processed', folder_name)
        else:
            self.output_dir = output_dir
        
        # Track time for auto-save
        self.last_save_time = 0.0
        self.save_counter = 0
        
        # Store ALL data for saving (not limited by buffer_size)
        self.all_frame_numbers = []
        self.all_focus_values = []
        self.all_times = []
        
        # Data storage
        self.frame_numbers = deque(maxlen=buffer_size)
        self.focus_values = deque(maxlen=buffer_size)
        self.times = deque(maxlen=buffer_size)
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate crop dimensions
        self.crop_width = self.width // self.crop_factor[0]
        self.crop_height = self.height // self.crop_factor[1]
        
        # Calculate crop boundaries (centered)
        self.crop_x1 = (self.width - self.crop_width) // 2
        self.crop_y1 = (self.height - self.crop_height) // 2
        self.crop_x2 = self.crop_x1 + self.crop_width
        self.crop_y2 = self.crop_y1 + self.crop_height
        
        # Create output directory and save parameters AFTER video properties are loaded
        if self.save_interval is not None:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Plots will be saved to: {os.path.abspath(self.output_dir)}/")
            
            # Save parameters.json now that we have all video properties
            self.save_parameters_json()
        
        # Select focus measure method
        self.focus_methods = {
            'laplacian': FocusMeasure.variance_of_laplacian,
            'tenengrad': FocusMeasure.tenengrad,
            'variance': FocusMeasure.normalized_variance,
            'modified_laplacian': FocusMeasure.modified_laplacian,
            'sum_modified_laplacian': FocusMeasure.sum_modified_laplacian
        }
        
        if method not in self.focus_methods:
            raise ValueError(f"Unknown method: {method}")
        
        self.focus_function = self.focus_methods[method]
        
        # Setup matplotlib for graph
        if display_mode == 'sidebyside':
            self.setup_sidebyside_display()
        else:
            self.setup_overlay_display()
        
        self.current_frame = None
        self.current_frame_num = 0
        self.paused = False
        self.start_time = time.time()
    
    def setup_sidebyside_display(self):
        """Setup side-by-side video and graph display"""
        self.fig = plt.figure(figsize=(16, 6))
        
        # Video display (left)
        self.ax_video = self.fig.add_subplot(1, 2, 1)
        self.ax_video.set_title('Video Frame')
        self.ax_video.axis('off')
        
        # Initialize with black frame
        dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.im_video = self.ax_video.imshow(cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB))
        
        # Graph display (right)
        self.ax_graph = self.fig.add_subplot(1, 2, 2)
        self.line, = self.ax_graph.plot([], [], 'b-', linewidth=2)
        self.current_point, = self.ax_graph.plot([], [], 'ro', markersize=10)
        
        # Set labels based on plot type
        if self.plot_type == 'time':
            self.ax_graph.set_xlabel('Time (seconds)', fontsize=12)
        else:  # frame
            self.ax_graph.set_xlabel('Frame Number', fontsize=12)
        
        self.ax_graph.set_ylabel('Focus Measure', fontsize=12)
        self.ax_graph.set_title(f'Focus Measure - {self.method.title()}', fontsize=12, fontweight='bold')
        self.ax_graph.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def setup_overlay_display(self):
        """Setup overlay display with graph on video"""
        # Use OpenCV window for video
        cv2.namedWindow('Video with Focus Graph', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video with Focus Graph', 1280, 720)
        
        # Matplotlib for mini graph (we'll embed this)
        # Taller figure to prevent title clipping
        self.fig, self.ax_graph = plt.subplots(1, 1, figsize=(6, 2.6))
        self.line, = self.ax_graph.plot([], [], 'b-', linewidth=2)
        self.current_point, = self.ax_graph.plot([], [], 'ro', markersize=8)
        
        # Set labels based on plot type
        if self.plot_type == 'time':
            self.ax_graph.set_xlabel('Time (s)', fontsize=8)
        else:  # frame
            self.ax_graph.set_xlabel('Frame', fontsize=8)
        
        self.ax_graph.set_ylabel('Focus', fontsize=8)
        self.ax_graph.grid(True, alpha=0.3)
        
        # Adjust layout to prevent title clipping
        # Leave substantial space at the top for the title
        plt.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.95)
    
    def crop_frame(self, frame):
        """Crop the frame to the region of interest (centered)"""
        if frame is None or frame.size == 0:
            return None
        
        # If crop_factor is (1, 1), return the full frame
        if self.crop_factor == (1, 1):
            return frame
        
        # Crop to the centered region
        cropped = frame[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
        return cropped
    
    def process_frame(self, frame):
        """Compute focus measure for a single frame"""
        if frame is None or frame.size == 0:
            return None
        
        # Crop the frame first
        cropped_frame = self.crop_frame(frame)
        if cropped_frame is None or cropped_frame.size == 0:
            return None
        
        # Compute focus measure on the cropped frame
        return self.focus_function(cropped_frame)
    
    def save_interval_plot(self, current_time, start_time):
        """Save a plot for the current interval only (not cumulative)"""
        import os
        
        # Filter data for current interval only
        interval_frame_numbers = []
        interval_focus_values = []
        interval_times = []
        
        for i, t in enumerate(self.all_times):
            if start_time <= t <= current_time:
                interval_frame_numbers.append(self.all_frame_numbers[i])
                interval_focus_values.append(self.all_focus_values[i])
                interval_times.append(self.all_times[i])
        
        # Skip if no data in this interval
        if not interval_focus_values:
            return
        
        # Create a new figure for saving (don't interfere with display)
        fig_save, ax_save = plt.subplots(1, 1, figsize=(14, 7))
        
        # Plot based on plot_type
        if self.plot_type == 'time':
            x_data = interval_times
            xlabel = 'Time (seconds)'
        else:  # frame
            x_data = interval_frame_numbers
            xlabel = 'Frame Number'
        
        # Plot the main line
        ax_save.plot(x_data, interval_focus_values, 'b-', linewidth=1.5, label='Focus Measure')
        
        # Highlight and label every N seconds (based on marker_interval)
        if self.plot_type == 'time':
            # Find data points at approximately every marker_interval seconds
            second_markers_x = []
            second_markers_y = []
            second_labels = []
            
            # Get seconds at marker_interval within this interval
            start_sec = int(np.ceil(start_time / self.marker_interval) * self.marker_interval)
            end_sec = int(np.floor(current_time / self.marker_interval) * self.marker_interval)
            
            for sec in range(start_sec, end_sec + 1, self.marker_interval):
                # Find the closest data point to this second
                if interval_times:
                    idx = np.argmin([abs(t - sec) for t in interval_times])
                    closest_time = interval_times[idx]
                    
                    # Only use if it's reasonably close (within 0.5 seconds)
                    if abs(closest_time - sec) < 0.5:
                        second_markers_x.append(closest_time)
                        second_markers_y.append(interval_focus_values[idx])
                        second_labels.append(f'{sec}s: {interval_focus_values[idx]:.1f}')
        else:
            # For frame-based plots, mark every Nth frame that corresponds to ~marker_interval seconds
            frames_per_second = self.fps
            second_markers_x = []
            second_markers_y = []
            second_labels = []
            
            if frames_per_second > 0:
                # Find frames that are approximately at each marker_interval
                start_sec = int(np.ceil(start_time / self.marker_interval) * self.marker_interval)
                end_sec = int(np.floor(current_time / self.marker_interval) * self.marker_interval)
                
                for sec in range(start_sec, end_sec + 1, self.marker_interval):
                    target_frame = int(sec * frames_per_second)
                    
                    # Find closest actual frame in our data
                    if interval_frame_numbers:
                        idx = np.argmin([abs(f - target_frame) for f in interval_frame_numbers])
                        closest_frame = interval_frame_numbers[idx]
                        
                        # Only use if reasonably close
                        if abs(closest_frame - target_frame) < frames_per_second * 0.5:
                            second_markers_x.append(closest_frame)
                            second_markers_y.append(interval_focus_values[idx])
                            second_labels.append(f'{sec}s: {interval_focus_values[idx]:.1f}')
        
        # Plot markers for each second
        if second_markers_x:
            marker_label = f'{self.marker_interval}-second intervals' if self.marker_interval > 1 else '1-second intervals'
            ax_save.scatter(second_markers_x, second_markers_y, 
                          color='red', s=50, zorder=5, label=marker_label)
            
            # Add text labels for each marker with alternating positions to reduce crowding
            for i, (x, y, label) in enumerate(zip(second_markers_x, second_markers_y, second_labels)):
                # Extract just the focus value (remove time prefix like "62s: ")
                focus_value = label.split(': ')[1] if ': ' in label else label
                
                # Alternate label positions: above for even indices, below for odd
                if i % 2 == 0:
                    xytext = (0, 12)  # Above the point
                    va = 'bottom'
                else:
                    xytext = (0, -12)  # Below the point
                    va = 'top'
                
                ax_save.annotate(focus_value, 
                               xy=(x, y), 
                               xytext=xytext,
                               textcoords='offset points',
                               fontsize=6,  # Smaller font
                               ha='center',
                               va=va,
                               bbox=dict(boxstyle='round,pad=0.2',  # Smaller padding
                                       facecolor='yellow', 
                                       alpha=0.7, 
                                       edgecolor='gray',
                                       linewidth=0.5),
                               zorder=6)
        
        ax_save.set_xlabel(xlabel, fontsize=12)
        ax_save.set_ylabel('Focus Measure', fontsize=12)
        ax_save.set_title(
            f'Focus Measure vs {self.plot_type.title()} - {self.method.title()}\n'
            f'Video time: {start_time:.1f}s to {current_time:.1f}s (Interval {self.save_counter})',
            fontsize=14, fontweight='bold'
        )
        ax_save.grid(True, alpha=0.3)
        ax_save.legend(loc='best', fontsize=10)
        
        # Add statistics for this interval
        if interval_focus_values:
            mean_focus = np.mean(interval_focus_values)
            max_focus = np.max(interval_focus_values)
            min_focus = np.min(interval_focus_values)
            std_focus = np.std(interval_focus_values)
            stats_text = f'Mean: {mean_focus:.1f} | Max: {max_focus:.1f} | Min: {min_focus:.1f} | Std: {std_focus:.1f}'
            ax_save.text(0.98, 0.98, stats_text, transform=ax_save.transAxes,
                        verticalalignment='top', horizontalalignment='right', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # Save the figure
        filename = f'plot_interval_{self.save_counter:03d}_time_{start_time:.0f}s-{current_time:.0f}s.png'
        filepath = os.path.join(self.output_dir, filename)
        fig_save.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_save)
        
        print(f"  → Saved plot: {filename}")
        self.save_counter += 1
    
    def save_parameters_json(self):
        """Save experiment parameters to a JSON file"""
        import os
        import json
        from datetime import datetime
        
        # Gather all parameters
        parameters = {
            "experiment_info": {
                "analysis_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "script_version": "video_with_graph.py v2.0"
            },
            "input_video": {
                "video_path": self.video_path,
                "video_filename": os.path.basename(self.video_path),
                "resolution": f"{self.width}x{self.height}",
                "width_pixels": self.width,
                "height_pixels": self.height,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "duration_seconds": self.total_frames / self.fps if self.fps > 0 else 0,
                "duration_formatted": f"{int((self.total_frames / self.fps) / 60)}m {int((self.total_frames / self.fps) % 60)}s" if self.fps > 0 else "N/A"
            },
            "processing_parameters": {
                "focus_measure_method": self.method,
                "focus_measure_description": {
                    "laplacian": "Variance of Laplacian - measures edge sharpness using second derivatives",
                    "tenengrad": "Tenengrad - based on gradient magnitude using Sobel operator",
                    "variance": "Normalized variance of pixel intensities",
                    "modified_laplacian": "Modified Laplacian - uses two 1D second derivative kernels, more sensitive to directional edges",
                    "sum_modified_laplacian": "Sum Modified Laplacian - sums ML values above threshold, focuses on sharp edges only"
                }.get(self.method, "Unknown method"),
                "crop_factor": self.crop_factor,
                "crop_enabled": self.crop_factor != (1, 1),
                "crop_width": self.crop_width,
                "crop_height": self.crop_height,
                "crop_description": f"Centered {self.crop_width}x{self.crop_height} region" if self.crop_factor != (1, 1) else "Full frame (no crop)",
                "crop_boundaries": {
                    "x_start": self.crop_x1,
                    "x_end": self.crop_x2,
                    "y_start": self.crop_y1,
                    "y_end": self.crop_y2
                },
                "downsample_rate": self.downsample,
                "frames_processed": f"Every {self.downsample} frame(s)" if self.downsample > 1 else "Every frame",
                "effective_sampling_rate_fps": self.fps / self.downsample if self.fps > 0 and self.downsample > 0 else 0,
                "total_frames_analyzed": self.total_frames // self.downsample if self.downsample > 0 else self.total_frames
            },
            "plot_settings": {
                "plot_type": self.plot_type,
                "plot_type_description": "Focus Measure vs Time (seconds)" if self.plot_type == "time" else "Focus Measure vs Frame Number",
                "save_interval_minutes": self.save_interval,
                "save_interval_seconds": self.save_interval * 60 if self.save_interval else None,
                "display_buffer_size": self.buffer_size,
                "display_mode": self.display_mode,
                "display_mode_description": "Side-by-side (video left, graph right)" if self.display_mode == "sidebyside" else "Overlay (graph on video)"
            },
            "visualization_features": {
                "one_second_markers": True,
                "marker_interval_seconds": self.marker_interval,
                "marker_description": f"Red dots and yellow labels showing focus measure value every {self.marker_interval} second(s)",
                "label_style": "Alternating above/below with compact styling",
                "statistics_on_plots": True,
                "statistics_included": ["Mean", "Max", "Min", "Standard Deviation"]
            },
            "output_settings": {
                "output_directory": self.output_dir,
                "plot_format": "PNG",
                "plot_dpi": 150,
                "plot_size_inches": "14 x 7",
                "individual_interval_plots": True,
                "interval_plot_description": "Each plot contains data from a single time interval (non-cumulative)"
            },
            "playback_settings": {
                "playback_speed_multiplier": self.playback_speed,
                "actual_playback_speed": f"{self.playback_speed}x" if self.playback_speed != 1.0 else "Normal speed"
            }
        }
        
        # Save to JSON file
        json_path = os.path.join(self.output_dir, 'parameters.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parameters, f, indent=4, ensure_ascii=False)
        
        print(f"  → Saved parameters: parameters.json")
    
    def update_sidebyside(self, frame_data):
        """Update side-by-side display"""
        if frame_data is None:
            return [self.im_video, self.line, self.current_point]
        
        frame, frame_num, focus_value, time_sec = frame_data
        
        if frame is None:
            return [self.im_video, self.line, self.current_point]
        
        # Update video display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.im_video.set_data(frame_rgb)
        
        # Update data
        self.frame_numbers.append(frame_num)
        self.focus_values.append(focus_value)
        self.times.append(time_sec)
        
        # Store all data for saving (not limited by buffer)
        self.all_frame_numbers.append(frame_num)
        self.all_focus_values.append(focus_value)
        self.all_times.append(time_sec)
        
        # Check if we should save a plot
        if self.save_interval is not None:
            time_since_last_save = time_sec - self.last_save_time
            if time_since_last_save >= (self.save_interval * 60):  # Convert minutes to seconds
                self.save_interval_plot(time_sec, self.last_save_time)
                self.last_save_time = time_sec
        
        # Update graph based on plot_type
        if self.plot_type == 'time':
            x_data = list(self.times)
            x_current = time_sec
        else:  # frame
            x_data = list(self.frame_numbers)
            x_current = frame_num
        
        self.line.set_data(x_data, list(self.focus_values))
        self.current_point.set_data([x_current], [focus_value])
        
        # Auto-scale graph
        self.ax_graph.relim()
        self.ax_graph.autoscale_view()
        
        # Update title with info
        progress_pct = (frame_num / self.total_frames) * 100
        self.ax_video.set_title(
            f'Frame: {frame_num}/{self.total_frames} ({progress_pct:.1f}%) | '
            f'Time: {time_sec:.1f}s | Focus: {focus_value:.1f}'
        )
        
        return [self.im_video, self.line, self.current_point]
    
    def create_graph_image(self):
        """Create graph as an image that can be overlaid on video"""
        # Draw the current graph
        self.fig.canvas.draw()
        
        # Convert to numpy array (compatible with newer matplotlib versions)
        try:
            # Try newer method first
            graph_img = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            graph_img = graph_img.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB
            graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
        except AttributeError:
            # Fallback to older method
            graph_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
        
        return graph_img
    
    def overlay_graph_on_video(self, frame, focus_value, time_sec, frame_num):
        """Overlay focus graph on video frame"""
        # Update graph data
        self.frame_numbers.append(frame_num)
        self.focus_values.append(focus_value)
        self.times.append(time_sec)
        
        # Store all data for saving (not limited by buffer)
        self.all_frame_numbers.append(frame_num)
        self.all_focus_values.append(focus_value)
        self.all_times.append(time_sec)
        
        # Update graph based on plot_type
        if self.plot_type == 'time':
            x_data = list(self.times)
            x_current = time_sec
        else:  # frame
            x_data = list(self.frame_numbers)
            x_current = frame_num
        
        self.line.set_data(x_data, list(self.focus_values))
        self.current_point.set_data([x_current], [focus_value])
        self.ax_graph.relim()
        self.ax_graph.autoscale_view()
        
        # Set title with extra padding to prevent clipping
        self.ax_graph.set_title(f'Focus: {focus_value:.1f}', fontsize=10, pad=15)
        
        # Get graph as image
        graph_img = self.create_graph_image()
        
        # Resize graph to fit on video
        graph_height = int(self.height * 0.25)
        graph_width = int(graph_height * graph_img.shape[1] / graph_img.shape[0])
        graph_resized = cv2.resize(graph_img, (graph_width, graph_height))
        
        # Create a copy of the frame
        result = frame.copy()
        
        # Position for overlay (bottom-right corner with margin)
        margin = 20
        y_start = self.height - graph_height - margin
        x_start = self.width - graph_width - margin
        
        # Add semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, 
                     (x_start - 10, y_start - 10),
                     (x_start + graph_width + 10, y_start + graph_height + 10),
                     (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
        
        # Overlay the graph
        result[y_start:y_start + graph_height, x_start:x_start + graph_width] = graph_resized
        
        # Add text info
        info_text = f'Frame: {frame_num}/{self.total_frames} | Time: {time_sec:.1f}s'
        cv2.putText(result, info_text, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        progress_pct = (frame_num / self.total_frames) * 100
        progress_text = f'Progress: {progress_pct:.1f}%'
        cv2.putText(result, progress_text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def frame_generator_sidebyside(self):
        """Generator for side-by-side display"""
        frame_num = 0
        
        while frame_num < self.total_frames:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame (optionally downsample)
                if frame_num % self.downsample == 0:
                    time_sec = frame_num / self.fps if self.fps > 0 else frame_num
                    focus_value = self.process_frame(frame)
                    
                    yield (frame, frame_num, focus_value, time_sec)
                
                frame_num += 1
            else:
                yield None
                time.sleep(0.01)
        
        # Keep displaying last frame
        while True:
            yield None
    
    def run_sidebyside(self, interval=1):
        """Run side-by-side display with animation"""
        print(f"Processing video: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        if self.crop_factor != (1, 1):
            print(f"Crop region: {self.crop_width}x{self.crop_height} (centered, factor: {self.crop_factor[0]}x{self.crop_factor[1]})")
            print(f"Crop boundaries: x=[{self.crop_x1}:{self.crop_x2}], y=[{self.crop_y1}:{self.crop_y2}]")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Duration: {self.total_frames/self.fps:.1f} seconds")
        print(f"Focus method: {self.method}")
        print(f"Plot type: Focus Measure vs {self.plot_type.title()}")
        if self.downsample > 1:
            print(f"Downsampling: Processing every {self.downsample} frame(s)")
        if self.save_interval is not None:
            print(f"Auto-save: Every {self.save_interval} minute(s)")
        print("\nClose the window to stop.")
        
        anim = FuncAnimation(
            self.fig,
            self.update_sidebyside,
            frames=self.frame_generator_sidebyside(),
            interval=interval,
            blit=True,
            repeat=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nPlayback interrupted.")
        finally:
            # Save final plot if auto-save is enabled and we have data
            if self.save_interval is not None and self.all_times:
                final_time = self.all_times[-1]
                if final_time - self.last_save_time > 0:  # If there's unsaved data
                    print("\nSaving final plot...")
                    self.save_interval_plot(final_time, self.last_save_time)
            
            self.cap.release()
    
    def run_overlay(self):
        """Run overlay display using OpenCV"""
        print(f"Processing video: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        if self.crop_factor != (1, 1):
            print(f"Crop region: {self.crop_width}x{self.crop_height} (centered, factor: {self.crop_factor[0]}x{self.crop_factor[1]})")
            print(f"Crop boundaries: x=[{self.crop_x1}:{self.crop_x2}], y=[{self.crop_y1}:{self.crop_y2}]")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Duration: {self.total_frames/self.fps:.1f} seconds")
        print(f"Focus method: {self.method}")
        print(f"Plot type: Focus Measure vs {self.plot_type.title()}")
        if self.downsample > 1:
            print(f"Downsampling: Processing every {self.downsample} frame(s)")
        if self.save_interval is not None:
            print(f"Auto-save: Every {self.save_interval} minute(s)")
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  Q or ESC - Quit")
        print("  S - Save current frame")
        
        frame_num = 0
        paused = False
        
        # Calculate delay for playback speed
        delay_ms = max(1, int((1000 / self.fps) / self.playback_speed))
        
        try:
            while frame_num < self.total_frames:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # Process and overlay
                    if frame_num % self.downsample == 0:
                        time_sec = frame_num / self.fps if self.fps > 0 else frame_num
                        focus_value = self.process_frame(frame)
                        
                        # Check if we should save a plot
                        if self.save_interval is not None:
                            time_since_last_save = time_sec - self.last_save_time
                            if time_since_last_save >= (self.save_interval * 60):  # Convert minutes to seconds
                                self.save_interval_plot(time_sec, self.last_save_time)
                                self.last_save_time = time_sec
                        
                        display_frame = self.overlay_graph_on_video(
                            frame, focus_value, time_sec, frame_num
                        )
                        
                        cv2.imshow('Video with Focus Graph', display_frame)
                    
                    frame_num += 1
                
                # Handle keyboard input
                key = cv2.waitKey(delay_ms if not paused else 100) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('s'):  # S - save frame
                    save_path = f'frame_{frame_num:06d}.png'
                    cv2.imwrite(save_path, display_frame)
                    print(f"Saved frame to {save_path}")
        
        except KeyboardInterrupt:
            print("\nPlayback interrupted.")
        finally:
            # Save final plot if auto-save is enabled and we have data
            if self.save_interval is not None and self.all_times:
                final_time = self.all_times[-1]
                if final_time - self.last_save_time > 0:  # If there's unsaved data
                    print("\nSaving final plot...")
                    self.save_interval_plot(final_time, self.last_save_time)
            
            self.cap.release()
            cv2.destroyAllWindows()
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(
        description='Display video with live focus measure graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Side-by-side display (default)
  python video_with_graph.py video.mp4
  
  # Overlay graph on video
  python video_with_graph.py video.mp4 --overlay
  
  # Plot focus vs frame number instead of time
  python video_with_graph.py video.mp4 --plot-type frame
  
  # Save plots every 1 minute (saved to 'processed' folder next to video)
  python video_with_graph.py video.mp4 --save-interval 1.0
  
  # Crop to center quarter of frame (width/4, height/4)
  python video_with_graph.py video.mp4 --crop-width-factor 4 --crop-height-factor 4
  
  # Crop to center 480x270 region from 1920x1080 video
  python video_with_graph.py video.mp4 --crop-width-factor 4 --crop-height-factor 4 --save-interval 1.0
  
  # Complete example: cropped, overlay, save every minute
  python video_with_graph.py video.mp4 --crop-width-factor 2 --crop-height-factor 2 --overlay --save-interval 1.0 --downsample 3
  
  # Try Modified Laplacian method
  python video_with_graph.py video.mp4 --method modified_laplacian --save-interval 1.0
  
  # Try Sum Modified Laplacian method
  python video_with_graph.py video.mp4 --method sum_modified_laplacian --save-interval 1.0
  
  # Faster playback
  python video_with_graph.py video.mp4 --speed 2.0
  
Note: By default, plots are saved to a 'processed' folder in the same directory as your video.
      Use --output-dir to specify a different location.
      Crop factors: --crop-width-factor 4 --crop-height-factor 4 means process center (width/4 x height/4) region.
        """
    )
    
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--method', '-m',
                       choices=['laplacian', 'tenengrad', 'variance', 'modified_laplacian', 'sum_modified_laplacian'],
                       default='laplacian',
                       help='Focus measure method (default: laplacian)')
    parser.add_argument('--overlay', action='store_true',
                       help='Overlay graph on video instead of side-by-side')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--downsample', '-d', type=int, default=1,
                       help='Process every Nth frame for speed (default: 1)')
    parser.add_argument('--buffer', type=int, default=500,
                       help='Maximum frames to display in graph (default: 500)')
    parser.add_argument('--plot-type', choices=['time', 'frame'],
                       default='time',
                       help='Plot focus measure vs time or frame number (default: time)')
    parser.add_argument('--save-interval', type=float, default=None,
                       help='Save plot every N minutes (e.g., 1.0 for every minute). If not specified, plots are not saved.')
    parser.add_argument('--output-dir', default='focus_plots',
                       help='Directory to save interval plots. Default: "processed" folder next to input video. Use this option to specify a custom location.')
    parser.add_argument('--marker-interval', type=int, default=2,
                       help='Interval in seconds between markers on saved plots (default: 2). Use 1 for every second, 5 for every 5 seconds, etc.')
    parser.add_argument('--crop-width-factor', type=int, default=1,
                       help='Crop width divisor (m). Crops to (width/m) centered region. Default: 1 (full width). Example: 4 means width/4')
    parser.add_argument('--crop-height-factor', type=int, default=1,
                       help='Crop height divisor (n). Crops to (height/n) centered region. Default: 1 (full height). Example: 4 means height/4')
    
    args = parser.parse_args()
    
    try:
        display_mode = 'overlay' if args.overlay else 'sidebyside'
        
        analyzer = VideoWithFocusGraph(
            args.video,
            method=args.method,
            buffer_size=args.buffer,
            display_mode=display_mode,
            playback_speed=args.speed,
            downsample=args.downsample,
            plot_type=args.plot_type,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            marker_interval=args.marker_interval,
            crop_factor=(args.crop_width_factor, args.crop_height_factor)
        )
        
        if display_mode == 'overlay':
            analyzer.run_overlay()
        else:
            analyzer.run_sidebyside(interval=1)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
