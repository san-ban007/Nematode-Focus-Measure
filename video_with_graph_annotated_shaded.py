#!/usr/bin/env python3
"""
Focus Measure Analyzer with Live Video Display and CSV Event Annotations
Shows the video alongside the focus measure graph in real-time with event markers
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import argparse
import time
import pandas as pd
import os


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
    """Display video alongside live focus measure graph with CSV event annotations"""
    
    def __init__(self, video_path, csv_path=None, method='laplacian', buffer_size=500, 
                 display_mode='sidebyside', playback_speed=1.0, downsample=1,
                 plot_type='time', save_interval=None, output_dir='focus_plots',
                 marker_interval=1, crop_factor=(1, 1)):
        self.video_path = video_path
        self.csv_path = csv_path
        self.method = method
        self.buffer_size = buffer_size
        self.display_mode = display_mode  # 'sidebyside' or 'overlay'
        self.playback_speed = playback_speed
        self.downsample = downsample  # Process every Nth frame for speed
        self.plot_type = plot_type  # 'time' or 'frame'
        self.save_interval = save_interval  # Minutes between saves
        self.marker_interval = marker_interval  # Seconds between markers (1, 2, 5, etc.)
        self.crop_factor = crop_factor  # (m, n) where crop is (width/m, height/n)
        
        # Load CSV events if provided
        self.events = None
        self.event_regions = []  # Initialize event regions list
        if csv_path:
            # Convert to absolute path if relative
            if not os.path.isabs(csv_path):
                csv_path = os.path.abspath(csv_path)
            
            print(f"\nLooking for CSV file at: {csv_path}")
            
            if os.path.exists(csv_path):
                self.load_events(csv_path)
            else:
                print(f"WARNING: CSV file not found at: {csv_path}")
                print(f"Current working directory: {os.getcwd()}")
                # Try looking in the video directory
                video_dir = os.path.dirname(os.path.abspath(video_path))
                csv_in_video_dir = os.path.join(video_dir, os.path.basename(csv_path))
                if os.path.exists(csv_in_video_dir):
                    print(f"Found CSV in video directory: {csv_in_video_dir}")
                    self.load_events(csv_in_video_dir)
                else:
                    print(f"CSV also not found in video directory: {csv_in_video_dir}")
        else:
            print("\nNo CSV file specified - running without event annotations")
        
        # Determine output directory
        # If output_dir is the default 'focus_plots', create folder with video name and timestamp
        # Otherwise use the user-specified directory
        if output_dir == 'focus_plots':
            # Get the directory containing the video and video filename
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
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.focus_methods.keys())}")
        
        self.focus_func = self.focus_methods[method]
        
        # Setup matplotlib figure based on display mode
        if display_mode == 'overlay':
            self.setup_overlay_figure()
        else:
            self.setup_sidebyside_figure()
    
    def load_events(self, csv_path):
        """Load focus events from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            print(f"\nLoaded CSV with {len(df)} total entries")
            
            # Keep all events for shaded regions
            self.events = df
            
            # Parse event pairs for shaded regions
            self.event_regions = self.parse_event_regions(df)
            
            print(f"Loaded events from CSV:")
            print(f"  - Total events: {len(df)}")
            print(f"  - Event regions parsed: {len(self.event_regions)}")
            
        except Exception as e:
            print(f"Warning: Could not load CSV events: {e}")
            import traceback
            traceback.print_exc()
            self.events = None
            self.event_regions = []
    
    def parse_event_regions(self, df):
        """Parse start/end event pairs into regions for shading"""
        regions = []
        
        # Event type patterns to match (will match with scan prefixes like scan1_, scan2_)
        # Order matters: check more specific patterns first to avoid substring matches
        event_patterns = [
            ('line_position', 'green', '-', 'Line Position'),
            ('first_focus_down', 'red', '--', 'First Focus Down'),  # Must come before focus_down
            ('x_move', 'yellow', '-', 'X Move'),
            ('focus_up', 'blue', '-', 'Focus Up'),
            ('focus_down', 'red', '-', 'Focus Down')  # Must come after first_focus_down
        ]
        
        for pattern, color, linestyle, label in event_patterns:
            # Find all start events for this pattern
            # For focus_down, we need to explicitly exclude first_focus_down
            if pattern == 'focus_down':
                # Only match events that have focus_down but NOT first_focus_down
                start_events = df[
                    df['event'].str.contains(f'{pattern}_start', na=False, regex=False) &
                    ~df['event'].str.contains('first_focus_down', na=False, regex=False)
                ].copy()
            else:
                start_events = df[df['event'].str.contains(f'{pattern}_start', na=False, regex=False)].copy()
            
            # For events with duplicate names (like x_move), pair them sequentially
            if len(start_events) > 0:
                # Get the event name pattern (e.g., "x_move_start")
                first_event_name = start_events.iloc[0]['event']
                end_event_name = first_event_name.replace('_start', '_end')
                
                # Check if this is a repeating event name
                if start_events['event'].nunique() == 1:
                    # All start events have the same name, so we need sequential pairing
                    end_events = df[df['event'] == end_event_name].copy()
                    
                    # Pair them by order (assume they alternate properly)
                    for idx, (_, start_row) in enumerate(start_events.iterrows()):
                        if idx < len(end_events):
                            start_time = start_row['elapsed_time']
                            end_time = end_events.iloc[idx]['elapsed_time']
                            
                            regions.append({
                                'start': start_time,
                                'end': end_time,
                                'type': pattern,
                                'color': color,
                                'linestyle': linestyle,
                                'label': label
                            })
                else:
                    # Different event names (like scan1_line_position_start vs scan2_line_position_start)
                    # Match each start with its specific end
                    for _, start_row in start_events.iterrows():
                        start_time = start_row['elapsed_time']
                        start_event = start_row['event']
                        end_event_name = start_event.replace('_start', '_end')
                        end_row = df[df['event'] == end_event_name]
                        
                        if not end_row.empty:
                            end_time = end_row.iloc[0]['elapsed_time']
                            
                            regions.append({
                                'start': start_time,
                                'end': end_time,
                                'type': pattern,
                                'color': color,
                                'linestyle': linestyle,
                                'label': label
                            })
        
        return regions
    
    def add_event_regions(self, ax, x_min=None, x_max=None, for_save=False):
        """Add shaded regions for event pairs to the plot"""
        if not hasattr(self, 'event_regions') or not self.event_regions:
            return
        
        # Filter regions based on x range if provided
        regions_to_plot = [r for r in self.event_regions 
                          if x_min is None or x_max is None or 
                          (r['start'] <= x_max and r['end'] >= x_min)]
        
        # Track which event types we've seen for legend
        seen_types = set()
        
        # Define which event types should have shading - ALL events since they don't overlap
        shaded_events = {'line_position', 'first_focus_down', 'x_move', 'focus_up', 'focus_down'}
        
        for region in regions_to_plot:
            start = region['start']
            end = region['end']
            color = region['color']
            linestyle = region['linestyle']
            label = region['label']
            event_type = region['type']
            
            # Only add shaded region for specific event types
            if event_type in shaded_events:
                ax.axvspan(start, end, alpha=0.2, color=color, zorder=1)
            
            # Add boundary lines for ALL event types
            ax.axvline(x=start, color=color, linestyle=linestyle, alpha=0.8, linewidth=1.5, zorder=10)
            ax.axvline(x=end, color=color, linestyle=linestyle, alpha=0.8, linewidth=1.5, zorder=10)
            
            seen_types.add(event_type)
        
        # Add legend only for saved plots (not for live overlay view)
        should_add_legend = len(regions_to_plot) > 0 and for_save
        if not for_save and hasattr(self, 'display_mode') and self.display_mode == 'sidebyside':
            should_add_legend = False  
        
        if should_add_legend:
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_elements = []
            
            # Define legend order 
            legend_order = [
                ('line_position', 'green', '-', 'Line Position', True),
                ('first_focus_down', 'red', '--', 'First Focus Down', True),
                ('x_move', 'yellow', '-', 'X Move', True),
                ('focus_down', 'red', '-', 'Focus Down', True),
                ('focus_up', 'blue', '-', 'Focus Up', True)
            ]
            
            for event_type, color, linestyle, label, is_shaded in legend_order:
                if event_type in seen_types:
                    if is_shaded:
                        # Create patch for shaded regions
                        patch = Patch(facecolor=color, alpha=0.2, edgecolor=color, 
                                    linewidth=1.5, linestyle=linestyle, label=label)
                        legend_elements.append(patch)
                    else:
                        # Create line for boundary-only events
                        line = Line2D([0], [0], color=color, linestyle=linestyle, 
                                    linewidth=1.5, label=label)
                        legend_elements.append(line)
            
            if legend_elements:
                # Get existing handles and labels for the focus measure line
                handles, labels = ax.get_legend_handles_labels()
                # Add to legend
                ax.legend(handles=handles + legend_elements, loc='upper right', fontsize=10)
    
    
    def setup_overlay_figure(self):
        """Setup matplotlib figure for overlay mode"""
        # Use OpenCV window for video
        cv2.namedWindow('Video with Focus Graph', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video with Focus Graph', 1280, 720)
        
        # Matplotlib for mini graph (we'll embed this)
        # Taller figure to prevent title clipping
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 2.6))
        self.line, = self.ax.plot([], [], 'g-', linewidth=2, label='Focus Measure')
        self.current_point, = self.ax.plot([], [], 'ro', markersize=8)
        
        # Set labels based on plot type
        if self.plot_type == 'time':
            self.ax.set_xlabel('Time (s)', fontsize=8)
        else:  # frame
            self.ax.set_xlabel('Frame', fontsize=8)
        
        self.ax.set_ylabel('Focus', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent title clipping
        # Leave substantial space at the top for the title
        plt.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.95)
    
    def setup_sidebyside_figure(self):
        """Setup matplotlib figure for side-by-side display"""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.line, = self.ax.plot([], [], 'g-', linewidth=1.5, label='Focus Measure')
        
        xlabel = 'Time (seconds)' if self.plot_type == 'time' else 'Frame Number'
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel('Focus Measure')
        self.ax.set_title(f'Focus Measure ({self.method})')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        plt.tight_layout()
    
    def process_frame(self, frame):
        """Process frame and return focus measure"""
        # Crop frame if needed
        if self.crop_factor != (1, 1):
            frame = frame[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
        
        return self.focus_func(frame)
    
    def update_graph(self, focus_value, time_sec, frame_num):
        """Update the graph with new data point"""
        x_value = time_sec if self.plot_type == 'time' else frame_num
        
        # Add to display buffers
        self.times.append(time_sec)
        self.frame_numbers.append(frame_num)
        self.focus_values.append(focus_value)
        
        # Also add to permanent storage
        self.all_times.append(time_sec)
        self.all_frame_numbers.append(frame_num)
        self.all_focus_values.append(focus_value)
        
        # Update plot data
        x_data = list(self.times) if self.plot_type == 'time' else list(self.frame_numbers)
        self.line.set_data(x_data, list(self.focus_values))
        self.current_point.set_data([x_value], [focus_value])
        
        # Auto-scale axes
        if len(x_data) > 0:
            self.ax.set_xlim(min(x_data), max(x_data) * 1.05)
            
            y_data = list(self.focus_values)
            y_min, y_max = min(y_data), max(y_data)
            y_range = y_max - y_min
            margin = 0.1 * y_range if y_range > 0 else 0.1
            self.ax.set_ylim(y_min - margin, y_max + margin)
            
            # Clear previous event regions - keep only the focus measure line and current point
            # Store references to lines we want to keep
            lines_to_keep = [self.line, self.current_point]
            
            # Remove all other artists (lines, patches, collections)
            for line in self.ax.lines[:]:
                if line not in lines_to_keep:
                    line.remove()
            
            # Remove patches (shaded regions)
            for patch in self.ax.patches[:]:
                patch.remove()
            
            # Remove collections (if any)
            for collection in self.ax.collections[:]:
                collection.remove()
            
            # Add event regions for current view
            self.add_event_regions(self.ax, min(x_data), max(x_data))
    
    
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
    
    def save_parameters_json(self):
        """Save analysis parameters to JSON file"""
        import json
        from datetime import datetime
        
        params = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'video_path': os.path.abspath(self.video_path),
                'csv_path': os.path.abspath(self.csv_path) if self.csv_path else None,
                'output_directory': os.path.abspath(self.output_dir)
            },
            'video_properties': {
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0
            },
            'crop_settings': {
                'crop_factor': self.crop_factor,
                'crop_width': self.crop_width,
                'crop_height': self.crop_height,
                'crop_boundaries': {
                    'x1': self.crop_x1,
                    'y1': self.crop_y1,
                    'x2': self.crop_x2,
                    'y2': self.crop_y2
                }
            },
            'analysis_settings': {
                'focus_method': self.method,
                'plot_type': self.plot_type,
                'downsample': self.downsample,
                'buffer_size': self.buffer_size,
                'save_interval_minutes': self.save_interval,
                'marker_interval_seconds': self.marker_interval
            },
            'event_annotations': {
                'csv_loaded': self.events is not None,
                'total_events': len(self.events) if self.events is not None else 0
            }
        }
        
        output_path = os.path.join(self.output_dir, 'parameters.json')
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"Parameters saved to: {output_path}")
    
    def save_interval_plot(self, current_time, start_time):
        """Save a plot for a time interval"""
        if not self.all_times:
            return
        
        # Create new figure for saving
        fig_save, ax_save = plt.subplots(figsize=(12, 6))
        
        # Get data for this interval
        x_data = [t for t in self.all_times if start_time <= t <= current_time]
        indices = [i for i, t in enumerate(self.all_times) if start_time <= t <= current_time]
        y_data = [self.all_focus_values[i] for i in indices]
        
        if not x_data:
            plt.close(fig_save)
            return
        
        # Plot the data
        ax_save.plot(x_data, y_data, 'g-', linewidth=1.5, label='Focus Measure')
        
        # Add event regions for this interval (with legend for saved plots)
        self.add_event_regions(ax_save, start_time, current_time, for_save=True)
        
        # Formatting
        xlabel = 'Time (seconds)' if self.plot_type == 'time' else 'Frame Number'
        ax_save.set_xlabel(xlabel, fontsize=12)
        ax_save.set_ylabel('Focus Measure', fontsize=12)
        
        # Calculate interval label in seconds
        start_sec = int(start_time)
        end_sec = int(current_time)
        
        title = f'Focus Measure ({self.method}) - Time Interval: {start_sec}-{end_sec}s'
        ax_save.set_title(title, fontsize=14)
        
        ax_save.grid(True, alpha=0.3)
        
        # Set explicit x-axis limits to match the interval exactly (no extra padding)
        ax_save.set_xlim(start_time, current_time)
        
        # Add x-axis markers at specified intervals
        if self.marker_interval > 0:
            marker_times = []
            t = int(start_time / self.marker_interval) * self.marker_interval
            while t <= current_time:
                if t >= start_time:
                    marker_times.append(t)
                t += self.marker_interval
            
            if marker_times:
                ax_save.set_xticks(marker_times)
                ax_save.set_xticklabels([f'{t:.0f}' for t in marker_times])
        
        plt.tight_layout()
        
        # Save the plot
        self.save_counter += 1
        filename = f'focus_plot_{self.save_counter:03d}_t{start_sec:04d}-{end_sec:04d}s.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_save)
        
        print(f"Saved interval plot: {filename}")
    
    def overlay_graph_on_video(self, frame, focus_value, time_sec, frame_num):
        """Overlay focus graph on video frame with event markers"""
        # Update graph data
        self.update_graph(focus_value, time_sec, frame_num)
        
        # Set title with extra padding to prevent clipping
        self.ax.set_title(f'Focus: {focus_value:.1f}', fontsize=10, pad=15)
        
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
        progress_pct = (frame_num / self.total_frames) * 100
        info_text = f'Frame: {frame_num}/{self.total_frames} | Time: {time_sec:.1f}s'
        cv2.putText(result, info_text, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add percentage completion on the line below
        progress_text = f'Progress: {progress_pct:.1f}%'
        cv2.putText(result, progress_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def run_sidebyside(self, interval=1):
        """Run the analyzer with side-by-side display"""
        print("\n=== Video Focus Measure Analysis ===")
        print(f"Video: {self.video_path}")
        if self.csv_path:
            print(f"CSV: {self.csv_path}")
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
        print("  S - Save current plot")
        
        def update(frame_num):
            if frame_num >= self.total_frames:
                return self.line,
            
            # Set video to correct frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if not ret:
                return self.line,
            
            # Process frame
            if frame_num % self.downsample == 0:
                time_sec = frame_num / self.fps if self.fps > 0 else frame_num
                focus_value = self.process_frame(frame)
                
                # Check if we should save a plot
                if self.save_interval is not None:
                    time_since_last_save = time_sec - self.last_save_time
                    if time_since_last_save >= (self.save_interval * 60):
                        self.save_interval_plot(time_sec, self.last_save_time)
                        self.last_save_time = time_sec
                
                self.update_graph(focus_value, time_sec, frame_num)
            
            return self.line,
        
        # Create animation
        delay_ms = max(1, int((1000 / self.fps) / self.playback_speed))
        anim = FuncAnimation(
            self.fig, update, frames=self.total_frames,
            interval=delay_ms, blit=True, repeat=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nPlayback interrupted.")
        finally:
            # Save final plot if auto-save is enabled and we have data
            if self.save_interval is not None and self.all_times:
                final_time = self.all_times[-1]
                if final_time - self.last_save_time > 0:
                    print("\nSaving final plot...")
                    self.save_interval_plot(final_time, self.last_save_time)
            
            self.cap.release()
            cv2.destroyAllWindows()
            plt.close(self.fig)
    
    def run_overlay(self):
        """Run the analyzer with overlay display"""
        print("\n=== Video Focus Measure Analysis (Overlay Mode) ===")
        print(f"Video: {self.video_path}")
        if self.csv_path:
            print(f"CSV: {self.csv_path}")
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
                            if time_since_last_save >= (self.save_interval * 60):
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
                if final_time - self.last_save_time > 0:
                    print("\nSaving final plot...")
                    self.save_interval_plot(final_time, self.last_save_time)
            
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Display video with live focus measure graph and CSV event annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with CSV annotations
  python video_with_graph_annotated.py video.mp4 --csv events.csv
  
  # Save plots every 1 minute with event markers
  python video_with_graph_annotated.py video.mp4 --csv events.csv --save-interval 1.0
  
  # Use modified Laplacian with CSV events
  python video_with_graph_annotated.py video.mp4 --csv events.csv --method modified_laplacian --save-interval 1.0
  
  # Cropped with CSV events
  python video_with_graph_annotated.py video.mp4 --csv events.csv --crop-width-factor 4 --crop-height-factor 4 --save-interval 1.0

Note: CSV file should have columns: elapsed_time, event, details
      Events should include: scan_start, first_focus_down, focus_up, focus_down
        """
    )
    
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--csv', help='Path to CSV file with focus events')
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
                       help='Directory to save interval plots. Default: "processed" folder next to input video.')
    parser.add_argument('--marker-interval', type=int, default=2,
                       help='Interval in seconds between markers on saved plots (default: 2)')
    parser.add_argument('--crop-width-factor', type=int, default=1,
                       help='Crop width divisor (m). Default: 1 (full width)')
    parser.add_argument('--crop-height-factor', type=int, default=1,
                       help='Crop height divisor (n). Default: 1 (full height)')
    
    args = parser.parse_args()
    
    try:
        display_mode = 'overlay' if args.overlay else 'sidebyside'
        
        analyzer = VideoWithFocusGraph(
            args.video,
            csv_path=args.csv,
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
