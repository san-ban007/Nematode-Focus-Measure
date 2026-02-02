
# Nematode-Focus-Measure
A Python toolkit for analyzing focus quality in microscopy videos. This tool processes videos frame-by-frame to compute focus measures, helping identify focal stacks and track focus changes over time. Particularly useful for microscopy, time-lapse imaging, and any application where maintaining or analyzing focus is critical. 
# Features
- Real-time focus analysis-Display video alongside live focus measure graphs
- Multiple focus measure algorithms
    - Variance ofLaplacian (default)
    - Tenengrad (gradient-based)
    - Normalized Variance
    - Modified Laplacian
    - Sum Modified Laplacian
- Flexible visualization options:
    - Side-by-side video and graph display
    - Overlay graph on video
    - Plot focus vs. time or frame number
- Automated focal stack detection - Identifies peak focus frames
- Auto-save functionality - Save plots at regular intervals during long recordings
- Region of Interest (ROI) - Analyze specific cropped regions
- Playback controls - Pause, resume, save frames, adjust speed
# Installation
### Using Conda (Recommended) 
Clone the repository
``` bash
git clone https://github.com/san-ban007/Nematode-Focus-Measure.git
```
### Create conda environment
``` bash
conda env create -f environment.yml
```
### Activate environment
```bash
conda activate nematode_env
```
# Quick Start
### Basic Usage - Simple Analysis
Analyze a video and display focus measure plots:
```bash
python simple_example.py your_video.mp4
```
This will:
- Process all frames in the video
- Display two plots (focus vs. frame number and focus vs. time)
- Detect and mark focal stack peaks
- Show the best focus frames

### Advanced Usage - Live Video Display
Display video with real-time focus graph:
```bash
python video_ with _graph.py your_ video.mp4
```
#### Usage Examples
Overlay graph on video
```bash
python video_with_graph.py video.mp4 --overlay
```
Plot focus vs frame number instead of time
```bash
python video_ with _graph.py video.mp4 --plot-type frame
```
Auto-save plots every minute
```bash
python video_with_graph.py video.mp4 --save-interval 1.0
```
Plots are saved to a (processed/) folder next to your video, organized by video name and timestamp.

Analyze specific region (crop to center quarter)
```bash
python video_ with _graph.py video.mp4 --crop-width-factor 4 --crop-height-factor 4
```
Try different focus measure methods
```bash
python video_ with _graph.py video.mp4 --method modified _laplacian
python video_ with _graph.py video.mp4 --method tenengrad
```
Speed up processing (process every 3rd frame)
```bash
python video_ with _graph.py video.mp4 --downsample 3
```
Complete example: cropped region with auto-save
```bash
python video_ with _graph.py video.mp4 \
--crop-width-factor 2 \
--crop-height-factor 2 \
--overlay\
--save-interval 1.0 \
--downsample 3
```

### Event-Annotated Analysis

For videos with corresponding CSV event logs, use video_with_graph_annotated_shaded.py to visualize focus changes alongside experimental events with color-coded shaded regions.
#### Usage
```bash
python video_with_graph_annotated_shaded.py video.mp4 --csv events.csv --save-interval 1.0
```
#### CSV Format
The script expects a CSV with columns: elapsed_time, event, details
Supported event pairs (automatically creates shaded regions):

line_position_start / line_position_end → Green shading
first_focus_down_start / first_focus_down_end → Red shading (dashed boundaries)
x_move_start / x_move_end → Yellow shading
focus_down_start / focus_down_end → Red shading (solid boundaries)
focus_up_start / focus_up_end → Blue shading (solid boundaries)

#### Features

Automatic event pairing from CSV timestamps
Color-coded shaded regions with boundary lines
Legend in saved plots (clean live view without legend)
Supports all standard options: --crop-width-factor, --downsample, --method, etc.

### Peak Focus Clip Extractor

`extract_peak_focus.py` automatically extracts short video clips around the best-focus moment in each focus sweep. It reads the CSV event log to identify all focus sweep ranges (`focus_up`, `focus_down`, and `first_focus_down` events), computes the focus measure frame-by-frame within each range, locates the peak, and uses ffmpeg to extract a clip centred on that peak.

#### Usage

```bash
# Basic usage — 2s clips around each focus peak
python extract_peak_focus.py video.mp4 --csv events.csv

# 1s clips with custom output directory and focus data export
python extract_peak_focus.py video.mp4 --csv events.csv -o output_folder/ --duration 1.0 --save-csv

# With centre-crop and focus plot
python extract_peak_focus.py video.mp4 --csv events.csv --crop-width-factor 4 --crop-height-factor 4 --plot
```

### Options

| Flag | Description |
|------|-------------|
| `--csv` | Path to CSV event file (required) |
| `-o`, `--output-dir` | Output directory for clips (default: `<video>_peak_clips/`) |
| `-m`, `--method` | Focus measure: `laplacian`, `tenengrad`, `variance`, `modified_laplacian`, `sum_modified_laplacian` |
| `-t`, `--duration` | Clip length in seconds around each peak (default: `2.0`) |
| `--crop-width-factor` | Centre-crop width divisor for focus measurement (default: `1`) |
| `--crop-height-factor` | Centre-crop height divisor for focus measurement (default: `1`) |
| `--plot` | Save a PNG plot of all focus sweeps with peaks annotated |
| `--save-csv` | Export per-frame focus values for all ranges |

### Requirements

Requires `ffmpeg` installed in your environment:

```bash
conda install -c conda-forge ffmpeg
```
