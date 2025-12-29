
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
