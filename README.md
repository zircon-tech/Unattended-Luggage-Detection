# Unattended Luggage Detection
This project aims to detect unattended luggage in airports and other public places.
The project is implemented using YOLOv8 object and MiDaS for depth estimation. 

## Requirements
- Python 3.10 or later
- Torch 2.4.0

## Installation
1. Clone the repository
2. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Usage
Run the `main.py` file to start the application. Make sure to modify the `main.py` file to give it a path to a video to process.
You can do this by modifiying the following line:
```python
video_path = "path/to/video.mp4"
```