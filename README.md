# Traffic Analysis with YOLOv8
## Overview
This Python script analyzes traffic in a YouTube video using YOLOv8 for vehicle detection and tracking. It downloads a video, detects vehicles (cars, motorcycles, buses, trucks), assigns them to three lanes, counts vehicles crossing a virtual line, and generates an annotated video and CSV file. The script runs for 30 seconds to limit processing time.
Features

## Downloads YouTube video using yt_dlp.
Uses YOLOv8 with BoT-SORT for vehicle detection and tracking.
Assigns vehicles to three lanes via polygon checks.
Counts vehicles crossing a horizontal line, avoiding duplicates.
Annotates video with lanes, bounding boxes, track IDs, and counts.
Saves tracking data (vehicle ID, lane, frame, timestamp) to vehicle_tracks.csv.
Outputs annotated video to output_video.mp4.
Prints vehicle counts per lane.

## Requirements

Python 3.8+
Libraries: opencv-python, numpy, pandas, ultralytics, yt-dlp
YOLOv8 weights (yolov8m.pt) from Ultralytics
Internet access for video download
mp4v codec for video output

## Installation

Install dependencies:pip install opencv-python numpy pandas ultralytics yt-dlp


Download yolov8m.pt from Ultralytics.

## Usage

Run:python traffic_analysis.py


### The script:
Downloads video from https://www.youtube.com/watch?v=MNn9qKG2UFI.
Processes for 30 seconds.
Saves output_video.mp4 and vehicle_tracks.csv.
Prints lane counts and deletes input video.



## Input

Video URL: Hardcoded YouTube URL.
Assumptions: Overhead view, three vertical lanes, downward movement.

## Output

output_video.mp4: Annotated video with lanes and counts.
vehicle_tracks.csv: Columns: Vehicle_ID, Lane, Frame, Timestamp.
Console: Frame progress and lane count summary.

## Notes

Adjust lane polygons for different video perspectives.
Requires internet and disk space.
