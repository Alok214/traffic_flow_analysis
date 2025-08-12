import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import yt_dlp
import os
from datetime import datetime
import time
start_time = time.time()

# Function to download YouTube video
def download_youtube_video(url, output_path='input_video.mp4'):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

# Function to check if a point is inside a polygon
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

# Main traffic analysis class
class TrafficAnalyzer:
    def __init__(self, video_path, output_video_path='output_video.mp4', model_weights='yolov8m.pt'):
        self.model = YOLO(model_weights)  # Use yolov8m for higher accuracy
        self.cap = cv2.VideoCapture(video_path)
        self.out = None
        self.output_video_path = output_video_path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.lane_polygons = self.define_lanes()
        self.vehicle_counts = {1: 0, 2: 0, 3: 0}
        self.tracked_vehicles = set()  # To prevent duplicate counts
        self.track_data = []
        self.track_history = {}  # To store last y position for line crossing
        self.counting_line_y = self.frame_height // 2  # Horizontal line at middle for counting crossings
        
    def define_lanes(self):
        # Define three lanes as polygons (adjust based on video; assuming overhead view)
        # Tuned for a typical highway scene with 3 lanes
        lane1 = [(0, 0), (self.frame_width//3, 0), (self.frame_width//3, self.frame_height), (0, self.frame_height)]
        lane2 = [(self.frame_width//3, 0), (2*self.frame_width//3, 0), (2*self.frame_width//3, self.frame_height), (self.frame_width//3, self.frame_height)]
        lane3 = [(2*self.frame_width//3, 0), (self.frame_width, 0), (self.frame_width, self.frame_height), (2*self.frame_width//3, self.frame_height)]
        return {1: lane1, 2: lane2, 3: lane3}
    
    def initialize_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, 
                                 (self.frame_width, self.frame_height))
    
    def process_frame(self, frame, frame_number):
        # YOLOv8 detection and tracking with built-in BoT-SORT for improved accuracy
        results = self.model.track(frame, persist=True, conf=0.6, iou=0.5, classes=[2, 3, 5, 7], tracker='botsort.yaml')  # Higher conf, BoT-SORT
        
        # Process tracked objects
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                if conf < 0.6:
                    continue
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Determine lane
                lane_number = None
                for lane_id, polygon in self.lane_polygons.items():
                    if point_in_polygon((center_x, center_y), polygon):
                        lane_number = lane_id
                        break
                
                # Line crossing logic for counting (assume downward flow)
                if lane_number and track_id not in self.tracked_vehicles:
                    last_y = self.track_history.get(track_id, 0)  # Default to 0 if new
                    if last_y < self.counting_line_y <= center_y:
                        self.vehicle_counts[lane_number] += 1
                        self.tracked_vehicles.add(track_id)
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.track_data.append({
                            'Vehicle_ID': track_id,
                            'Lane': lane_number,
                            'Frame': frame_number,
                            'Timestamp': timestamp
                        })
                
                # Update history
                self.track_history[track_id] = center_y
                
                # Draw bounding box and track ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw lanes and vehicle counts
        for lane_id, polygon in self.lane_polygons.items():
            cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (255, 0, 0), 2)
            cv2.putText(frame, f'Lane {lane_id}: {self.vehicle_counts[lane_id]}', 
                       (polygon[0][0] + 10, polygon[0][1] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), (self.frame_width, self.counting_line_y), (0, 255, 255), 2)
        
        return frame
    
    def save_track_data(self, output_csv='vehicle_tracks.csv'):
        df = pd.DataFrame(self.track_data)
        df.to_csv(output_csv, index=False)
    
    def print_summary(self):
        print("\nVehicle Count Summary:")
        for lane_id, count in self.vehicle_counts.items():
            print(f"Lane {lane_id}: {count} vehicles")
    
    def run(self):
        self.initialize_video_writer()
        frame_number = 0
        
        while self.cap.isOpened():
            # if time.time() - start_time > 30:
            #     break
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame, frame_number)
            self.out.write(frame)
            
            # Display for real-time visualization (optional)
            cv2.imshow('Traffic Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_number += 1
            print(f"Processed frame {frame_number}/{self.frame_count}", end='\r')
        
        self.save_track_data()
        self.print_summary()
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

def main():
    video_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
    video_path = download_youtube_video(video_url)
    
    analyzer = TrafficAnalyzer(video_path)
    analyzer.run()
    
    # Clean up
    os.remove(video_path)

if __name__ == "__main__":
    main()