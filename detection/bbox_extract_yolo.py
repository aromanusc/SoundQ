import os
import json
import glob
import cv2
from ultralytics import YOLO

class BoundingBoxExtractorYOLO:
    def __init__(self, model_path='yolov8n.pt'):
        import torch
        from ultralytics import YOLO

        # Load YOLOv8 model
        self.model = YOLO(model_path)

    def process_video(self, video_file, dest_path):
        import os
        import cv2
        import json

        video_capture = cv2.VideoCapture(video_file)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_data = []

        for frame_idx in range(frame_count):
            ret, frame = video_capture.read()
            if not ret:
                break

            # Run YOLOv8 inference on the frame
            results = self.model(frame)

            # Parse results
            instances_per_frame = []
            for result in results:
                bbox = [box.xyxy.cpu().numpy().tolist()[0] for box in result.boxes]  # Extract coordinates as a list
                confidence = [float(box.conf.cpu().item()) for box in result.boxes] # Extract confidence
                class_id = [int(box.cls.cpu().item()) for box in result.boxes]  # Extract class ID
                instances_per_frame.append({
                    'pred_boxes': bbox,
                    'score': confidence,
                    'pred_class': class_id
                })
            frame_data.append(instances_per_frame)

        video_capture.release()

        # Save JSON for the video
        frame_data_name = os.path.basename(video_file).replace(".mp4", ".json")
        output_path = os.path.join(dest_path, frame_data_name)
        with open(output_path, 'w') as f:
            json.dump(frame_data, f, indent=4)

if __name__ == "__main__":
    model_path = "yolov8l.pt"  # Path to the YOLOv8 model
    extractor = BoundingBoxExtractorYOLO(model_path)

    video_dev_src = "/scratch/ssd1/audiovisual_datasets/soundq2_dataset/videos" # Provide path to the input video file
    dest_path = "/scratch/ssd1/audiovisual_datasets/soundq2_dataset/video_bbox" # Provide the destination path to the processed JSON bbox files
    os.makedirs(dest_path, exist_ok=True)
    mp4_file_paths = glob.glob(os.path.join(video_dev_src, '**/*.mp4'), recursive=True)
    total_files = len(mp4_file_paths)

    for i, video_file in enumerate(mp4_file_paths):
        print(f"File progress {i+1}/{total_files}: Processing {video_file}")
        extractor.process_video(video_file, dest_path)

