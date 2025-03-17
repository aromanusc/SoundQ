import os
import cv2
import csv
import random
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips
from collections import defaultdict

import yaml
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import *
from audio_spatializer import *


class VisualSynthesizer:
    def __init__(self, input_360_video_path, overlay_video_paths_by_class, overlay_image_paths_by_class=None, total_duration=None, 
                fps=60, dark_background=False, use_blur=True):
        self.input_360_video_path = input_360_video_path
        self.overlay_video_paths_by_class = overlay_video_paths_by_class  # Dictionary mapping class indices to lists of videos
        self.overlay_image_paths_by_class = overlay_image_paths_by_class or {}  # Dictionary mapping class indices to lists of images
        self.total_duration = total_duration # seconds
        self.video_fps = fps      # 30 fps
        self.dark_background = dark_background # whether we want a black background video
        self.use_blur = use_blur # whether we want to apply blur to background
        self.video_blur_kernel = random.randrange(61, 81, 2) # background canvas blur (positive and odd)
        # Open the 360-degree video
        self.cap_360 = cv2.VideoCapture(input_360_video_path)
        self.frame_width = int(self.cap_360.get(3))
        self.frame_height = int(self.cap_360.get(4))
        if self.total_duration:
            self.stream_total_frames = int(self.video_fps * self.total_duration)  # Use original video's length
        else:
            self.stream_total_frames = None
        # Cache for loaded images
        self.image_cache = {}


    def load_predefined_metadata(self, metadata_path):
        """
        Load pre-defined metadata from a CSV file.
        The CSV format should be:
        frame_number,class_index,source_index,azimuth,elevation,distance
        """
        metadata_by_frame = defaultdict(list)
        source_tracks = {}  # Track source durations and paths
    
        with open(metadata_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) <= 5:
                    raise Exception("Ensure the csv metadata has all needed fields")
                
                # Scale frame number to 30fps
                frame_number_30fps = int(row[0]) * 3  # Convert 100ms-frame to 30fps-frame
                class_index = int(row[1])
                source_index = int(row[2])
                azimuth = int(row[3])
                elevation = int(row[4])
                distance = int(float(row[5])*100)
            
                for frame_number in range(frame_number_30fps, frame_number_30fps+3):
                    # Check if the source exists in our tracking dictionary
                    source_id = f"{class_index}_{source_index}"
                    if source_id not in source_tracks:
                        # Randomly decide whether to use an image or video
                        use_image = random.choice([True, False])
                        path = None
                        
                        class_str = str(class_index)
                        
                        # Try to get the asset based on the random choice
                        if use_image and class_str in self.overlay_image_paths_by_class and self.overlay_image_paths_by_class[class_str]:
                            path = random.choice(self.overlay_image_paths_by_class[class_str])
                        elif not use_image and class_str in self.overlay_video_paths_by_class and self.overlay_video_paths_by_class[class_str]:
                            path = random.choice(self.overlay_video_paths_by_class[class_str])
                        else:
                            # Fall back to whatever is available
                            if class_str in self.overlay_video_paths_by_class and self.overlay_video_paths_by_class[class_str]:
                                path = random.choice(self.overlay_video_paths_by_class[class_str])
                                use_image = False
                            elif class_str in self.overlay_image_paths_by_class and self.overlay_image_paths_by_class[class_str]:
                                path = random.choice(self.overlay_image_paths_by_class[class_str])
                                use_image = True
                            else:
                                print(f"Warning: No assets available for class {class_index}")
                                continue
                        
                        source_tracks[source_id] = {
                            'path': path,
                            'is_image': use_image,
                            'start_frame': frame_number,
                            'frames': [frame_number],
                            'azimuth_history': {frame_number: azimuth},
                            'elevation_history': {frame_number: elevation},
                            'distance_history': {frame_number: distance},
                            'class': class_str
                        }
                    else:
                        # Update the existing source track with new frame and coordinates
                        source_tracks[source_id]['frames'].append(frame_number)
                        source_tracks[source_id]['azimuth_history'][frame_number] = azimuth
                        source_tracks[source_id]['elevation_history'][frame_number] = elevation
                        source_tracks[source_id]['distance_history'][frame_number] = distance
                
                    # Add the event to the frame metadata in video
                    metadata_by_frame[frame_number].append({
                        'source_id': source_id,
                        'class': str(class_index),
                        'azimuth': azimuth,
                        'elevation': elevation,
                        'distance': distance
                    })

        events_history = self.get_events_history(source_tracks)
        return events_history, metadata_by_frame


    def get_events_history(self, source_tracks):
        """Process source tracks to determine start_frame, end_frame, and duration"""
        events_history = []
        for source_id, track_data in source_tracks.items():
            sorted_frames = sorted(track_data['frames'])
            start_frame = (sorted_frames[0]) 
            end_frame = (sorted_frames[-1] + 1)  # Add 1 since events end after the last frame
            events_history.append({
                'path': track_data['path'],
                'is_image': track_data.get('is_image', False),
                'class': track_data['class'],
                'trackidx': source_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration': end_frame - start_frame,
                'azimuth_history': track_data['azimuth_history'],
                'elevation_history': track_data['elevation_history'],
                'distance_history': track_data['distance_history']
            })

        return events_history


    def generate_visual_event_from_metadata(self, metadata_path, mix_name):
        """Generate audiovisual content using pre-defined metadata"""
        self.events_history, self.metadata_by_frame = self.load_predefined_metadata(metadata_path)
        self.generate_video_mix_360(os.path.join("output/video/", mix_name))


    def generate_video_mix_360(self, mix_name):
        """Create VideoWriter for the output video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(f'{mix_name}.mp4', fourcc, self.video_fps, (self.frame_width, self.frame_height))
        
        # Find the maximum frame from metadata
        max_frame = max(max(self.metadata_by_frame.keys()) if self.metadata_by_frame else 0, self.stream_total_frames if self.stream_total_frames else 0)
        with tqdm(total=max_frame) as pbar:
            for iframe in range(max_frame):
                frame_360 = self.get_frame_at_frame_number(iframe)  # get background to video to overlay events on
                
                # Get active events for this frame
                active_events = self.metadata_by_frame.get(iframe, [])
                
                # Overlay all active videos onto frame_360
                for event_data in active_events:
                    source_id = event_data['source_id']
                    azimuth, elevation = event_data['azimuth'], event_data['elevation']
                    
                    # Convert to projection coordinates (keeping the projection logic from original code)
                    azimuth_proj = azimuth + 180
                    elevation_proj = (-1) * elevation + 90
                    
                    # Find the corresponding event in events_history
                    for event in self.events_history:
                        if event['trackidx'] == source_id:
                            # Get the overlay frame (either from image or video)
                            overlay_frame = None
                            if event['is_image']:
                                overlay_frame = self.get_image_frame(event['path'])
                            else:
                                overlay_video = cv2.VideoCapture(event['path'])
                                # Calculate relative frame in the overlay video
                                relative_frame = iframe - event['start_frame']
                                if relative_frame < 0:
                                    continue
                                    
                                # Get overlay frame
                                overlay_frame = self.get_overlay_frame(overlay_video, relative_frame % int(overlay_video.get(cv2.CAP_PROP_FRAME_COUNT)))
                                overlay_video.release()
                            
                            if overlay_frame is not None:
                                overlay_frame = self.resize_overlay_frame(overlay_frame, 200, 200)
                                frame_360 = self.overlay_frame_on_360(frame_360, overlay_frame, azimuth_proj, elevation_proj)
                            break
                
                pbar.update(1)  # update progress bar
                out_video.write(frame_360)

        # Release video captures and writer
        self.cap_360.release()
        out_video.release()
        cv2.destroyAllWindows()


    def get_frame_at_frame_number(self, frame_number):
        frame_count = int(self.cap_360.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            # If we can't get frame count, create a blank frame
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
        # Ensure we don't exceed the video frame count
        frame_to_get = frame_number % frame_count
        
        self.cap_360.set(cv2.CAP_PROP_POS_FRAMES, frame_to_get)
        ret, frame = self.cap_360.read()
        
        if not ret:
            # If frame reading fails, create a blank frame
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        if self.dark_background:
            # Use a dark background if requested
            frame = np.zeros_like(frame, dtype=np.uint8)
        if self.use_blur:
            # Apply a strong Gaussian blur to create a background similar to video calls
            # The kernel size can be adjusted based on the desired blur amount
            # A larger kernel creates a more intense blur
            frame = cv2.GaussianBlur(frame, (self.video_blur_kernel, self.video_blur_kernel), 0) 

        return frame


    def get_overlay_frame(self, overlay_video, frame_number):
        total_frames = int(overlay_video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None
            
        frame_number = frame_number % total_frames
        overlay_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overlay, overlay_frame = overlay_video.read()
        if not ret_overlay:
            return None
        return overlay_frame


    def get_image_frame(self, image_path):
        """Load and return an image frame, using cache for efficiency"""
        if image_path in self.image_cache:
            return self.image_cache[image_path].copy()
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image: {image_path}")
            return None
        
        # Cache the image for future use
        self.image_cache[image_path] = image.copy()
        return image


    def overlay_frame_on_360(self, frame_360, overlay_frame, azimuth, elevation):
        overlay_height, overlay_width, _ = overlay_frame.shape

        x = int((azimuth / 360.0) * self.frame_width)
        y = int((elevation / 180.0) * self.frame_height)

        x = max(0, min(x, self.frame_width - overlay_width))
        y = max(0, min(y, self.frame_height - overlay_height))

        frame_360[y:y + overlay_height, x:x + overlay_width] = overlay_frame

        return frame_360


    def resize_overlay_frame(self, overlay_frame, width, height):
        return cv2.resize(overlay_frame, (width, height))


def create_video_paths_dictionary(root_directory):
    """
    Creates a dictionary where keys are class numbers (as strings) and values are lists of
    paths to .mp4 files in the corresponding class subdirectory.
    
    Args:
        root_directory (str): Path to the root directory containing class subdirectories
        
    Returns:
        dict: Dictionary mapping class numbers to lists of .mp4 file paths
    """
    video_paths_by_class = {}
    # Get all subdirectories in the root directory
    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    # Process each subdirectory
    for subdir in subdirectories:
        # Extract class number using regex
        class_num = subdir.split("_")[1]
        # Initialize an empty list for this class if it doesn't exist
        if class_num not in video_paths_by_class:
            video_paths_by_class[class_num] = []
        # Get all .mp4 files in this subdirectory
        subdir_path = os.path.join(root_directory, subdir)
        mp4_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) 
                     if f.endswith('.mp4') and os.path.isfile(os.path.join(subdir_path, f))]
        # Add all .mp4 files to the list for this class
        video_paths_by_class[class_num].extend(mp4_files)
    return video_paths_by_class


def create_image_paths_dictionary(root_directory):
    """
    Creates a dictionary where keys are class numbers (as strings) and values are lists of
    paths to image files (jpg, jpeg, png) in the corresponding class subdirectory.
    
    Args:
        root_directory (str): Path to the root directory containing class subdirectories
        
    Returns:
        dict: Dictionary mapping class numbers to lists of image file paths
    """
    image_paths_by_class = {}
    # Get all subdirectories in the root directory
    if not os.path.exists(root_directory):
        print(f"Warning: Directory does not exist: {root_directory}")
        return image_paths_by_class
        
    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    # Process each subdirectory
    for subdir in subdirectories:
        # Extract class number using regex
        class_num = subdir.split("_")[1]
        # Initialize an empty list for this class if it doesn't exist
        if class_num not in image_paths_by_class:
            image_paths_by_class[class_num] = []
        # Get all image files in this subdirectory
        subdir_path = os.path.join(root_directory, subdir)
        image_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(subdir_path, f))]
        # Add all image files to the list for this class
        image_paths_by_class[class_num].extend(image_files)
    return image_paths_by_class


def double_video_length(video_path):
    clip = VideoFileClip(video_path)
    if clip.duration < 2:
        doubled_clip = concatenate_videoclips([clip, clip])
        out_path = video_path.split(".")[0] + "_doubled.mp4"
        doubled_clip.write_videofile(out_path, codec="libx264", fps=24)
        os.rename(out_path, video_path)


def extend_clip(input_video_path):
    clip = VideoFileClip(input_video_path)	
    duration = 30
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 24:
        duration = 38
    if clip.duration >= duration:
        return
	
    repeat_times = max(1, -(-duration // clip.duration))  # Ceiling division

    clips = [clip] * int(repeat_times)
    final_clip = concatenate_videoclips(clips)

    out_path = input_video_path.split(".")[0] + "_doubled." + input_video_path.split(".")[-1]
    final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
    os.rename(out_path, input_video_path)

    clip.close()
    final_clip.close()
    print(f"Video extended and saved to {input_video_path}")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_csv_file(csv_path, config):
    """Process a single CSV metadata file to generate video"""
    # Get base filename without extension for output
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(config['output']['video_dir'], base_name)
    
    # Choose a random 360 video as canvas
    input_360_videos = [os.path.join(config['input']['video_360_path'], f) 
                      for f in os.listdir(config['input']['video_360_path']) 
                      if os.path.isfile(os.path.join(config['input']['video_360_path'], f))]
    input_360_video_path = random.choice(input_360_videos)
    
    # Create video paths dictionaries
    overlay_video_paths_by_class = create_video_paths_dictionary(config['input']['video_assets_dir'])
    overlay_image_paths_by_class = create_image_paths_dictionary(config['input']['image_assets_dir'])
    
    # Create video synthesizer instance with config parameters
    video_synth = VisualSynthesizer(
        input_360_video_path, 
        overlay_video_paths_by_class, 
        overlay_image_paths_by_class, 
        total_duration=config['processing']['video_duration'],
        fps=config['processing']['fps'],
        dark_background=config['processing']['dark_background'],
        use_blur=config['processing']['use_blur'],
    )
    
    # Generate video
    video_synth.generate_visual_event_from_metadata(csv_path, output_path)
    
    return f"Processed {base_name}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visual Synthesis from CSV metadata')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    os.makedirs(config['output']['video_dir'], exist_ok=True)
    
    # Get all CSV files in the metadata directory
    csv_files = glob.glob(os.path.join(config['input']['metadata_dir'], '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {config['input']['metadata_dir']}")
        return
    
    # Process files in parallel
    num_workers = min(config['processing']['workers'], len(csv_files))
    print(f"Processing {len(csv_files)} CSV files with {num_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all processing tasks
        future_to_csv = {executor.submit(process_csv_file, csv_file, config): csv_file for csv_file in csv_files}
        
        # Process results as they complete
        for future in as_completed(future_to_csv):
            csv_file = future_to_csv[future]
            try:
                result = future.result()
                results.append(result)
                print(result)
            except Exception as e:
                print(f"Error processing {os.path.basename(csv_file)}: {e}")
    
    print(f"Completed processing {len(results)} of {len(csv_files)} files.")

if __name__ == "__main__":
    main()
