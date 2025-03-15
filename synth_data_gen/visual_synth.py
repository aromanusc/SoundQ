import os
import cv2
import csv
import random
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import scipy
import scipy.signal as signal
from moviepy.editor import VideoFileClip, concatenate_videoclips
from collections import defaultdict

from utils import *
from audio_spatializer import *

class AudioVisualSynthesizer:
    def __init__(self, input_360_video_path, rirs, overlay_video_paths_by_class, total_duration=None):
        self.input_360_video_path = input_360_video_path
        self.rirs = rirs  # room_impulse responses used for audio spatialization
        self.channel_num = self.rirs[0].shape[1]  # channel count in mic array
        self.overlay_video_paths_by_class = overlay_video_paths_by_class  # Dictionary mapping class indices to lists of video paths
        self.total_duration = total_duration

        self.video_fps = 30      # 30 fps
        self.audio_fps = 10      # 10 fps
        self.audio_FS = 24000    # sampling rate (24kHz)
        self.win_size = 512      # window size for spatial convolutions

        # Open the 360-degree video
        self.cap_360 = cv2.VideoCapture(input_360_video_path)
        self.frame_width = int(self.cap_360.get(3))
        self.frame_height = int(self.cap_360.get(4))
        
        if self.total_duration is not None:
            self.stream_total_frames = int(self.total_duration * self.video_fps)  # Calculate total frames based on total_duration
        else:
            self.stream_total_frames = int(self.cap_360.get(cv2.CAP_PROP_FRAME_COUNT))  # Use original video's length

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
                if len(row) < 6:  # Ensure we have all needed fields
                    continue
                    
                frame_number = int(row[0])
                class_index = int(row[1])
                source_index = int(row[2])
                azimuth = int(row[3])
                elevation = int(row[4])
                distance = int(row[5])
                
                # Check if the source exists in our tracking dictionary
                source_id = f"{class_index}_{source_index}"
                if source_id not in source_tracks:
                    # Assign a random video path from the class
                    if str(class_index) in self.overlay_video_paths_by_class:
                        video_paths = self.overlay_video_paths_by_class[str(class_index)]
                        if video_paths:
                            source_tracks[source_id] = {
                                'path': random.choice(video_paths),
                                'start_frame': frame_number,
                                'frames': [frame_number],
                                'azimuth_history': {frame_number: azimuth},
                                'elevation_history': {frame_number: elevation},
                                'distance_history': {frame_number: distance},
                                'class': str(class_index)
                            }
                        else:
                            print(f"Warning: No video paths available for class {class_index}")
                            continue
                    else:
                        print(f"Warning: Class {class_index} not found in video paths")
                        continue
                else:
                    # Update the existing source track with new frame and coordinates
                    source_tracks[source_id]['frames'].append(frame_number)
                    source_tracks[source_id]['azimuth_history'][frame_number] = azimuth
                    source_tracks[source_id]['elevation_history'][frame_number] = elevation
                    source_tracks[source_id]['distance_history'][frame_number] = distance
                    
                # Add the event to the frame metadata
                metadata_by_frame[frame_number].append({
                    'source_id': source_id,
                    'class': str(class_index),
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'distance': distance
                })
        
        # Process source tracks to determine start_frame, end_frame, and duration
        events_history = []
        for source_id, track_data in source_tracks.items():
            sorted_frames = sorted(track_data['frames'])
            start_frame = sorted_frames[0]
            end_frame = sorted_frames[-1] + 1  # Add 1 since events end after the last frame
            
            events_history.append({
                'path': track_data['path'],
                'class': track_data['class'],
                'trackidx': source_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration': end_frame - start_frame,
                'azimuth_history': track_data['azimuth_history'],
                'elevation_history': track_data['elevation_history'],
                'distance_history': track_data['distance_history']
            })
        
        return events_history, metadata_by_frame

    def generate_audiovisual_event_from_metadata(self, metadata_path, mix_name):
        """Generate audiovisual content using pre-defined metadata"""
        self.events_history, self.metadata_by_frame = self.load_predefined_metadata(metadata_path)
        self.generate_video_mix_360(os.path.join("output/video/", mix_name))

    def generate_video_mix_360(self, mix_name):
        # Create VideoWriter for the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(f'{mix_name}.mp4', fourcc, self.video_fps, (self.frame_width, self.frame_height))
        
        # Find the maximum frame from metadata
        max_frame = max(max(self.metadata_by_frame.keys()) if self.metadata_by_frame else 0, self.stream_total_frames)
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

    def get_frame_at_frame_number(self, frame_number, dark_background=True):
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
        elif dark_background:
            # Use a dark background if requested
            frame = np.zeros_like(frame, dtype=np.uint8)
            
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
