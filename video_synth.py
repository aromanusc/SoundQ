import cv2
import csv
import random
import numpy as np
from tqdm import tqdm


from metadata_generator import MetadataSynth

class VideoSynthesizer:
	def __init__(self, input_360_video_path, output_video_path, overlay_coords, overlay_video_paths, min_duration, max_duration, metadata_name, total_duration=None):
		self.input_360_video_path = input_360_video_path
		self.output_video_path = output_video_path
		self.overlay_coords = overlay_coords
		self.overlay_video_paths = overlay_video_paths
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.metadata_name = metadata_name

		self.video_fps = 30 # 33.333 ms

		# Open the 360-degree video
		self.cap_360 = cv2.VideoCapture(input_360_video_path)
		print("HEREEEE", input_360_video_path, self.cap_360)
		self.frame_width = int(self.cap_360.get(3))
		self.frame_height = int(self.cap_360.get(4))
		
		if total_duration is not None:
			self.stream_total_frames = int(total_duration * self.video_fps)  # Calculate total frames based on total_duration
		else:
			self.stream_total_frames = int(self.cap_360.get(cv2.CAP_PROP_FRAME_COUNT))  # Use original video's length

		# Create VideoWriter for the output video
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.out = cv2.VideoWriter(output_video_path, fourcc, self.video_fps, (self.frame_width, self.frame_height))

		# Load overlay videos and store them
		self.overlay_videos = []
		self.overlay_info = []
		for overlay_path in overlay_video_paths:
			cap_overlay = cv2.VideoCapture(overlay_path)
			overlay_duration = int(cap_overlay.get(cv2.CAP_PROP_FRAME_COUNT) / self.video_fps)
			self.overlay_videos.append(cap_overlay)
			self.overlay_info.append({
				'path': overlay_path,
				'total_duration': overlay_duration
			})

		metadata_synth = MetadataSynth(self.metadata_name, self.overlay_coords, self.overlay_video_paths,
							self.min_duration, self.max_duration, stream_format='audiovisual', total_duration=total_duration)
		self.events_history = metadata_synth.gen_metadata() 

	def check_active_overlays(frame_number):
		# Here we will check fr active overlays 
		# return: active overlays list
		pass


	def overlay_videos_on_360(self, max_polyphony=3, silence_weight=36):
		with tqdm(total=self.stream_total_frames) as pbar:
			for iframe in range(self.stream_total_frames):
				frame_360 = self.get_frame_at_frame_number(iframe) # get background to video to overlay events on
				active_events = [event_data for event_data in self.events_history if (iframe >= event_data['start_frame'] and iframe < event_data["end_frame"])]		
				# Overlay all active videos onto frame_360
				for overlay_data in active_events:
					azimuth, elevation = overlay_data['azim'], overlay_data['elev']
					azimuth_mapped = azimuth + 180
					elevation_mapped = (-1) * elevation + 90

					overlay_video = cv2.VideoCapture(overlay_data['path'])
					overlay_frame = self.get_overlay_frame(overlay_video, iframe)
					overlay_frame = self.resize_overlay_frame(overlay_frame, 200, 200)
					frame_360 = self.overlay_frame_on_360(frame_360, overlay_frame, azimuth_mapped, elevation_mapped)

				pbar.update(1) # update progress bar
				self.out.write(frame_360)

		# Release video captures and writer
		self.cap_360.release()
		for overlay_video in self.overlay_videos:
			overlay_video.release()
		self.out.release()
		cv2.destroyAllWindows()


	def get_frame_at_frame_number(self, frame_number):
		self.cap_360.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
		_, frame = self.cap_360.read()
		return frame

	def get_overlay_frame(self, overlay_video, frame_number):
		overlay_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
		ret_overlay, overlay_frame = overlay_video.read()
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


# Example usage:
input_360_video_path = "/Users/adrianromanguzman/Downloads/video_dev/dev-train-sony/fold3_room22_mix011.mp4"
output_video_path = "output_video_overlayed.mp4"
overlay_coords = [(-90, 0), (90, 0), (30, 45), (120, 0), (120, 30), (-120, 40), (0, -45), (0, -35), (0, -25), (0, -15), (0, +35), (0, 25)]  # Example coordinates in azimuth and elevation
overlay_video_paths = ["/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/536.mp4",
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/320.mp4", 
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/321.mp4", 
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/328.mp4", 
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/525.mp4", 
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/264.mp4", 
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/152.mp4",
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/151.mp4",
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/153.mp4",
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/150.mp4",
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/148.mp4",
						"/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/149.mp4"
						]  # Paths to overlay videos
min_duration = 3  # Minimum duration for overlay videos (in seconds)
max_duration = 5  # Maximum duration for overlay videos (in seconds)
total_duration = 12
metadata_name = "event_metadata"  # File to save overlay info
video_overlay = VideoSynthesizer(input_360_video_path, output_video_path, overlay_coords, overlay_video_paths,
							 min_duration, max_duration, metadata_name, total_duration)
video_overlay.overlay_videos_on_360()