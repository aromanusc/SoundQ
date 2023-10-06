import cv2
import csv
import numpy as np
import random

class VideoOverlay:
	def __init__(self, input_360_video_path, output_video_path, overlay_coords, overlay_video_paths, min_duration, max_duration, info_file_path, total_duration=None):
		self.input_360_video_path = input_360_video_path
		self.output_video_path = output_video_path
		self.overlay_coords = overlay_coords
		self.overlay_video_paths = overlay_video_paths
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.info_file_path = info_file_path

		# Open the 360-degree video
		self.cap_360 = cv2.VideoCapture(input_360_video_path)
		self.frame_width = int(self.cap_360.get(3))
		self.frame_height = int(self.cap_360.get(4))
		
		if total_duration is not None:
			self.total_frames = int(total_duration * 30)  # Calculate total frames based on total_duration
		else:
			self.total_frames = int(self.cap_360.get(cv2.CAP_PROP_FRAME_COUNT))  # Use original video's length

		# Create VideoWriter for the output video
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.out = cv2.VideoWriter(output_video_path, fourcc, 30, (self.frame_width, self.frame_height))

		# Load overlay videos and store them
		self.overlay_videos = []
		self.overlay_info = []
		for overlay_path in overlay_video_paths:
			cap_overlay = cv2.VideoCapture(overlay_path)
			overlay_duration = int(cap_overlay.get(cv2.CAP_PROP_FRAME_COUNT) / 30)
			self.overlay_videos.append(cap_overlay)
			self.overlay_info.append({
				'path': overlay_path,
				'duration': overlay_duration
			})

		self.overlay_frame_positions = self.calculate_overlay_frame_positions()

		self.csv_file = open(self.info_file_path, 'w', newline='')
		self.csv_writer = csv.writer(self.csv_file)
		self.csv_writer.writerow(['Path', 'Start_Frame', 'End_Frame', 'Duration', 'Azimuth', 'Elevation'])
		with open(self.info_file_path,'w') as info_file:
			for i, overlay in enumerate(self.overlay_info):
				self.csv_writer.writerow([overlay["path"],self.overlay_frame_positions[i][0],self.overlay_frame_positions[i][1],overlay["duration"],self.overlay_coords[i][0],self.overlay_coords[i][1]])
		self.csv_file.close()



	def calculate_overlay_frame_positions(self):
		positions = []
		for i in range(len(self.overlay_coords)):
			start_time = random.uniform(0, self.total_frames / 30 - self.max_duration)
			end_time = start_time + random.uniform(self.min_duration, self.max_duration)
			
			start_frame = int(start_time * 30)
			end_frame = int(end_time * 30)

			positions.append((start_frame, end_frame))
		return positions

	def overlay_videos_on_360(self):
		for frame_number in range(self.total_frames):
			frame_360 = self.get_frame_at_frame_number(frame_number)
			active_overlays = []
			# Check if any overlays should be activated
			for i, (start_frame, end_frame) in enumerate(self.overlay_frame_positions):
				if start_frame <= frame_number and frame_number < end_frame:
					overlay_video = self.overlay_videos[i]
					active_overlays.append({
						'video': overlay_video,
						'info': self.overlay_info[i],
						'azimuth': self.overlay_coords[i][0],
						'elevation': self.overlay_coords[i][1]
					})
			# Overlay all active videos onto frame_360
			for overlay_data in active_overlays:
				overlay_info = overlay_data['info']
				azimuth, elevation = overlay_data['azimuth'], overlay_data['elevation']
				azimuth_mapped = azimuth + 180
				elevation_mapped = (-1) * elevation + 90

				overlay_video = overlay_data['video']
				overlay_frame = self.get_overlay_frame(overlay_video)
				overlay_frame = self.resize_overlay_frame(overlay_frame, 200, 200)
				frame_360 = self.overlay_frame_on_360(frame_360, overlay_frame, azimuth_mapped, elevation_mapped)

			self.out.write(frame_360)
			# Remove overlays that have finished their duration
			active_overlays = [overlay_data for overlay_data in active_overlays if frame_number < overlay_data['info']["duration"]]

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

	def get_overlay_frame(self, overlay_video):
		ret_overlay, overlay_frame = overlay_video.read()
		if not ret_overlay:
			overlay_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
overlay_coords = [(-90, 0), (90, 0), (30, 45), (0, -45)]  # Example coordinates in azimuth and elevation
overlay_video_paths = ["/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/536.mp4", "/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/525.mp4", "/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/264.mp4", "/Users/adrianromanguzman/Downloads/MUSIC_dataset_script/data/155.mp4"]  # Paths to overlay videos
min_duration = 5  # Minimum duration for overlay videos (in seconds)
max_duration = 10  # Maximum duration for overlay videos (in seconds)
total_duration = 30
overlay_info_file = "overlay_info.csv"  # File to save overlay info
video_overlay = VideoOverlay(input_360_video_path, output_video_path, overlay_coords, overlay_video_paths,
							 min_duration, max_duration, overlay_info_file, total_duration)
video_overlay.overlay_videos_on_360()