import cv2
import csv
import copy
import numpy as np
import random

class MetadataSynth:
	def __init__(self, metadata_name, event_coords, event_paths, min_duration, max_duration, stream_format='audio', total_duration=None):
		self.metadata_name = metadata_name
		self.timeline_name = f'{metadata_name}_timeline'
		self.event_coords = event_coords
		self.event_paths = event_paths
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.stream_format = stream_format
		self.total_duration = total_duration

		self.video_fps = 30 # 300ms
		self.audio_fps = 10 # 100ms

		# Use assert to check if stream_format is valid
		valid_formats = ["audio", "audiovisual"]
		assert isinstance(self.stream_format, str) and self.stream_format in valid_formats, "Invalid stream_format"

		self.stream_total_frames = None
		if self.stream_format == "audio":
			self.stream_total_frames = self.audio_fps*self.total_duration
			self.max_frame_dur = self.audio_fps*self.max_duration
		else: # "audiovisual"
			self.stream_total_frames = self.video_fps*self.total_duration
			self.max_frame_dur = self.video_fps*self.max_duration

		# Load and store event information
		self.event_info = []
		for event_path in event_paths:
			path_parts = event_path.split("/")
			class_number = None
			for part in path_parts:
				if part.startswith("Class_") and part.count('_') >= 2:
					class_number = part.split('_')[1]

			self.event_info.append({
				'path': event_path,
				'class': class_number
			})

		self.metadata_file_csv = open(f'{self.timeline_name}.csv', 'w', newline='')
		self.metadata_writer_csv = csv.writer(self.metadata_file_csv)
		self.metadata_file = open(f'{self.metadata_name}.csv', 'w', newline='')
		self.metadata_writer = csv.writer(self.metadata_file)


	def get_event_duration(self, stream_format):
		if stream_format == "audio":
			return int(self.audio_fps*random.uniform(self.min_duration, self.max_duration))
		else: # "audiovisual"
			return int(self.video_fps*random.uniform(self.min_duration, self.max_duration))

	def gen_metadata(self, max_polyphony=3, silence_weight=36*2):
		active_events = [] # tracks active events per frame
		events_history = [] # keep track of the events history
		available_tacks = self.event_info
		available_coords = self.event_coords
		for frame_number in range(self.stream_total_frames-self.max_frame_dur):
			n_active = len(active_events) # get number of currently active events
			used_coords_in_frame = set()
			for i in range(n_active, max_polyphony):
				if np.random.rand() < (max_polyphony-n_active)/(silence_weight+max_polyphony):
					temp = list(enumerate(available_tacks))
					if temp == []:
						break
					idxe, event_info = random.choice(temp)
					available_tacks.pop(idxe)
					# randomly choose duration for event and convert to FPS (*30)
					choose_duration = self.get_event_duration(self.stream_format)
					# choose a location at random from a trayectory
					# this will have to be updated later to account for moving sources.
					# we will get a list of discrete location points and update at each 100ms frame
					max_attempts = 0
					while True:
						idxc, (rir_id, azi, ele) = random.choice(list(enumerate(self.event_coords)))
						if (azi, ele) not in used_coords_in_frame:
							used_coords_in_frame.add((azi, ele))
							break
						max_attempts += 1
						if max_attempts > len(self.event_coords):
							break

					# idxc, (rir_id, azi, ele) = random.choice(list(enumerate(self.event_coords)))
					# available_coords.pop(idxc) # ensure no overlapping sources at a frame (mainly for video overlays)
					# populate event metadata
					new_event = {
						'path': event_info["path"],
						'class': event_info["class"],
						'trackidx': event_info["path"].split("/")[-1][:3], # TODO: here use instead another ideantifier??
						'start_frame': frame_number,
						'end_frame': frame_number+choose_duration,
						'duration': choose_duration,
						'rir_id': rir_id,
						'azim': azi,
						'elev': ele,
						'duration': choose_duration,
						'curr_frame': 0
					}
					active_events.append(new_event)
					events_history.append(copy.deepcopy(new_event)) # make a deepcopy to avoid curr_frame changes
			# Remove events that have finished their duration
			active_events = [event_data for event_data in active_events if event_data['curr_frame'] < event_data["duration"]]
			# increase frame count of active events
			for event in active_events:
				event['curr_frame'] += 1

		self.write_events_list_to_csv(events_history)
		self.write_metadata_DCASE_format(events_history)

		return events_history

	def write_events_list_to_csv(self, event_list):
		for event in event_list:
			# 'Path', 'TrackID', 'Start_Frame', 'Duration', 'Azimuth', 'Elevation'])
			self.metadata_writer_csv.writerow([event["path"],
									event["trackidx"],
									event["start_frame"],
									event["duration"],
									event["azim"],
									event["elev"]])
		self.metadata_file_csv.close()

	def write_metadata_DCASE_format(self, event_list):
		# 'frame', 'class' 'trackID', 'azimuth', 'elevation', 'distance'
		for iframe in range(self.stream_total_frames):
			frame_step = 1 if self.stream_format == "audio" else 3
			active_events = [event_data for event_data in event_list if (iframe*frame_step >= event_data['start_frame'] and iframe*frame_step < event_data["end_frame"])]
			if len(active_events) > 0:
				for event in active_events:
					self.metadata_writer.writerow([iframe,event["class"],event["trackidx"],event["azim"],event["elev"], 0])
		self.metadata_file.close()