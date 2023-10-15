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
from moviepy.editor import VideoFileClip

from utils import *
from metadata_generator import MetadataSynth

class VideoSynthesizer:
	def __init__(self, input_360_video_path, output_video_path, rirs, source_coords, overlay_video_paths, min_duration, max_duration, track_name, total_duration=None):
		self.input_360_video_path = input_360_video_path
		self.output_video_path = output_video_path
		self.rirs = rirs # room_impulse responses
		self.source_coords = source_coords
		self.overlay_video_paths = overlay_video_paths
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.track_name = track_name
		self.metadata_name = f'{track_name}_metadata'
		self.total_duration = total_duration
		self.channel_num = self.rirs[0].shape[1] # channel count in array

		self.video_fps = 30		# 33.333 ms
		self.audio_fps = 10 	# 100ms
		self.audio_FS = 24000	# sampling rate (24kHz)

		# Open the 360-degree video
		self.cap_360 = cv2.VideoCapture(input_360_video_path)
		self.frame_width = int(self.cap_360.get(3))
		self.frame_height = int(self.cap_360.get(4))
		
		if self.total_duration is not None:
			self.stream_total_frames = int(self.total_duration * self.video_fps)  # Calculate total frames based on total_duration
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

		metadata_synth = MetadataSynth(self.track_name, self.source_coords, self.overlay_video_paths,
							self.min_duration, self.max_duration, stream_format='audiovisual', total_duration=total_duration)
		self.events_history = metadata_synth.gen_metadata() 


	def generate_video_mix_360(self):
		with tqdm(total=self.stream_total_frames) as pbar:
			for iframe in range(self.stream_total_frames):
				frame_360 = self.get_frame_at_frame_number(iframe) # get background to video to overlay events on
				active_events = [event_data for event_data in self.events_history if (iframe >= event_data['start_frame'] and iframe < event_data["end_frame"])]		
				# Overlay all active videos onto frame_360
				for overlay_data in active_events:
					azimuth, elevation = overlay_data['azim'], overlay_data['elev']
					azimuth_proj = azimuth + 180
					elevation_proj = (-1) * elevation + 90

					overlay_video = cv2.VideoCapture(overlay_data['path'])
					overlay_frame = self.get_overlay_frame(overlay_video, iframe)
					overlay_frame = self.resize_overlay_frame(overlay_frame, 200, 200)
					frame_360 = self.overlay_frame_on_360(frame_360, overlay_frame, azimuth_proj, elevation_proj)

				pbar.update(1) # update progress bar
				self.out.write(frame_360)

		# Release video captures and writer
		self.cap_360.release()
		for overlay_video in self.overlay_videos:
			overlay_video.release()
		self.out.release()
		cv2.destroyAllWindows()


	def generate_audio_mix_spatialized(self):
		audio_mix = np.zeros((self.channel_num, self.audio_FS*self.total_duration), dtype=np.float64)
		for event_data in self.events_history:
			# Load the video file
			video_clip = VideoFileClip(event_data['path'])
			# Extract the audio
			audio_clip = video_clip.subclip(0, event_data['duration']/self.video_fps).audio
			audio_sr = audio_clip.fps
			print("audio_clip.to_soundarray().T", audio_clip.to_soundarray().T.shape, event_data['duration']/self.video_fps)
			audio_sig = librosa.resample(audio_clip.to_soundarray().T, orig_sr=audio_sr, target_sr=self.audio_FS)
			print("resampled audio shape", audio_sig.shape)
			audio_sig = self.spatialize_audio_event(audio_sig.mean(axis=0), event_data['rir_id'])
			start_idx = int(self.audio_FS * event_data['start_frame']/self.video_fps)
			end_idx = int(self.audio_FS * event_data['duration']/self.video_fps)
			print("shape on track to overly", audio_mix[:, start_idx:start_idx+audio_sig.shape[0]].shape, audio_sig.shape[0], end_idx)
			audio_mix[:, start_idx:start_idx+audio_sig.shape[0]] += audio_sig.T #.mean(axis=0) # TODO: [fix] this may cause a 1 frame delay between audio and video streams
		sf.write(f'{self.track_name}.wav', audio_mix.T, self.audio_FS)


	def spatialize_audio_event(self, eventsig, ir_times, rir_idxs):
		trim_samps = 256*21 # trim padding applied during the convolution process (constant independent of win_size or dur)
		dur = eventsig.shape[0] / self.audio_FS 
		irs = self.irs[rir_idxs]
		rirs.append(rirs[-1]) # append a copy to enable the correct duration
		output_signal = ctf_ltv_direct(sig, irs, ir_times, FS, win_size) 
		output_signal /= output_signal.max() # apply max normalization (prevents clipping/distortion)
		output_signal = output_signal[trim_samps:,:] # trim front padding segment 
		return spateventsig

		# nCh = self.rirs[rir_idxs].shape[1] # get number of channels
		# spateventsig_list = []
		# for ichan in range(nCh):
		# 	spatsig_channel = signal.convolve(eventsig, np.squeeze(self.rirs[rir_idxs][:, ichan]), mode='same', method='fft')
		# 	spateventsig_list.append(spatsig_channel)
		# spateventsig = np.stack(spateventsig_list, axis=1)


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
input_360_video_path = "/scratch/data/audio-visual-seld-dcase2023/data_dcase2023_task3/video_dev/dev-train-sony/fold3_room22_mix011.mp4"
output_video_path = "output_video_overlayed.mp4"

def get_audio_spatial_data(aud_fmt="em32", room="METU"):
	if aud_fmt != "em32" and aud_fmt != "mic":
		parser.error("You must provide a valid microphone name: em32, mic")

	metu_db_dir = None
	if room == "METU":
		metu_db_dir = "/scratch/data/RIR_datasets/spargair/em32"
	top_height = 5
	mic_xyz = get_mic_xyz()
	source_coords, rirs = [], []

	rir_id = 0
	# Outter trayectory: bottom to top
	for height in range(0, top_height):
		for num in REF_OUT_TRAJ:
			# Coords computed based on documentation.pdf from METU Sparg
			x = (3 - int(num[0])) * 0.5
			y = (3 - int(num[1])) * 0.5
			z = (2 - (int(num[2])-height)) * 0.3 + 1.5
			source_xyz = [x, y, z] # note -1 since METU is flipped up-side-down

			azim, elev, _ = az_ele_from_source(mic_xyz, source_xyz)
			elev *= -1 # Account for elevation being swapped in METU

			source_coords.append((rir_id, azim, elev))
			rir_name = num[0] + num[1] + str(int(num[2])-height)
			ir_path = os.path.join(metu_db_dir, rir_name, f"IR_{aud_fmt}.wav")
			irdata, sr = librosa.load(ir_path, mono=False, sr=48000)
			irdata_resamp = librosa.resample(irdata, orig_sr=sr, target_sr=24000)
			rirs.append(irdata_resamp.T)
			rir_id += 1

	return rirs, source_coords

# rirs, source_coords = get_audio_spatial_data()
# directory_path = "/scratch/ssd1/audio_datasets/MUSIC_dataset/data/"
# overlay_video_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]
# min_duration = 3  # Minimum duration for overlay videos (in seconds)
# max_duration = 4  # Maximum duration for overlay videos (in seconds)
# total_duration = 20
# track_name = "fold1_room001_mix"  # File to save overlay info
# video_overlay = VideoSynthesizer(input_360_video_path, output_video_path, rirs, source_coords, overlay_video_paths,
# 							 min_duration, max_duration, track_name, total_duration)
# video_overlay.generate_audio_mix_spatialized()
# video_overlay.generate_video_mix_360()




rirs, source_coords = get_audio_spatial_data()
directory_path = "/scratch/ssd1/audio_datasets/MUSIC_dataset/data/"
overlay_video_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]
min_duration = 3  # Minimum duration for overlay videos (in seconds)
max_duration = 4  # Maximum duration for overlay videos (in seconds)
total_duration = 20
track_name = "fold1_room001_mix"  # File to save overlay info
video_overlay = VideoSynthesizer(input_360_video_path, output_video_path, rirs, source_coords, overlay_video_paths,
							 min_duration, max_duration, track_name, total_duration)

sig, sr = librosa.load('/home/asroman/repos/Acoustic_Spatialvisualizer/violin.wav', sr=44100, mono=True)
sig = librosa.resample(sig, orig_sr=sr, target_sr=24000)
sig = sig[:24000 * 1]
print(sig.shape)
new_sig = video_overlay.spatialize_audio_event(sig, 0)
print(new_sig.shape)

sf.write('violin_orig.wav', sig, 24000)
sf.write('violin_metu.wav', new_sig, 24000)


# extra: 47,999




