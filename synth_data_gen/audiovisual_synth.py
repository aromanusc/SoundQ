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

from utils import *
from audio_spatializer import *
from metadata_generator import MetadataSynth

class AudioVisualSynthesizer:
	def __init__(self, input_360_video_path, rirs, source_coords, overlay_video_paths, min_duration, max_duration, total_duration=None):
		self.input_360_video_path = input_360_video_path
		self.rirs = rirs # room_impulse responses used for audio spatialization
		self.channel_num = self.rirs[0].shape[1] # channel count in mic array
		self.source_coords = source_coords
		self.overlay_video_paths = overlay_video_paths
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.total_duration = total_duration
		self.channel_num = self.rirs[0].shape[1] # channel count in array

		self.video_fps = 30		# 33.333 ms
		self.audio_fps = 10 	# 100ms
		self.audio_FS = 24000	# sampling rate (24kHz)
		self.win_size = 512		# window size for spatial convolutions

		# Open the 360-degree video
		self.cap_360 = cv2.VideoCapture(input_360_video_path)
		self.frame_width = int(self.cap_360.get(3))
		self.frame_height = int(self.cap_360.get(4))
		
		if self.total_duration is not None:
			self.stream_total_frames = int(self.total_duration * self.video_fps)  # Calculate total frames based on total_duration
		else:
			self.stream_total_frames = int(self.cap_360.get(cv2.CAP_PROP_FRAME_COUNT))  # Use original video's length

		# Load overlay videos and store them
		self.overlay_videos = []
		self.overlay_info = []
		for overlay_path in overlay_video_paths:
			self.overlay_info.append({
				'path': overlay_path,
			})

	def generate_track_metadata(self, metadata_name):
		metadata_synth = MetadataSynth(metadata_name, self.source_coords, self.overlay_video_paths,
										self.min_duration, self.max_duration, 
										stream_format='audiovisual', total_duration=total_duration)
		self.events_history = metadata_synth.gen_metadata() 

	def generate_audiovisual_event(self, mix_name):
		self.generate_track_metadata(os.path.join("output/metadata/", mix_name))
		self.generate_video_mix_360(os.path.join("output/video/", mix_name))
		self.generate_audio_mix_spatialized(os.path.join("output/audio/", mix_name))

	def generate_video_mix_360(self, mix_name):
		# Create VideoWriter for the output video
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out_video = cv2.VideoWriter(f'{mix_name}.mp4', fourcc, self.video_fps, (self.frame_width, self.frame_height))
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
					if overlay_frame is None:
						print(overlay_data['path'], iframe)
						print("none for some reason")
						continue
					overlay_frame = self.resize_overlay_frame(overlay_frame, 200, 200)
					frame_360 = self.overlay_frame_on_360(frame_360, overlay_frame, azimuth_proj, elevation_proj)

				pbar.update(1) # update progress bar

				out_video.write(frame_360)

		# Release video captures and writer
		self.cap_360.release()
		out_video.release()
		cv2.destroyAllWindows()


	def generate_audio_mix_spatialized(self, track_name):
		audio_mix = np.zeros((self.channel_num, self.audio_FS*self.total_duration), dtype=np.float64)
		for event_data in self.events_history:
			# Load the video file
			video_clip = VideoFileClip(event_data['path'])
			print(event_data['path'])
			# Extract the audio
			audio_clip = video_clip.audio
			audio_sr = audio_clip.fps
			audio_sig = librosa.resample(audio_clip.to_soundarray().T, orig_sr=audio_sr, target_sr=self.audio_FS)
			start_idx = int(self.audio_FS * event_data['start_frame']/self.video_fps)
			dur_samps = int(self.audio_FS * event_data['duration']/self.video_fps)
			audio_sig = self.spatialize_audio_event(audio_sig.mean(axis=0), event_data['rir_id'], dur_samps)
			audio_mix[:, start_idx:start_idx+audio_sig.shape[0]] += audio_sig.T 
		audio_mix /= audio_mix.max()
		sf.write(f'{track_name}.wav', audio_mix.T, self.audio_FS)


	def spatialize_audio_event(self, eventsig, rir_idxs, dur_samps):
		trim_samps = 256*21 # trim padding applied during the convolution process (constant independent of win_size or dur)
		trim_dur = trim_samps/self.audio_FS # get duration in seconds for the padding section
		dur_sec = dur_samps/self.audio_FS
		print(dur_samps, trim_samps, type(eventsig))
		eventsig = eventsig[:dur_samps+trim_samps]
		dur_sec += trim_dur 

		# prepare impulse responses
		rirs = [self.rirs[rir_idxs]]
		rirs.append(rirs[-1]) # append a copy to enable the correct duration
		rirs = np.transpose(np.array(rirs), (1, 2, 0))
		NIRS = rirs.shape[2]

		# setup RIRs spatialization scheme
		ir_times = np.linspace(0, dur_sec, NIRS) # linear

		# spatialize sound event
		output_signal = ctf_ltv_direct(eventsig, rirs, ir_times, self.audio_FS, self.win_size) 
		output_signal /= output_signal.max() # apply max normalization (prevents clipping/distortion)
		output_signal = output_signal[trim_samps:,:] # trim front padding segment 
		return output_signal


	def get_frame_at_frame_number(self, frame_number, dark_background=True):
		self.cap_360.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
		_, frame = self.cap_360.read()
		if dark_background:
			frame = np.zeros_like(frame, dtype=np.uint8)
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

# A collection of 360 videos to use as a canvas. Default condition is to make them all black to have a black canvas.
input_360_video_path = "/scratch/data/audio-visual-seld-dcase2023/data_dcase2023_task3/video_dev/dev-train-tau-aug-acs"
input_360_videos = [os.path.join(input_360_video_path, f) for f in os.listdir(input_360_video_path) if os.path.isfile(os.path.join(input_360_video_path, f))]

# A directory containing all video assets by event class (shared a sample in drive, see readme)
rirs, source_coords = get_audio_spatial_data(aud_fmt="em32", room="METU")
video_assets_dir =  "/scratch/ssd1/audiovisual_datasets/class_events"
overlay_video_paths = []
for root, dirs, files in os.walk(video_assets_dir):
    for file in files:
        overlay_video_paths.append(os.path.join(root, file))

# Initialize directoies:

os.makedirs("./output", exist_ok=True)
os.makedirs("./output/audio", exist_ok=True)  
os.makedirs("./output/video", exist_ok=True)  
os.makedirs("./output/metadata", exist_ok=True)  

min_duration = 2  # Minimum duration for overlay videos (in seconds)
max_duration = 4  # Maximum duration for overlay videos (in seconds)
total_duration = 30

for i in range(98, 120):
	overlay_video_paths = [item for item in overlay_video_paths if ".DS_Store" not in item]
	random.shuffle(overlay_video_paths)
	track_name = f'fold6_roomMETU_mix{i:03d}'.format(i)  # File to save overlay info
	input_360_video_path = random.choice(input_360_videos)
	video_overlay = AudioVisualSynthesizer(input_360_video_path, rirs, source_coords, overlay_video_paths,
							min_duration, max_duration, total_duration)
	video_overlay.generate_audiovisual_event(track_name)
