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
from audio_spatializer import *
from metadata_generator import MetadataSynth

class AudioSynthesizer:
	def __init__(self, rirs, source_coords, audio_tracks_paths, min_duration, max_duration, total_duration):
		self.rirs = rirs # room_impulse responses
		self.source_coords = source_coords
		self.audio_tracks_paths = audio_tracks_paths
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.total_duration = total_duration
		self.channel_num = self.rirs[0].shape[1] # channel count in array
		self.audio_fps = 10 	# 100ms
		self.audio_FS = 24000	# sampling rate (24kHz)
		self.win_size = 512		# window size for spatial convolutions
		self.stream_total_frames = int(self.total_duration * self.audio_fps)  # Calculate total frames based on total_duration

	def generate_track_metadata(self, metadata_name):
		metadata_synth = MetadataSynth(metadata_name, self.source_coords, self.audio_tracks_paths,
										self.min_duration, self.max_duration, 
										stream_format='audio', total_duration=total_duration)
		self.events_history = metadata_synth.gen_metadata() 

	def generate_audio_mix_spatialized(self, mix_name):
		self.generate_track_metadata(mix_name)
		audio_mix = np.zeros((self.channel_num, self.audio_FS*self.total_duration), dtype=np.float64)
		for event_data in self.events_history:
			# Load the video file
			audio_sig, sr = librosa.load(event_data['path'], sr=None, mono=None)
			# Extract the audio
			start_idx = int(self.audio_FS * event_data['start_frame']/self.audio_fps)
			duration_samps = int(self.audio_FS * event_data['duration']/self.audio_fps)
			audio_sig = librosa.resample(audio_sig.mean(axis=0), orig_sr=sr, target_sr=self.audio_FS)
			audio_sig = self.spatialize_audio_event(audio_sig, event_data['rir_id'], duration_samps)
			audio_mix[:, start_idx:start_idx+audio_sig.shape[0]] += audio_sig.T #.mean(axis=0) # TODO: [fix] this may cause a 1 frame delay between audio and video streams
		audio_mix /= audio_mix.max()
		sf.write(f'{mix_name}.wav', audio_mix.T, self.audio_FS)


	def spatialize_audio_event(self, eventsig, rir_idxs, dur_samps):
		trim_samps = 256*21 # trim padding applied during the convolution process (constant independent of win_size or dur)
		trim_dur = trim_samps/self.audio_FS # get duration in seconds for the padding section
		dur_sec = dur_samps/self.audio_FS

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


# Example usage:
rirs, source_coords = get_audio_spatial_data(aud_fmt="em32", room="METU")
directory_path = "/scratch/data/SELD-data-generator/dcase_datagen/data/fma_small/000/"
audio_tracks_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]
min_duration = 2  # Minimum duration for overlay videos (in seconds)
max_duration = 3  # Maximum duration for overlay videos (in seconds)
total_duration = 15
track_name = "fold1_room002_mix"  # File to save overlay info
AudioSynth = AudioSynthesizer(rirs, source_coords, audio_tracks_paths,
							 min_duration, max_duration, total_duration)

print("Synthesizing spatial audio")
AudioSynth.generate_audio_mix_spatialized(track_name)



