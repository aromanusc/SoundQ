# from audiovisual_synth import AudioVisualSynthesizer
import os
import cv2
import csv
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import scipy
import scipy.signal as signal
import random
from moviepy.editor import VideoFileClip
import random

from utils import *
from audio_spatializer import *
from metadata_generator import MetadataSynth

from moviepy.editor import VideoFileClip, concatenate_videoclips

def double_video_length(video_path):
    clip = VideoFileClip(video_path)
    if clip.duration < 2:
        doubled_clip = concatenate_videoclips([clip, clip])
        out_path = video_path.split(".mp4")[0] + "_doubled.mp4"
        doubled_clip.write_videofile(out_path, codec="libx264", fps=24)
        os.rename(out_path, video_path)

# Usage
# video_path = "/Users/rithikpothuganti/cs677/new-project/SoundQ/data/Dataset/Class_7_Door_Open_Close/clip11.mp4"
# # result = double_video_length(video_path)
# clip = VideoFileClip(video_path).audio
# print(clip.to_soundarray(fps=11025))


# Example usage:
input_360_video_path = "/Users/rithikpothuganti/cs677/new-project/SoundQ/video_dev/video_dev/dev-test-sony/fold4_room23_mix001.mp4"

rirs, source_coords = get_audio_spatial_data(aud_fmt="em32", room="METU")
directory =  "/Users/rithikpothuganti/cs677/new-project/SoundQ/data/Dataset/"
overlay_video_paths = []
for root, dirs, files in os.walk(directory):
    for file in files:
        overlay_video_paths.append(os.path.join(root, file))
print("test")

overlay_video_paths = [item for item in overlay_video_paths if item != "/Users/rithikpothuganti/cs677/new-project/SoundQ/data/Dataset/.DS_Store"]

for video in overlay_video_paths:
    double_video_length(video)

random.shuffle(overlay_video_paths)

# for video in overlay_video_paths:
#     print(video)

overlay_video_path_groups = [overlay_video_paths[i:i + 3] for i in range(0, len(overlay_video_paths), 3)]

for group in overlay_video_path_groups:
    print(group)


min_duration = 2  # Minimum duration for overlay videos (in seconds)
max_duration = 3  # Maximum duration for overlay videos (in seconds)

# print(overlay_video_paths)

# min_duration = 2  # Minimum duration for overlay videos (in seconds)
# max_duration = 3  # Maximum duration for overlay videos (in seconds)
# total_duration = 8
# track_name = "fold1_room001_mix"  # File to save overlay info
# video_overlay = AudioVisualSynthesizer(input_360_video_path, rirs, source_coords, overlay_video_paths,
# 							 min_duration, max_duration, total_duration)

# print("Synthesizing spatial audiovisual event")
# video_overlay.generate_audiovisual_event(track_name)

# synthesizer = AudioVisualSynthesizer(input_360_video_path, rirs, source_coords, overlay_video_paths, min_duration, max_duration, total_duration)