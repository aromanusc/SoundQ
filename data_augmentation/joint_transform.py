import os

from audio_transformation import *  # Assuming this contains functions to transform audio files
from video_transformation import VideoPixelTransformer
from video_transformation import acs as acs_video


def transform_audio_and_video(video_dir, audio_foa_dir, audio_mic_dir, metadata_dir,
								output_audio_foa_dir, output_audio_mic_dir, 
								output_video_dir, output_metadata_dir):
	# Create output directories if they don't exist
	if not os.path.exists(output_audio_foa_dir):
		os.makedirs(output_audio_foa_dir)
	if not os.path.exists(output_audio_mic_dir):
		os.makedirs(output_audio_mic_dir)
	if not os.path.exists(output_video_dir):
		os.makedirs(output_video_dir)
	if not os.path.exists(output_metadata_dir):
		os.makedirs(output_metadata_dir)
	
	# acs(metadata_dir, output_metadata_dir)		# metadata
	# acs(audio_foa_dir, output_audio_foa_dir)	# microphone
	# acs(audio_mic_dir, output_audio_mic_dir)	# first order ambisonics
	acs_video(video_dir, output_video_dir)		# video

def main():
	transformed_dataset_path = "/scratch/ssd1/audiovisual_datasets/STARSS_chSwap"
	original_dataset_path = "/scratch/data/audio-visual-seld-dcase2023/data_dcase2023_task3/"

	# source files for video, audio and metadata
	video_dir = os.path.join(original_dataset_path, "video_dev")
	audio_foa_dir = os.path.join(original_dataset_path, "foa_dev")
	audio_mic_dir = os.path.join(original_dataset_path, "mic_dev")
	metadata_dir = os.path.join(original_dataset_path, "metadata_dev")
	
	# directories to save transformed audio and video
	output_audio_foa_dir = os.path.join(transformed_dataset_path, "foa_dev")
	output_audio_mic_dir = os.path.join(transformed_dataset_path, "mic_dev")
	output_video_dir = os.path.join(transformed_dataset_path, "video_dev")
	output_metadata_dir = os.path.join(transformed_dataset_path, "metadata_dev")

	transform_audio_and_video(video_dir, audio_foa_dir, audio_mic_dir, metadata_dir, 
							output_audio_foa_dir, output_audio_mic_dir, 
							output_video_dir, output_metadata_dir)
	

if __name__ == "__main__":
	main()
