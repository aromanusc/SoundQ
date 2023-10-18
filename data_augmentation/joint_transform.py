import os

from audio_transformation import *  # Assuming this contains functions to transform audio files
from video_transformation import VideoPixelTransformer
from video_transformation import acs as acs_video

# channels match up like this: https://chat.openai.com/c/a1bb8b95-18a4-4779-bec8-a25c24e9a345


def transform_audio_and_video(video_dir, audio_dir, 
                              metadata_dir, output_audio_dir, 
                              output_video_dir, output_metadata_dir):
    # Create output directories if they don't exist
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)

    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    # acs(metadata_dir, output_metadata_dir)
    # acs(audio_dir, output_audio_dir)
    acs_video(video_dir, output_video_dir)
    

    # Loop over each file in data directory
    # for audio_file in os.listdir(audio_dir):
        # # Assuming the audio file has .wav extension and the video has .mp4 extension
        # video_file = audio_file.replace('.wav', '.mp4')
        # metadata_file = audio_file.replace('.wav', '.csv')
        
        # # Full paths to the files
        # audio_path = os.path.join(audio_dir, audio_file)
        # video_path = os.path.join(video_dir, video_file)
        
        # # Apply audio transformation
        # output_audio_path = os.path.join(output_audio_dir, audio_file)
        # audio_transformations = acs_foa(audio_path, metadata_path, output_audio_path)
        # transformed_audio = transform_audio(audio_path)
        
        # # Apply video transformation
        # output_video_path = os.path.join(output_video_dir, video_file)
        # video_transformer = VideoPixelTransformer(video_path)
        # video_transformer.process_video(output_video_path, 1)

        # # Save the transformed audio and video (assuming the transformation functions don't already save them)
        # transformed_audio.save(os.path.join(output_audio_dir, audio_file))

def main():
    base_dir = "/Users/rithikpothuganti/cs677/new-project/SoundQ/data"  # Current directory
    subset_dir = "/dev-train-sony"
    data_dir = os.path.join(base_dir, "data")
    video_dir = os.path.join(base_dir, "video_dev")
    audio_dir = os.path.join(base_dir, "foa_dev")
    metadata_dir = os.path.join(base_dir, "metadata_dev")
    
    # Directories to save transformed audio and video
    output_audio_dir = os.path.join(base_dir, "transformed_audio")
    output_video_dir = os.path.join(base_dir, "transformed_video")
    output_metadata_dir = os.path.join(base_dir, "transformed_metadata")

    transform_audio_and_video(video_dir, audio_dir, 
                              metadata_dir, output_audio_dir, 
                              output_video_dir, output_metadata_dir)
    

if __name__ == "__main__":
    main()
