import os
import subprocess

# Source directory containing subdirectories with audio files
source_directory = "./path/to/METU"
aud_fmt = "em32" # em32

# Iterate through each subdirectory
for subdir in os.listdir(source_directory):
    subdir_path = os.path.join(source_directory, subdir)
    
    # Check if it's a directory
    if os.path.isdir(subdir_path):
        # List all the audio files in the subdirectory
        audio_files = sorted([
            os.path.join(subdir_path, file) 
            for file in os.listdir(subdir_path) 
            if file.endswith(".wav")
        ])
        audio_files = audio_files[:32]
        # Ensure there are 32 audio files
        if len(audio_files) == 32:
            # Prepare the command to merge audio files
            merge_command = ["sox"] + ["-M"] + audio_files + [os.path.join(subdir, f"IR_{aud_fmt}.wav")]
            print(merge_command)
            # Execute the command
            subprocess.run(merge_command)

            print(f"Merged files for {subdir}")

        else:
            print(f"Not enough audio files in {subdir}")

    else:
        print(f"{subdir} is not a directory")

print("All files merged successfully.")
