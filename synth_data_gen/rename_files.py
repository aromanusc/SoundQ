import os

def rename_files_in_directory(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    files.sort()

    for i, filename in enumerate(files, start=1):
        new_name = f"{i:03d}{os.path.splitext(filename)[1]}"
        
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)

        os.rename(old_file, new_file)

        print(f"Renamed '{filename}' to '{new_name}'")

base_dir = '/Users/rithikpothuganti/cs677/new-project/SoundQ/data/Dataset/'

subdirs = [x[0] for x in os.walk(base_dir)]
subdirs = subdirs[1:]

for subdir in subdirs:
    rename_files_in_directory(subdir)