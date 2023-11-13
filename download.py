import csv
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
import os
from pydub import AudioSegment

# Directory names for each class
class_directories = {
    '0': 'Class_0_Female_Speech',
    '1': 'Class_1_Male_Speech',
    '2': 'Class_2_Clapping',
    '3': 'Class_3_Telephone',
    '4': 'Class_4_Laughter',
    '5': 'Class_5_Domestic_Sounds',
    '6': 'Class_6_Walk_Footsteps',
    '7': 'Class_7_Door_Open_Close',
    '8': 'Class_8_Music',
    '9': 'Class_9_Musical_Instrument',
    '10': 'Class_10_Water_Tap_Faucet',
    '11': 'Class_11_Bell',
    '12': 'Class_12_Knock'
}

# Set the base directory for the dataset
dataset_base_dir = "/Volumes/T7/SoundQ-YT-data" #"/Users/rithikpothuganti/cs677/new-project/SoundQ/data/download_data/Dataset"

def ensure_directory_structure():
    if not os.path.exists(dataset_base_dir):
        os.mkdir(dataset_base_dir)

    for dir_name in class_directories.values():
        class_dir_path = os.path.join(dataset_base_dir, dir_name)
        if not os.path.exists(class_dir_path):
            os.mkdir(class_dir_path)

def download_and_trim(url, start_time, end_time, class_label, sequence_number):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension="mp4", progressive=True).first() #yt.streams.get_highest_resolution()
        temp_filename = f"{yt.video_id}.mp4"

        # Download to a temporary file
        stream.download(filename=temp_filename)

        output_directory = os.path.join(dataset_base_dir, class_directories[class_label])
        output_filename = os.path.join(output_directory, f"clip{sequence_number}.mp4")
        
        # Trim the video
        with VideoFileClip(temp_filename) as video:
            new = video.subclip(start_time, end_time)
            audio = AudioFileClip(temp_filename).subclip(start_time, end_time)
            new = new.set_audio(audio)
            new.write_videofile(output_filename, codec="libx264", audio_codec="aac")

        os.remove(temp_filename)
    except Exception as e:
        print(f"Error processing {url}: {e}")

def convert_time(time_str):
    h, m, s = 0, 0, 0
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        m, s = map(int, parts)
    elif len(parts) == 1:
        s = int(parts[0])
    return h * 3600 + m * 60 + s

def process_csv(csv_file):
    ensure_directory_structure()

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        video_counter = {}
        for row in reader:
            class_label = row['class']
            video_counter[class_label] = video_counter.get(class_label, 0) + 1

            start_seconds = convert_time(row['start'])
            end_seconds = convert_time(row['end'])
            download_and_trim(row['link'], start_seconds, end_seconds, class_label, video_counter[class_label])

csv_path = "./dataset.csv"
process_csv(csv_path)
