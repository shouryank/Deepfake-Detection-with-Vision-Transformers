import os
import shutil
import random
import json

# Root directory where you run this script
project_dir = 'D:\\NCSU\\sem-2\\NN\\Project'

# Subfolder to process (set manually each run)
source_root = os.path.join(project_dir, 'dfdc_train')

# Destination folders
train_dir = os.path.join(project_dir, 'train')
test_dir = os.path.join(project_dir, 'test')
final_metadata_path = os.path.join(project_dir, 'metadata.json')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load existing metadata.json
if os.path.exists(final_metadata_path):
    try:
        with open(final_metadata_path, 'r') as f:
            final_metadata = json.load(f)
    except json.JSONDecodeError:
        print("Existing metadata.json is corrupted or empty. Starting fresh.")
        final_metadata = {}
else:
    final_metadata = {}

# Get all valid subfolders
all_folders = [f for f in os.listdir(source_root)
               if os.path.isdir(os.path.join(source_root, f))]
random.shuffle(all_folders)

# 80-20 train-test split
split_index = max(1, int(len(all_folders) * 0.8)) if len(all_folders) > 1 else 1
train_folders = all_folders[:split_index]
test_folders = all_folders[split_index:]

print(f"Total subfolders: {len(all_folders)}")
print(f"Assigned to train: {len(train_folders)}")
print(f"Assigned to test: {len(test_folders)}")

# Extensions considered as videos
video_exts = ('.mp4', '.mov', '.avi', '.mkv')

def move_videos_and_metadata(folders, dest_dir):
    for folder in folders:
        folder_path = os.path.join(source_root, folder)
        metadata_path = os.path.join(folder_path, 'metadata.json')

        all_videos = [v for v in os.listdir(folder_path) if v.lower().endswith(video_exts)]
        real_video_name = folder + '.mp4'

        # Identify real and fake videos
        real_video = real_video_name if real_video_name in all_videos else None
        fake_videos = [v for v in all_videos if v != real_video_name]

        # Pick 2 fake videos randomly
        selected_fakes = random.sample(fake_videos, min(1, len(fake_videos)))

        # Move real video if exists
        if real_video:
            src = os.path.join(folder_path, real_video)
            dst = os.path.join(dest_dir, real_video)
            shutil.move(src, dst)

        # Move selected fake videos
        for fake in selected_fakes:
            src = os.path.join(folder_path, fake)
            dst = os.path.join(dest_dir, fake)
            shutil.move(src, dst)

        # Merge metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as meta_file:
                try:
                    data = json.load(meta_file)
                    for vid in [real_video] + selected_fakes:
                        if vid and vid in data:
                            final_metadata[vid] = data[vid]
                except json.JSONDecodeError:
                    print(f"Skipped invalid metadata in {folder}")

        # Optional: delete folder if empty
        try:
            os.rmdir(folder_path)
        except OSError:
            pass

# Run for both train and test
move_videos_and_metadata(train_folders, train_dir)
move_videos_and_metadata(test_folders, test_dir)

# Save updated metadata
with open(final_metadata_path, 'w') as f:
    json.dump(final_metadata, f, indent=4)

print(f"All videos moved and metadata saved. Total entries: {len(final_metadata)}")
