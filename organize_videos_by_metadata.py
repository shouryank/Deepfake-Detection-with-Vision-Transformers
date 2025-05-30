import json
import os
import shutil

def organize_videos(base_folder):
    json_file_path = os.path.join(base_folder, 'metadata.json')

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return

    for filename, metadata in data.items():
        if metadata.get('label') == 'REAL':
            folder_name = os.path.join(base_folder, filename.replace(".mp4", ""))
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print(f"Created folder: {folder_name}")
                source_path = os.path.join(base_folder, filename)
                destination_path = os.path.join(folder_name, filename)
                shutil.move(source_path, destination_path)

            for other_filename, other_metadata in data.items():
                if other_metadata.get('original') == filename:
                    try:
                        source_path = os.path.join(base_folder, other_filename)
                        destination_path = os.path.join(folder_name, other_filename)
                        if os.path.exists(source_path):
                            shutil.move(source_path, destination_path)
                            print(f"Moved video '{other_filename}' to folder '{folder_name}'")
                        else:
                            print(f"Error: Video file '{other_filename}' not found.")

                    except FileNotFoundError:
                        print(f"Error: Video file '{other_filename}' not found.")
                    except Exception as e:
                        print(f"Error moving '{other_filename}': {e}")

if __name__ == "__main__":
    base_folder_path = 'dfdc_train'
    organize_videos(base_folder_path)