# Deepfake Detection Pipeline

This project detects deepfake videos using a transformer-based model. It organizes and splits video data, then runs detection and attention-based explanation.

## File Descriptions

| File Name (Recommended)                  | Purpose                                                                                      |
|------------------------------------------|----------------------------------------------------------------------------------------------|
| `organize_videos_by_metadata.py`         | Groups real videos with their corresponding fake videos based on metadata.                   |
| `split_videos_train_test_with_metadata.py` | Splits videos into train/test sets, preserves metadata, and moves selected videos.           |
| `deit_unmasking_optimized.py`            | Runs deepfake detection on videos, outputs predictions and attention regions.                |

## Usage

1. **Organize Videos:**  
python organize_videos_by_metadata.py --base_folder dfdc_train
2. **Split Data:**  
python split_videos_train_test_with_metadata.py
3. **Detect Deepfakes:**  
python deit_unmasking_optimized.py


## Dataset

- **Deepfake Detection Challenge (DFDC) dataset**—contains real and fake videos with metadata labels.

## Output

- **Binary classification:** REAL or FAKE for each video.
- **Attention regions:** Highlights video regions influencing the model’s decision.
- **CSV results:** Saves predictions and attention details.
