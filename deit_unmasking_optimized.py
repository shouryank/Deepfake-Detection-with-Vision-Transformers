import os
import json
import csv
import cv2
import numpy as np
import tensorflow as tf
from transformers import DeiTConfig, TFDeiTForImageClassification, DeiTImageProcessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------
# Setup paths
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(current_dir, "train")
test_dir = os.path.join(current_dir, "test")
metadata_path = os.path.join(current_dir, "metadata.json")
output_csv = os.path.join(current_dir, "attention_results.csv")

# ----------------------------
# Extract multiple frames per video
# ----------------------------
def extract_multiple_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * total_frames // num_frames)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return np.array(frames)

# ----------------------------
# Patch (row, col) to body region description
# ----------------------------
def patch_to_region(row, col):
    if row < 4:
        return "Forehead"
    elif row < 7:
        return "Eyes / Nose"
    elif row < 10:
        return "Mouth / Cheeks"
    else:
        return "Chin / Jaw"

# ----------------------------
# Load DeiT model with attention output
# ----------------------------
print("Loading model...")
config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224", output_attentions=True)
model = TFDeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")

# ----------------------------
# Load metadata
# ----------------------------
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# ----------------------------
# Prediction + unmasking per video
# ----------------------------
results = []
print("Processing videos...")

for video_name in os.listdir(test_dir):
    if not video_name.endswith(".mp4") or video_name not in metadata:
        continue

    video_path = os.path.join(test_dir, video_name)
    label = metadata[video_name]['label']
    frames = extract_multiple_frames(video_path)

    if len(frames) == 0:
        continue

    predictions = []
    attention_scores = []

    for frame in frames:
        inputs = processor(images=frame, return_tensors="tf")
        outputs = model(**inputs)

        # Prediction
        logits = outputs.logits
        softmax_score = tf.nn.softmax(logits, axis=1).numpy()[0]
        predictions.append(softmax_score[1])  # FAKE class probability

        # Attention
        attn = outputs.attentions[-1]  # last layer attention
        attn_map = tf.reduce_mean(attn, axis=1)[0]  # avg heads
        patch_attn = attn_map[0, 1:]  # CLS token to patches
        attention_scores.append(patch_attn.numpy())

    # Average predictions
    avg_score = np.mean(predictions)
    pred_label = "FAKE" if avg_score > 0.5 else "REAL"

    if pred_label == "FAKE":
        # Average attention across frames
        avg_attn = np.mean(attention_scores, axis=0)
        max_patch = np.argmax(avg_attn)
        row, col = divmod(max_patch, 14)
        region = patch_to_region(row, col)
        results.append([video_name, pred_label, f"({row}, {col})", region])
    else:
        results.append([video_name, pred_label, "-", "-"])

# ----------------------------
# Save results to CSV
# ----------------------------
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Video Name", "Prediction", "Attention Patch (row,col)", "Region Focus"])
    writer.writerows(results)

print(f"\nDone! Results saved to: {output_csv}")
