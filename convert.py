import json
import os

# https://universe.roboflow.com/fire-dataset-tp9jt/fire-detection-sejra/dataset/1

# === CONFIG ===
coco_json_path = "_annotations.coco.json"  # Path to your COCO .json file
image_dir = "positives"  # Path to your images folder (can be empty if not needed)
output_txt_path = "positives.txt"  # Output file

# === LOAD JSON ===
with open(coco_json_path, 'r') as f:
    data = json.load(f)

images = {img['id']: img for img in data['images']}
annotations_by_image = {}

# === ORGANIZE ANNOTATIONS BY IMAGE ===
for ann in data['annotations']:
    img_id = ann['image_id']
    bbox = ann['bbox']  # Format: [x, y, width, height]
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(bbox)

# === WRITE TO TXT ===
with open(output_txt_path, 'w') as f:
    for img_id, image in images.items():
        filename = image['file_name']
        full_path = os.path.join(image_dir, filename)
        bboxes = annotations_by_image.get(img_id, [])
        if len(bboxes) == 0:
            continue  # Skip images with no annotations

        line = f"{full_path} {len(bboxes)}"
        for bbox in bboxes:
            x, y, w, h = map(int, bbox)
            line += f" {x} {y} {w} {h}"
        f.write(line + "\n")

print(f"Saved: {output_txt_path}")
