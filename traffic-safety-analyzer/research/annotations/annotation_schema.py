"""
Dataset annotation schema for the Traffic Safety Analyzer.

Copy this file to your dataset directory and label your images
following the YOLO format described below.
"""

ANNOTATION_SCHEMA = """
# Dataset Annotation Schema
# Traffic Safety Analyzer — YOLO Format

## Directory Layout
    my_dataset/
    ├── images/
    │   ├── train/   ← training images (.jpg, .png)
    │   └── val/     ← validation images (.jpg, .png)
    ├── labels/
    │   ├── train/   ← YOLO label files (.txt) matching image basenames
    │   └── val/
    └── data.yaml    ← dataset config (see below)

## Label File Format (YOLO)
One detection per line:
    <class_id>  <cx>  <cy>  <w>  <h>
    
All values are normalised to [0, 1] relative to image width/height.
cx, cy = center of bounding box; w, h = width and height.

## Class IDs (Standard COCO subset)
    0  person
    3  motorcycle

## Extended Class IDs (for custom helmet model)
    4  helmet          ← rider WITH a helmet
    5  no_helmet       ← rider WITHOUT a helmet

## data.yaml (for YOLOv8 training)
    path: /absolute/path/to/my_dataset
    train: images/train
    val: images/val
    nc: 6
    names:
      0: person
      1: bicycle
      2: car
      3: motorcycle
      4: helmet
      5: no_helmet

## Annotation Guidelines
- Label ALL motorcycles visible in the frame (even partial ones).
- Label ALL persons on or beside motorcycles.
- For helmet detection, label each rider's head region:
    - Use class 4 (helmet) if a helmet is clearly visible.
    - Use class 5 (no_helmet) if the head is visible without a helmet.
    - Do NOT label if the head is occluded or the image is too small.
- Exclude pedestrians standing far from any motorcycle.
- Split data by recording session, not randomly across frames.

## Recommended Annotation Tools
- CVAT (https://cvat.ai)
- Roboflow (https://roboflow.com)
- Label Studio (https://labelstud.io)

## Minimum Dataset Size (for reasonable results)
- 500+ labeled images for helmet detection
- 300+ images covering triple-riding scenarios
- Balanced split: ~80% train / 20% val, by recording session
"""

if __name__ == "__main__":
    print(ANNOTATION_SCHEMA)

