"""
Shared class-label constants for COCO-trained YOLO models.
Extend / override when using a custom model.
"""

# COCO class indices relevant to this project
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7

# Classes the detector may return
VEHICLE_CLASSES = {COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK, COCO_BICYCLE}
TWO_WHEELER_CLASSES = {COCO_MOTORCYCLE, COCO_BICYCLE}
PERSON_CLASSES = {COCO_PERSON}

# ------------------------------------------------------------------
# NOTE: Standard COCO-trained models (yolov8n.pt, yolov8s.pt …) do
# NOT include "helmet" or "no-helmet" classes.  This system uses a
# proxy approach:
#   • Helmet present   → person bounding box height/width ratio and
#                        head-region occupancy heuristic (or a fine-
#                        tuned classifier placed after detection).
#   • When a helmet-specific model is loaded, add its class indices
#     to HELMET_PRESENT_CLASSES / HELMET_ABSENT_CLASSES below.
# ------------------------------------------------------------------
HELMET_PRESENT_CLASSES: set[int] = set()   # filled by custom model
HELMET_ABSENT_CLASSES:  set[int] = set()   # filled by custom model

# Label names (for display)
COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
}

VIOLATION_TYPE_HELMET = "helmet_violation"
VIOLATION_TYPE_TRIPLE = "triple_riding"

REVIEW_STATUS_PENDING   = "pending"
REVIEW_STATUS_CONFIRMED = "confirmed"
REVIEW_STATUS_DISMISSED = "dismissed"

OUTCOME_POTENTIAL   = "potential_violation"
OUTCOME_NO_VIOLATION = "no_violation_detected"
OUTCOME_UNCERTAIN   = "uncertain"

