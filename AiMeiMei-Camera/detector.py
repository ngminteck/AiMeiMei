import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

DISTANCE_THRESHOLD = 100  # Maximum distance in pixels to group objects


def detect_objects(frame):
    """Detect objects in an image and return bounding boxes."""
    results = model(frame)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            detected_objects.append({"confidence": confidence, "bbox": (x1, y1, x2, y2)})

    return detected_objects  # Returns a list of detected objects


def are_close(box1, box2):
    """Checks if two bounding boxes are close enough to be grouped."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute center points of both boxes
    center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
    center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)

    # Compute Euclidean distance
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    return distance < DISTANCE_THRESHOLD  # Return True if close


def group_objects(objects):
    """Groups all closely positioned objects into clusters."""
    if not objects:
        return []  # No objects detected

    grouped_clusters = []
    used = set()

    for i, obj in enumerate(objects):
        if i in used:
            continue  # Skip already grouped objects
        close_group = [obj]
        used.add(i)

        for j, other_obj in enumerate(objects):
            if j != i and j not in used and are_close(obj["bbox"], other_obj["bbox"]):
                close_group.append(other_obj)
                used.add(j)

        # Merge group into a single bounding box
        x1 = min(o["bbox"][0] for o in close_group)
        y1 = min(o["bbox"][1] for o in close_group)
        x2 = max(o["bbox"][2] for o in close_group)
        y2 = max(o["bbox"][3] for o in close_group)

        grouped_clusters.append({
            "confidence": max(o["confidence"] for o in close_group),  # Keep the highest confidence
            "bbox": (x1, y1, x2, y2)
        })

    return grouped_clusters


def select_focus_object(grouped_clusters):
    """Selects the most relevant group as the focus object."""
    if not grouped_clusters:
        return None

    # Select the largest group or highest confidence object
    focus_object = max(grouped_clusters, key=lambda obj: (obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1]), default=None)

    return focus_object


def detect_main_object(frame):
    """Detects the main object and returns the modified frame with bounding box."""
    objects = detect_objects(frame)

    if not objects:
        return frame, None  # No object detected

    # First group all close objects
    grouped_clusters = group_objects(objects)

    # Then select the focus object from the grouped clusters
    main_object = select_focus_object(grouped_clusters)

    # Draw bounding box for main object (no labels)
    frame = draw_bounding_box(frame, main_object) if main_object else frame

    return frame, main_object


def draw_bounding_box(frame, obj):
    """Draws a single bounding box without labels."""
    if obj is None:
        return frame
    x1, y1, x2, y2 = obj["bbox"]
    color = (0, 255, 0)  # Green for detected main object
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame
