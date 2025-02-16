import cv2
import numpy as np
import os
import csv


def calculate_position_score(focus_object, w, h, lines):
    """Calculates the position score with a more forgiving approach to composition."""
    if not focus_object:
        return 5  # Neutral if no detected subject

    x1, y1, x2, y2 = focus_object["bbox"]
    object_x, object_y = (x1 + x2) // 2, (y1 + y2) // 2
    thirds_x, thirds_y = [w // 3, 2 * w // 3], [h // 3, 2 * h // 3]

    position_score = 10  # Start at perfect and apply gradual reductions

    # **Rule of Thirds Boost (More Forgiving)**
    near_thirds_x = min(abs(object_x - thirds_x[0]), abs(object_x - thirds_x[1])) < w * 0.08  # 8% tolerance
    near_thirds_y = min(abs(object_y - thirds_y[0]), abs(object_y - thirds_y[1])) < h * 0.08

    if near_thirds_x or near_thirds_y:
        position_score = 8  # Proper thirds placement (high score)
    elif abs(object_x - w // 2) < w * 0.12 and abs(object_y - h // 2) < h * 0.12:
        position_score = 6  # Near center but not exactly dead center
    elif abs(object_x - w // 2) < w * 0.05 and abs(object_y - h // 2) < h * 0.05:
        position_score = 5  # Dead center but still reasonable

    # **Dead-Center Penalty (Less Strict)**
    if abs(object_x - w // 2) < w * 0.02 and abs(object_y - h // 2) < h * 0.02:
        if lines is None or len(lines) < 8:  # If no strong symmetry
            position_score = 4  # Still not too harsh
        else:
            position_score = 6  # If symmetry is present, allow higher score

    # **Too-Close Penalty (Strong Reduction)**
    subject_area = (x2 - x1) * (y2 - y1)
    frame_area = w * h
    if subject_area / frame_area > 0.5:
        position_score = max(3, position_score - 3)  # Strong penalty if subject is too close

    return round(position_score, 2)


def calculate_photo_score(frame, objects):
    """Evaluates the photo-taking score with improved composition checks."""
    h, w, _ = frame.shape

    if not objects:
        return {
            "Final Score": 2, "Position": 2, "Angle": 2, "Lighting": 2, "Focus": 2,
            "Feedback": ["No subject detected."], "Suggestions": ["Move subject into frame."]
        }

    # **Scene-Based Scores (Lighting, Angle, Sharpness)**
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # **1️⃣ Angle Score (Horizon & Perspective Distortion)**
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    angle_score = 5  # Neutral default
    if lines is not None:
        angles = [abs(np.degrees(theta)) for rho, theta in lines[:, 0]]
        avg_angle = np.mean(angles)
        angle_deviation = abs(avg_angle - 90)
        angle_score = round(max(1, 10 - (angle_deviation / 6)), 2)  # More severe penalty

    # **2️⃣ Lighting Score (Brightness & Shadows)**
    brightness = np.mean(gray)
    lighting_score = round(max(1, min(10, 10 - abs(130 - brightness) / 7)), 2)  # Tighter scaling

    # **Shadow Detection (Higher Variance Penalty)**
    shadow_variance = np.std(gray)
    if shadow_variance > 70:  # Increased shadow penalty threshold
        lighting_score = max(1, lighting_score - 3)

    # **3️⃣ Sharpness & Focus Score (Depth of Field Check)**
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance < 30:
        focus_score = 1  # Very blurry
    elif variance > 150:
        focus_score = 10  # Exceptionally sharp
    else:
        focus_score = round(max(1, min(10, (variance - 30) / 12)), 2)  # Stronger penalty

    # **4️⃣ Subject Position & Composition Score**
    focus_object = max(objects, key=lambda obj: obj["confidence"], default=None)
    position_score = calculate_position_score(focus_object, w, h, lines)

    # **Final Score Calculation (Recalibrated Weights)**
    final_score = round(
        (position_score * 0.35) +
        (angle_score * 0.2) +
        (lighting_score * 0.25) +
        (focus_score * 0.2), 2
    )

    # **Generate Feedback & Suggestions**
    feedback = []
    suggestions = []

    if position_score < 5:
        feedback.append("Reposition subject to rule of thirds.")
        if focus_object:
            x1, y1, x2, y2 = focus_object["bbox"]
            object_x, object_y = (x1 + x2) // 2, (y1 + y2) // 2
            thirds_x, thirds_y = [w // 3, 2 * w // 3], [h // 3, 2 * h // 3]

            if object_x < thirds_x[0]:
                suggestions.append("Move subject to the right.")
            elif object_x > thirds_x[1]:
                suggestions.append("Move subject to the left.")
            if object_y < thirds_y[0]:
                suggestions.append("Move subject lower.")
            elif object_y > thirds_y[1]:
                suggestions.append("Move subject higher.")

    if angle_score < 5:
        feedback.append("Align camera to avoid tilt.")
        suggestions.append("Adjust camera to straighten horizon.")

    if lighting_score < 5:
        feedback.append("Adjust brightness for better exposure.")
        if brightness < 100:
            suggestions.append("Increase lighting or use flash.")
        elif brightness > 180:
            suggestions.append("Reduce exposure to avoid overexposure.")
        if shadow_variance > 70:
            suggestions.append("Avoid harsh lighting; use a diffuser or reposition subject.")

    if focus_score < 5:
        feedback.append("Hold camera steady to avoid blur.")
        suggestions.append("Use a tripod or stabilize hands.")

    return {
        "Final Score": final_score,
        "Position": position_score,
        "Angle": angle_score,
        "Lighting": lighting_score,
        "Focus": focus_score,
        "Feedback": feedback,
        "Suggestions": suggestions
    }


def save_photo_score(image_path, score_data):
    """Saves the photo-taking score to a CSV file."""
    csv_file = "photo_scores.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["Image", "Final Score", "Position", "Angle", "Lighting", "Focus", "Feedback", "Suggestions"]
            )
        writer.writerow([
            image_path, score_data["Final Score"], score_data["Position"], score_data["Angle"],
            score_data["Lighting"], score_data["Focus"], "; ".join(score_data["Feedback"]),
            "; ".join(score_data["Suggestions"])
        ])
