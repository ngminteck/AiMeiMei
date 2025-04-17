import cv2
import screeninfo
import os
from datetime import datetime
from detector import detect_main_object  # Import main object detection
from score import calculate_photo_score  # Import scoring functions


def draw_camera_grid(frame):
    """Draws a 3x3 grid on the camera feed."""
    height, width, _ = frame.shape
    x_step = width // 3
    y_step = height // 3
    for i in range(1, 3):
        cv2.line(frame, (i * x_step, 0), (i * x_step, height), (255, 255, 255), 2)
        cv2.line(frame, (0, i * y_step), (width, i * y_step), (255, 255, 255), 2)
    return frame


def set_camera_resolution(cap, width, height):
    """Forces the camera to match the screen's resolution & aspect ratio."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def save_photo(raw_frame, score_data):
    """Saves the raw camera frame with a timestamp and stores the score."""
    if not os.path.exists("captured_photos"):
        os.makedirs("captured_photos")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_photos/photo_{timestamp}.jpg"
    cv2.imwrite(filename, raw_frame)
    print(f"Photo saved as: {filename} with score {score_data['Final Score']}")


def draw_text(frame, text, position, font_scale=1, thickness=2, color=(255, 255, 255)):
    """Draws outlined text on the frame to ensure visibility."""
    x, y = position
    cv2.putText(frame, text, (x, y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height
print(f"Detected Screen Resolution: {screen_width}x{screen_height}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
set_camera_resolution(cap, screen_width, screen_height)

cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

capture_message_timer = 0
capture_message_duration = 30  # Frames (~1 second at 30 FPS)

while True:
    ret, raw_frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    display_frame = cv2.resize(raw_frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
    display_frame = draw_camera_grid(display_frame)

    # Detect main object and draw bounding box
    display_frame, main_object = detect_main_object(display_frame)

    # Get real-time photo-taking score
    score_data = calculate_photo_score(display_frame, [main_object] if main_object else [])

    # Display score & feedback
    draw_text(display_frame, f"Overall Score: {score_data['Final Score']}/10", (50, 50), font_scale=1.5,
              color=(0, 255, 0))
    draw_text(display_frame, f"Position: {score_data['Position']}/10 | Angle: {score_data['Angle']}/10", (50, 100), font_scale=1, color=(0, 255, 255))
    draw_text(display_frame,f"Brightness: {score_data['Brightness']}/10 | Sharpness: {score_data['Sharpness']}/10",(50, 150), font_scale=1, color=(0, 255, 255))
    draw_text(display_frame, f"Colorfulness: {score_data['Colorfulness']}/10 | Contrast: {score_data['Contrast']}/10 | Noisiness: {score_data['Noisiness']}/10",(50, 200), font_scale=1, color=(0, 255, 255))

    y_offset = 250
    for feedback in score_data["Suggestions"][:3]:  # Show only top 3 suggestions
        draw_text(display_frame, feedback, (50, y_offset), font_scale=0.8, color=(0, 0, 255))
        y_offset += 30

    draw_text(display_frame, "Press 'S' to Capture | Press 'Q' to Quit", (50, screen_height - 50), font_scale=1.2,
              thickness=3)

    if capture_message_timer > 0:
        draw_text(display_frame, "Scene Captured!", (screen_width // 2 - 100, 100), font_scale=2, thickness=4,
                  color=(0, 255, 0))
        capture_message_timer -= 1

    cv2.imshow("Camera Feed", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        save_photo(raw_frame, score_data)
        capture_message_timer = capture_message_duration

cap.release()
cv2.destroyAllWindows()