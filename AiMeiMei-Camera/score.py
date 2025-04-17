import cv2
import numpy as np
import os
import csv
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

##############################################
# SPAQ Model and Image Preparation Definitions
##############################################

class MTA(nn.Module):
    """
    Multi-Task Learning (MTA) model for SPAQ.
    This model uses a ResNet50 backbone modified to output 6 values.
    (Assuming the first 5 outputs correspond to Brightness, Colorfulness,
     Contrast, Noisiness, and Sharpness, respectively.)
    """

    def __init__(self):
        super(MTA, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 6, bias=True)

    def forward(self, x):
        return self.backbone(x)


class ImageLoad(object):
    """
    Given a PIL image, adaptively resizes it and generates overlapping patches.
    This is based on the SPAQ demo's image preparation.
    """

    def __init__(self, size, stride, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.stride = stride
        self.interpolation = interpolation

    def __call__(self, img):
        image = self.adaptive_resize(img)
        return self.generate_patches(image, input_size=self.stride)

    def adaptive_resize(self, img):
        w, h = img.size
        if h < self.size or w < self.size:
            return img
        else:
            return transforms.ToTensor()(transforms.Resize(self.size, self.interpolation)(img))

    def to_numpy(self, image):
        p = image.numpy()
        return p.transpose((1, 2, 0))

    def generate_patches(self, image, input_size, dtype=np.float32):
        img = self.to_numpy(image)
        img_shape = img.shape
        img = img.astype(dtype=dtype)
        if len(img_shape) == 2:
            H, W = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray([img] * 3, dtype=img.dtype)
        stride = int(input_size / 2)  # overlapping patches
        hIdxMax = H - input_size
        wIdxMax = W - input_size
        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_numpy = [img[hId:hId + input_size, wId:wId + input_size, :]
                         for hId in hIdx
                         for wId in wIdx]
        patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        return patches_tensor


##############################################
# SPAQ Quality Assessor Class (with Attribute Scores)
##############################################

class SPAQQualityAssessor:
    def __init__(self, model_path="MT-A_release.pt", patch_size=512, stride=224):
        self.model = MTA()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if os.path.isfile(model_path):
            print(f"[*] Loading SPAQ checkpoint from '{model_path}'")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("SPAQ model loaded successfully!")
        else:
            raise IOError(f"SPAQ model file not found at: {model_path}")
        self.model.eval()
        self.patch_transform = ImageLoad(size=patch_size, stride=stride)

    def assess(self, pil_img):
        """
        Given a PIL image, generate overlapping patches and compute the attribute scores.
        Returns a dictionary with keys:
          'Brightness', 'Colorfulness', 'Contrast', 'Noisiness', and 'Sharpness'
        computed as the average output (over patches) from the corresponding channels.
        """
        patches = self.patch_transform(pil_img)  # Tensor shape: [N, C, H, W]
        patches = patches.to(self.device)
        with torch.no_grad():
            outputs = self.model(patches)  # outputs shape: [N, 6]

        attr_scores = outputs[:, :6].mean(dim=0).cpu().numpy().tolist()
        print(attr_scores)
        # Average the first 5 channels over all patches.
        # Assumed order: 0: Brightness, 1: Colorfulness, 2: Contrast, 3: Noisiness, 4: Sharpness.
        attr_scores = outputs[:, :5].mean(dim=0).cpu().numpy().tolist()
        # Scale from original 0-1 to 0-10
        attr_scores = [round(s * 0.1, 2) for s in attr_scores]
        return {
            "Brightness": attr_scores[0],
            "Colorfulness": attr_scores[1],
            "Contrast": attr_scores[2],
            "Noisiness": attr_scores[3],
            "Sharpness": attr_scores[4],
        }


##############################################
# Global SPAQ Assessor Instance
##############################################
try:
    spaq_assessor = SPAQQualityAssessor(model_path="MT-A_release.pt")
except Exception as e:
    print("Error initializing SPAQ Quality Assessor:", e)
    spaq_assessor = None

##############################################
# Composition & Scene Evaluation Functions
##############################################

def calculate_position_score(focus_object, w, h):
    """
    Calculate position score based on how close the subject's center is
    to ideal rule-of-thirds intersections.
    Returns a float rounded to 2 decimal places.
    """
    if not focus_object:
        return 5.0

    x1, y1, x2, y2 = focus_object["bbox"]
    object_x, object_y = (x1 + x2) // 2, (y1 + y2) // 2

    thirds_x = [w // 3, 2 * w // 3]
    thirds_y = [h // 3, 2 * h // 3]
    intersections = [(tx, ty) for tx in thirds_x for ty in thirds_y]

    distances = [np.sqrt((object_x - ix) ** 2 + (object_y - iy) ** 2) for ix, iy in intersections]
    min_distance = min(distances)
    max_possible = np.sqrt((w / 3) ** 2 + (h / 3) ** 2)
    pos_ratio = min_distance / max_possible

    score = round(10 - pos_ratio * 5, 2)
    subject_area = (x2 - x1) * (y2 - y1)
    frame_area = w * h
    if subject_area / frame_area > 0.5:
        score -= 3
    score = max(0.0, score)
    return round(score, 2)


def calculate_angle_score(gray):
    """
    Calculate an angle score based on the median deviation from 90°.
    Returns a float rounded to 2 decimal places.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return 5.0
    angles = [abs(np.degrees(theta)) for rho, theta in lines[:, 0]]
    valid = [a for a in angles if 70 <= a <= 110]
    if not valid:
        return 5.0
    median_angle = np.median(valid)
    deviation = abs(median_angle - 90)
    score = round(max(1, 10 - (deviation / 6)), 2)
    return score



def calculate_photo_score(frame, objects):
    """Evaluate the overall photo score based on composition, angle, lighting, and focus."""
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert the frame to a PIL RGB image.
    if frame.shape[2] == 4:
        frame_cv = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        frame_cv = frame
    pil_img = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))

    # Obtain SPAQ attribute scores.
    if spaq_assessor is None:
        spaq_data = {
            "Brightness": 5,
            "Colorfulness": 5,
            "Contrast": 5,
            "Noisiness": 5,
            "Sharpness": 5,
        }
        spa_feedback = ["SPAQ model not initialized."]
    else:
        try:
            spaq_data = spaq_assessor.assess(pil_img)
            spa_feedback = []
        except Exception as e:
            print("Error during SPAQ assessment:", e)
            spaq_data = {
                "Brightness": 5,
                "Colorfulness": 5,
                "Contrast": 5,
                "Noisiness": 5,
                "Sharpness": 5,
            }
            spa_feedback = ["Error assessing image quality."]

    # Compute the SPAQ composite quality score as an average of the five attributes,
    # using the inverted noisiness score.
    spaq_composite = (spaq_data["Brightness"] +
                      spaq_data["Colorfulness"] +
                      spaq_data["Contrast"] +
                      spaq_data["Noisiness"]+
                      spaq_data["Sharpness"]) / 5.0

    feedback = list(spa_feedback)
    suggestions = []

    # SPAQ attribute-based suggestions.
    if spaq_data["Brightness"] < 5:
        feedback.append("Image is too dark.")
        suggestions.append("Increase brightness or exposure.")
    if spaq_data["Colorfulness"] < 5:
        feedback.append("Image lacks vibrancy.")
        suggestions.append("Enhance saturation or adjust color balance.")
    if spaq_data["Contrast"] < 5:
        feedback.append("Image contrast is low.")
        suggestions.append("Increase contrast for more definition.")
    if spaq_data["Noisiness"] < 5:  # higher noisiness is worse
        feedback.append("Image is noisy.")
        suggestions.append("Apply noise reduction or use better lighting.")
    if spaq_data["Sharpness"] < 5:
        feedback.append("Image appears blurry.")
        suggestions.append("Use a tripod or improve focus.")

    angle_score = calculate_angle_score(gray)

    if angle_score < 5:
        feedback.append("Camera tilt detected.")
        suggestions.append("Reposition camera to reduce tilt.")

    position_score = 5
    if objects:
        focus_object = max(objects, key=lambda obj: obj["confidence"], default=None)
        if focus_object:
            position_score = calculate_position_score(focus_object, w, h, None)
            if position_score < 5:
                feedback.append("Subject placement could improve.")
                if focus_object:
                    x1, y1, x2, y2 = focus_object["bbox"]
                    object_x, object_y = (x1 + x2) // 2, (y1 + y2) // 2
                    if object_x < w // 3:
                        suggestions.append("Move subject to the right.")
                    elif object_x > 2 * w // 3:
                        suggestions.append("Move subject to the left.")
                    if object_y < h // 3:
                        suggestions.append("Move subject downward.")
                    elif object_y > 2 * h // 3:
                        suggestions.append("Move subject upward.")

    # Define weights for composite final score.
    WEIGHT_SPAQ = 0.55
    WEIGHT_POSITION = 0.25
    WEIGHT_ANGLE = 0.20

    final_score = round(
        spaq_composite * WEIGHT_SPAQ +
        position_score * WEIGHT_POSITION +
        angle_score * WEIGHT_ANGLE, 2
    )

    return {
        "Final Score": final_score,
        "Position": position_score,
        "Angle": angle_score,
        "Brightness": spaq_data["Brightness"],
        "Sharpness": spaq_data["Sharpness"],
        "Colorfulness": spaq_data["Colorfulness"],
        "Contrast": spaq_data["Contrast"],
        "Noisiness": spaq_data["Noisiness"],
        "Feedback": feedback,
        "Suggestions": suggestions
    }


