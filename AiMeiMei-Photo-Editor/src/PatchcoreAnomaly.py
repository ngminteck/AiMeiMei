from anomalib.models import Patchcore
import cv2
import torch

def load_patchcore():
    return Patchcore(task="segmentation").to("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = load_patchcore()
    test_image = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
    result = model(test_image)
    cv2.imwrite("patchcore_result.jpg", result)
    print("Patchcore processing complete!")