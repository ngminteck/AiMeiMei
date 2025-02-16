from realesrgan import RealESRGANer
import cv2

def load_realesrgan():
    return RealESRGANer(model_path="models/RealESRGAN_x8.pth")

if __name__ == "__main__":
    model = load_realesrgan()
    test_image = cv2.imread("test_image.jpg")
    enhanced, _ = model.enhance(test_image)
    cv2.imwrite("realesrgan_result.jpg", enhanced)