import torch
import cv2
import numpy as np
from PIL import Image

def load_duconet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load("models/DucoNet1024.pth", map_location=device)
    model.to(device).eval()
    return model

def harmonize_with_duconet(model, composite_image):
    composite_image_lab = cv2.cvtColor(composite_image, cv2.COLOR_BGR2LAB)
    composite_image_lab = Image.fromarray(composite_image_lab)
    transform = torch.nn.Sequential(
        torch.nn.Upsample((256, 256)),
        torch.nn.Identity()
    )
    input_tensor = transform(torch.tensor(np.array(composite_image_lab)).permute(2, 0, 1).unsqueeze(0).float()).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = output_tensor.squeeze().cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))
    output_image = (output_image * 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_LAB2BGR)
    return output_image

if __name__ == "__main__":
    model = load_duconet()
    test_image = cv2.imread("test_image.jpg")
    harmonized_image = harmonize_with_duconet(model, test_image)
    cv2.imwrite("duconet_result.jpg", harmonized_image)
    print("DucoNet harmonization complete!")