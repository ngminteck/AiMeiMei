import torch
import os
import numpy as np
from PIL import Image


class SimpleLama:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "models/big-lama.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LaMa model not found at {model_path}. Please download it.")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval().to(self.device)

    def __call__(self, image, mask):
        """ Perform inpainting using LaMa model while ensuring correct dimensions. """

        # Convert image and mask to NumPy arrays to force exact shape matching
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Ensure same size before tensor conversion
        min_width = min(image_np.shape[1], mask_np.shape[1])
        min_height = min(image_np.shape[0], mask_np.shape[0])

        # Resize both images to the smallest found dimensions
        image = image.resize((min_width, min_height), Image.Resampling.NEAREST)
        mask = mask.resize((min_width, min_height), Image.Resampling.NEAREST)

        # Convert to NumPy after enforcing exact same size
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Debugging print statements
        print(f"✅ Image Shape: {image_np.shape}, Mask Shape: {mask_np.shape}")

        # Convert to tensors
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).float() / 255.0

        # Move to device
        image_tensor, mask_tensor = image_tensor.to(self.device), mask_tensor.to(self.device)

        # Inpaint (Swap order: mask first, then image)
        with torch.no_grad():
            inpainted = self.model(mask_tensor, image_tensor)

        # Convert back to image
        result = (inpainted[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)


def generate_mask_from_alpha(image):
    """ Create a binary mask from the alpha channel where transparency is masked. """
    if image.mode != "RGBA":
        raise ValueError("Input image must be in RGBA format.")

    # Extract alpha channel
    alpha = np.array(image.getchannel("A"))
    mask = (alpha == 0).astype(np.uint8) * 255  # 255 for transparent, 0 for opaque

    # Convert to grayscale mask
    mask_img = Image.fromarray(mask, mode="L")

    return mask_img


if __name__ == "__main__":
    model = SimpleLama()

    # Load input PNG with transparency
    input_path = "images/segmentation_result/2_people_together/2_people_together_background.png"
    input_png = Image.open(input_path).convert("RGBA")

    # Extract original filename
    filename = os.path.basename(input_path)

    # Preserve original size
    original_size = input_png.size

    # Generate mask and resize to match input dimensions **EXACTLY**
    mask = generate_mask_from_alpha(input_png)
    mask = mask.resize(original_size, Image.Resampling.NEAREST)

    # Convert RGBA to RGB for LaMa processing
    image_rgb = input_png.convert("RGB")
    image_rgb = image_rgb.resize(original_size, Image.Resampling.NEAREST)  # Enforce size consistency

    # Inpaint the image
    result = model(image_rgb, mask)

    # Resize result back to original size (if needed)
    result = result.resize(original_size, Image.Resampling.LANCZOS)

    # Define output directory and save the file
    output_dir = "images/lama_result"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Save output
    result.save(output_path)

    print(f"✅ Saved inpainted image to: {output_path}")
