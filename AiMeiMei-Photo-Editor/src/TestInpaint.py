
import cv2
import numpy as np
from PIL import Image
from LaMaInpaint import SimpleLama
from ControlNetInpaint import load_controlnet
from RealesrganEnchance import load_realesrgan
from PatchcoreAnomaly import load_patchcore
from DuconetHarmonization import load_duconet, harmonize_with_duconet


def process_image(image_path, mask_path, reference_images=[]):
    test_image = Image.open(image_path).convert("RGB")
    test_mask = Image.open(mask_path).convert("L")

    # Load models
    lama_model = SimpleLama()
    controlnet_pipe = load_controlnet()
    esrgan = load_realesrgan()
    patchcore = load_patchcore()
    duconet = load_duconet()

    # Step 1: Inpainting (Small holes: LaMa, Large holes: ControlNet + LaMa)
    result = lama_model(test_image, test_mask)
    if reference_images:
        result = controlnet_pipe(prompt="Fill missing area", image=test_image, mask_image=test_mask,
                                 conditioning_image=reference_images).images[0]
        result = lama_model(result, test_mask)

    # Step 2: Artifact Detection & Fixing (PatchCore)
    result_np = np.array(result)
    anomaly_map = patchcore(result_np)
    refined_image = result_np  # Placeholder for artifact correction

    # Step 3: Harmonization using DucoNet
    refined_image = harmonize_with_duconet(duconet, refined_image)

    # Step 4: Super-Resolution with Real-ESRGAN
    enhanced_image, _ = esrgan.enhance(refined_image)

    # Save the output
    output_file = image_path.replace(".jpg", "_final.jpg")
    cv2.imwrite(output_file, enhanced_image)
    print(f"✅ Image processing complete! Saved as {output_file}.")


if __name__ == "__main__":
    process_image("test_image.jpg", "test_mask.jpg", reference_images=[Image.open("reference_1.jpg").convert("RGB")])