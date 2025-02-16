import requests
import os
import json
import base64
import time
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


# Function to Read API Key from a Text File
def get_api_key(filename="../../keys/googlevision.txt"):
    """Reads the API key from a text file."""
    with open(filename, "r") as file:
        return file.read().strip()


# Read API Key
GOOGLE_API_KEY = get_api_key()


def encode_image(image_path):
    """Convert an image file (local or URL) to base64."""
    if image_path.startswith("http"):
        response = requests.get(image_path)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            raise ValueError(f"Failed to download image from {image_path}, status code: {response.status_code}")
    elif os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        raise ValueError(f"Invalid image path: {image_path}")


def get_image_gps(image_path):
    """Extract GPS coordinates from image metadata (if available)."""
    img = Image.open(image_path)
    exif_data = img.getexif()

    if not exif_data:
        print("❌ No EXIF metadata found in image.")
        return None, None  # No GPS data found

    gps_info = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for gps_tag in value:
                sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                gps_info[sub_tag] = value[gps_tag]

    if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
        lat = gps_info["GPSLatitude"]
        lon = gps_info["GPSLongitude"]
        lat_ref = gps_info["GPSLatitudeRef"]
        lon_ref = gps_info["GPSLongitudeRef"]

        def convert_to_decimal(coord, ref):
            d, m, s = coord
            decimal = d + (m / 60.0) + (s / 3600.0)
            if ref in ["S", "W"]:
                decimal = -decimal
            return decimal

        latitude = convert_to_decimal(lat, lat_ref)
        longitude = convert_to_decimal(lon, lon_ref)
        print(f"📍 Detected GPS Coordinates: Latitude={latitude}, Longitude={longitude}")
        return latitude, longitude

    print("❌ No GPS data found in EXIF metadata.")
    return None, None


def download_images(image_urls, uploaded_image_path, save_folder="images/google_similar_images"):
    """Downloads images from given URLs into a folder named after the uploaded file."""
    uploaded_filename = os.path.splitext(os.path.basename(uploaded_image_path))[0]
    image_folder = os.path.join(save_folder, uploaded_filename)
    os.makedirs(image_folder, exist_ok=True)

    print("\n📥 Starting Download...\n")

    for idx, img_url in enumerate(image_urls, start=1):
        image_path = os.path.join(image_folder, f"{uploaded_filename}_{idx}.jpg")
        for attempt in range(3):
            try:
                print(f"🔄 Downloading ({attempt + 1}/3): {img_url}")
                response = requests.get(img_url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(image_path, "wb") as file:
                        for chunk in response.iter_content(1024):
                            file.write(chunk)
                    print(f"✅ Downloaded: {image_path}")
                    break
                else:
                    print(f"⚠️ Failed to download (Status Code: {response.status_code})")
            except requests.exceptions.RequestException as e:
                print(f"❌ Error: {e} (Attempt {attempt + 1}/3)")
            time.sleep(2)
        else:
            print(f"❌ Skipping image {img_url} after 3 failed attempts.")


def check_similarity(image_gps, image_landmarks, similar_image_gps, similar_image_landmarks):
    """Check if the GPS and landmarks match between the uploaded and similar images."""
    gps_match = image_gps == similar_image_gps
    landmark_match = any(landmark in similar_image_landmarks for landmark in image_landmarks)
    return gps_match or landmark_match


def get_landmarks(image_path):
    """Detect landmarks in the given image."""
    print(f"📤 Detecting landmarks in image: {image_path}")
    base64_image = encode_image(image_path)

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "requests": [{
            "image": {"content": base64_image},
            "features": [{"type": "LANDMARK_DETECTION"}]
        }]
    }
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    landmarks = []
    landmark_annotations = data["responses"][0].get("landmarkAnnotations", [])
    for landmark in landmark_annotations:
        landmarks.append(landmark["description"])
    print(f"🏛️ Detected landmarks: {landmarks}")
    return landmarks


def get_similar_images(image_path, num_images=30):
    """Uploads an image and retrieves visually similar images."""
    print(f"📤 Uploading image: {image_path}")
    base64_image = encode_image(image_path)

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "requests": [{
            "image": {"content": base64_image},
            "features": [{"type": "WEB_DETECTION"}]
        }]
    }
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    similar_images = []
    web_detection = data["responses"][0].get("webDetection", {})
    if "visuallySimilarImages" in web_detection:
        for img in web_detection["visuallySimilarImages"][:num_images]:
            similar_images.append(img["url"])
    print(f"🔍 Found {len(similar_images)} visually similar images.")
    return similar_images


# Example Usage
uploaded_image = "images/test/2_people_together.jpg"
latitude, longitude = get_image_gps(uploaded_image)
landmarks = get_landmarks(uploaded_image)
similar_images = get_similar_images(uploaded_image)

if similar_images:
    print("✅ Similar images found:")
    filtered_images = []
    for idx, img_url in enumerate(similar_images, 1):
        similar_image_landmarks = get_landmarks(img_url)  # Extract landmarks for similar images
        similarity_reason = check_similarity((latitude, longitude), landmarks, None, similar_image_landmarks)
        print(f"{idx}. {img_url} - Similarity: {similarity_reason}")
        if similarity_reason:
            filtered_images.append(img_url)

    if filtered_images:
        download_images(filtered_images, uploaded_image)
else:
    print("❌ No similar images found.")
