import requests
import os
from PIL import Image
from io import BytesIO


def get_api_key(filename="../../keys/streetview.txt"):
    """Reads the API key from a text file."""
    try:
        with open(filename, "r") as file:
            api_key = file.read().strip()
            if api_key:
                print(f"[DEBUG] Successfully read API key from {filename}")
                return api_key
            else:
                print("[ERROR] API key file is empty!")
                return None
    except Exception as e:
        print(f"[ERROR] Unable to read API key file: {e}")
        return None


# Read API Key
GOOGLE_API_KEY = get_api_key()

if not GOOGLE_API_KEY:
    print("[ERROR] No valid API key found. Exiting program.")
    exit(1)

# Define location (latitude and longitude)
latitude = 1.286789  # Example: Merlion Park, Singapore
longitude = 103.854535

# Optimized settings
size = "640x640"
fov = 80
pitch_values = [-20, -10, 0, 10, 20]
heading_values = list(range(0, 360, 10))


def is_street_view_available(lat, lng, api_key):
    """Check if a given GPS coordinate has Google Street View coverage."""
    metadata_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&key={api_key}"
    response = requests.get(metadata_url)
    data = response.json()

    if data["status"] == "OK":
        print(f"[INFO] Street View available at {lat}, {lng}")
        return True
    else:
        print(f"[WARNING] No Street View available at {lat}, {lng} (Status: {data['status']})")
        return False


def get_location_name(lat, lng, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            location_name = data["results"][0]["formatted_address"]
            location_name = location_name.replace(",", "").replace(" ", "_")
            print(f"[DEBUG] Fetched location name: {location_name}")
            return location_name
        else:
            print(f"[ERROR] Geocoding API error: {data['status']}")
    else:
        print(f"[ERROR] HTTP error {response.status_code} while fetching location name.")

    return "Unknown_Location"


def fetch_street_view_images(lat, lng, api_key):
    if not is_street_view_available(lat, lng, api_key):
        print(f"[SKIP] Skipping {lat}, {lng} as no Street View is available.")
        return

    location_name = get_location_name(lat, lng, api_key)
    save_path = f"images/google_street_views/{location_name}/"
    os.makedirs(save_path, exist_ok=True)

    for pitch in pitch_values:
        for heading in heading_values:
            url = f"https://maps.googleapis.com/maps/api/streetview?size={size}&location={lat},{lng}&heading={heading}&pitch={pitch}&fov={fov}&key={api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))  # Using Pillow
                filename = f"{location_name}_pitch{pitch}_heading{heading}.jpg"
                image.save(os.path.join(save_path, filename), "JPEG")  # Saving image using Pillow
                print(f"[INFO] Saved image: {save_path}{filename}")
            else:
                print(
                    f"[ERROR] Failed to fetch image at heading {heading}, pitch {pitch} (HTTP {response.status_code})")


# Run function
fetch_street_view_images(latitude, longitude, GOOGLE_API_KEY)
