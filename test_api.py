import requests

# ------------------------------
# Replace with your Render URL
# ------------------------------
url = "https://cat-dog-app-slim-latest.onrender.com/predict"

# ------------------------------
# Replace with your local image path
# Use raw string (r"") to avoid Windows backslash issues
# ------------------------------
image_path = r"C:\Users\admin\Downloads\dataSet\PetImages\Dog\998.jpg"

# ------------------------------
# Send POST request with the image
# ------------------------------
try:
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    # Check if request was successful
    if response.status_code == 200:
        print("✅ Prediction Result:")
        print(response.json())
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

except FileNotFoundError:
    print(f"❌ Image not found at {image_path}")
except Exception as e:
    print(f"❌ An error occurred: {str(e)}")
