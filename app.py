import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from roboflow import Roboflow


st.set_page_config(
    page_title="Wall Crack Detection",
    page_icon="🧱",
    layout="wide",
)

st.title("🧱 Wall Crack Detection")

# Read the API key from Streamlit Secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error(
        "Roboflow API key not found. "
        "Please add ROBOFLOW_API_KEY in Streamlit Cloud Secrets."
    )
    st.stop()

# Connect to the Roboflow hosted model
rf = Roboflow(api_key=api_key)

project = rf.workspace().project(
    "wall-crack-detection-demo"
)

model = project.version(3).model

# Upload image
uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("Please upload a wall image to begin detection.")
    st.stop()

# Open and standardize the image
image = Image.open(uploaded).convert("RGB")

# Save temporarily because Roboflow predict() expects a file path
with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".jpg",
) as temp:
    image.save(temp.name)

    prediction = model.predict(
        temp.name,
        confidence=50,
        overlap=50,
    ).json()

# Convert image to a NumPy array for OpenCV drawing
image_np = np.array(image).copy()

# Draw bounding boxes
for pred in prediction.get("predictions", []):
    x = pred["x"]
    y = pred["y"]
    width = pred["width"]
    height = pred["height"]

    x1 = int(x - width / 2)
    y1 = int(y - height / 2)
    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    confidence = float(pred["confidence"])
    class_name = pred.get("class", "wall-crack")

    cv2.rectangle(
        image_np,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        image_np,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(
        image,
        use_container_width=True,
    )

with col2:
    st.subheader("Detection Result")
    st.image(
        image_np,
        use_container_width=True,
    )

# Detection summary
predictions = prediction.get("predictions", [])

st.subheader("Detection Summary")

if predictions:
    st.metric(
        "Detected Crack Regions",
        len(predictions),
    )

    confidences = [
        float(pred["confidence"])
        for pred in predictions
    ]

    st.metric(
        "Highest Confidence",
        f"{max(confidences):.2f}",
    )
else:
    st.warning("No wall cracks were detected.")

# Optional raw API output
with st.expander("View Roboflow API response"):
    st.json(prediction)