import tempfile

import cv2
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image


st.set_page_config(
    page_title="Wall Crack Detection",
    page_icon="🧱",
    layout="wide",
)

st.title("🧱 Wall Crack Detection")

# Read API key securely
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error(
        "Roboflow API key not found. "
        "Please add ROBOFLOW_API_KEY in Streamlit Cloud Secrets."
    )
    st.stop()

# Roboflow Serverless Hosted API client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key,
)

# Roboflow project ID and version
MODEL_ID = "wall-crack-detection-demo/3"

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("Please upload a wall image to begin detection.")
    st.stop()

image = Image.open(uploaded).convert("RGB")

# Save uploaded image temporarily
with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".jpg",
) as temp:
    image.save(temp.name)
    temp_path = temp.name

# Run hosted inference
try:
    prediction = client.infer(
        temp_path,
        model_id=MODEL_ID,
    )
except Exception as error:
    st.error("Roboflow inference failed.")
    st.exception(error)
    st.stop()

image_np = np.array(image).copy()

predictions = prediction.get("predictions", [])

for pred in predictions:
    x = float(pred["x"])
    y = float(pred["y"])
    width = float(pred["width"])
    height = float(pred["height"])

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

st.subheader("Detection Summary")

if predictions:
    confidences = [
        float(pred["confidence"])
        for pred in predictions
    ]

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Detected Crack Regions",
            len(predictions),
        )

    with metric2:
        st.metric(
            "Highest Confidence",
            f"{max(confidences):.2f}",
        )
else:
    st.warning("No wall cracks were detected.")

with st.expander("View Roboflow API response"):
    st.json(prediction)