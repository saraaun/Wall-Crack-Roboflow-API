import os
import tempfile
from typing import Any

import cv2
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Wall Crack Detection",
    page_icon="🧱",
    layout="wide",
)

st.title("🧱 Wall Crack Detection")

st.write(
    "Upload an image of a wall to detect visible cracks "
    "using a YOLO11 model hosted through Roboflow."
)


# ---------------------------------------------------------
# Roboflow configuration
# ---------------------------------------------------------

WORKSPACE_NAME = "sarawans-workspace"

# WORKFLOW_ID = (
#     "wall-crack-detection-demo-"
#     "vwall-crack-detection-demo-6-yolo11n-t1-logic"
# )

# WORKFLOW_ID = (
#     "wall-crack-detection-demo-"
#     "vwall-crack-detection-demo-6-yolo11n-t1-logic"
# )

WORKFLOW_ID = (
    "wall-crack-detection-demo-"
    "vwall-crack-detection-demo-6-yolo11n-t1-logic"
)

# WORKFLOW_ID = "custom-workflow"

def get_api_key() -> str:
    """Read the Roboflow API key from Streamlit Secrets."""

    try:
        return st.secrets["ROBOFLOW_API_KEY"]
    except KeyError:
        st.error(
            "The Roboflow API key was not found.\n\n"
            "Add the following entry in Streamlit Cloud Secrets:\n\n"
            '`ROBOFLOW_API_KEY = "YOUR_NEW_API_KEY"`'
        )
        st.stop()


@st.cache_resource
def create_roboflow_client(api_key: str) -> InferenceHTTPClient:
    """Create and cache the Roboflow API client."""

    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )


# ---------------------------------------------------------
# Workflow-result parsing
# ---------------------------------------------------------

def is_detection(item: Any) -> bool:
    """
    Check whether a dictionary looks like an object-detection result.
    """

    if not isinstance(item, dict):
        return False

    required_keys = {
        "x",
        "y",
        "width",
        "height",
        "confidence",
    }

    return required_keys.issubset(item.keys())


def find_detections(data: Any) -> list[dict]:
    """
    Recursively search a Roboflow Workflow response for detection objects.

    Workflow outputs can be nested differently depending on the output
    blocks configured in Roboflow. This function searches dictionaries
    and lists until it finds objects containing bounding-box fields.
    """

    detections: list[dict] = []

    if isinstance(data, list):
        for item in data:
            if is_detection(item):
                detections.append(item)
            else:
                detections.extend(find_detections(item))

    elif isinstance(data, dict):
        if is_detection(data):
            detections.append(data)
        else:
            for value in data.values():
                detections.extend(find_detections(value))

    return detections


# ---------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------

def draw_detections(
    image: Image.Image,
    detections: list[dict],
) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels on an RGB image.
    """

    annotated = np.array(image).copy()

    image_height, image_width = annotated.shape[:2]

    for detection in detections:
        x = float(detection["x"])
        y = float(detection["y"])
        width = float(detection["width"])
        height = float(detection["height"])
        confidence = float(detection["confidence"])

        class_name = detection.get(
            "class",
            detection.get("class_name", "wall-crack"),
        )

        # Roboflow returns centre-based bounding boxes.
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        # Keep box coordinates inside the image.
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3,
        )

        label = f"{class_name} {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        text_size, baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        text_width, text_height = text_size

        label_y1 = max(0, y1 - text_height - baseline - 10)
        label_y2 = max(text_height + baseline + 10, y1)

        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (
                min(x1 + text_width + 10, image_width - 1),
                label_y2,
            ),
            (255, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 5, max(text_height + 2, y1 - 7)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------
# Initialise Roboflow client
# ---------------------------------------------------------

api_key = get_api_key()
client = create_roboflow_client(api_key)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Detection Settings")

minimum_confidence = st.sidebar.slider(
    "Displayed Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    help=(
        "This filters results returned by the Workflow. "
        "It does not change the confidence setting inside "
        "the Roboflow Workflow itself."
    ),
)

show_raw_output = st.sidebar.checkbox(
    "Show raw Workflow output",
    value=False,
)


# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a wall image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Please upload an image to begin detection.")
    st.stop()


# ---------------------------------------------------------
# Read uploaded image
# ---------------------------------------------------------

try:
    original_image = Image.open(uploaded_file).convert("RGB")
except Exception as error:
    st.error("The uploaded file could not be opened as an image.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Save temporary image and run Roboflow Workflow
# ---------------------------------------------------------

temporary_path = None

try:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg",
    ) as temporary_file:
        original_image.save(
            temporary_file.name,
            format="JPEG",
            quality=95,
        )
        temporary_path = temporary_file.name

    with st.spinner("Analysing the image with Roboflow..."):
        workflow_result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={
                "image": temporary_path,
            },
            use_cache=True,
        )

except Exception as error:
    st.error("Roboflow Workflow inference failed.")
    st.exception(error)
    st.stop()

finally:
    if temporary_path and os.path.exists(temporary_path):
        os.remove(temporary_path)


# ---------------------------------------------------------
# Extract and filter detections
# ---------------------------------------------------------

all_detections = find_detections(workflow_result)

detections = [
    detection
    for detection in all_detections
    if float(detection.get("confidence", 0.0))
    >= minimum_confidence
]


# ---------------------------------------------------------
# Draw results
# ---------------------------------------------------------

annotated_image = draw_detections(
    original_image,
    detections,
)


# ---------------------------------------------------------
# Display images
# ---------------------------------------------------------

column_original, column_result = st.columns(2)

with column_original:
    st.subheader("Original Image")
    st.image(
        original_image,
        use_container_width=True,
    )

with column_result:
    st.subheader("Detection Result")
    st.image(
        annotated_image,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Detection summary
# ---------------------------------------------------------

st.divider()
st.subheader("Detection Summary")

if detections:
    confidence_values = [
        float(detection["confidence"])
        for detection in detections
    ]

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Detected Crack Regions",
            len(detections),
        )

    with metric2:
        st.metric(
            "Highest Confidence",
            f"{max(confidence_values):.2f}",
        )

    with metric3:
        st.metric(
            "Average Confidence",
            f"{np.mean(confidence_values):.2f}",
        )

    detection_rows = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        detection_rows.append(
            {
                "Crack ID": index,
                "Class": detection.get(
                    "class",
                    detection.get(
                        "class_name",
                        "wall-crack",
                    ),
                ),
                "Confidence": round(
                    float(detection["confidence"]),
                    3,
                ),
                "Centre X": round(
                    float(detection["x"]),
                    1,
                ),
                "Centre Y": round(
                    float(detection["y"]),
                    1,
                ),
                "Width": round(
                    float(detection["width"]),
                    1,
                ),
                "Height": round(
                    float(detection["height"]),
                    1,
                ),
            }
        )

    st.dataframe(
        detection_rows,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.warning(
        "No wall crack was detected above the selected "
        "confidence threshold."
    )


# ---------------------------------------------------------
# Optional debugging output
# ---------------------------------------------------------

if show_raw_output:
    with st.expander(
        "Raw Roboflow Workflow output",
        expanded=True,
    ):
        st.json(workflow_result)