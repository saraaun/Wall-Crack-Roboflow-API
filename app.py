# Corrected app.py for WORKFLOW_ID = "custom-workflow" or "Custom Workflow" of Model V3
"""
What changed from the previous version
- Removed the cv2 (opencv-python-headless) dependency entirely.
  Bounding boxes and labels are now drawn with PIL.ImageDraw
  instead of cv2. This avoids the recurring `ImportError:
  libGL.so.1` failure that opencv-python-headless can trigger
  on Streamlit Cloud when its base image is rebuilt, and it
  removes a fairly heavy dependency for something we only used
  to draw rectangles and text.
- Added packages.txt (libgl1, libglib2.0-0) as a belt-and-suspenders
  safety net in case any other dependency ever needs those system
  libraries again.
- Everything else — the recursive detection parser, base64 image
  encoding, sidebar controls, summary metrics, raw output viewer —
  is unchanged.
"""
import base64
from io import BytesIO
from typing import Any

import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont, ImageOps


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
    "using a YOLO11 model hosted through a Roboflow Workflow."
)


# ---------------------------------------------------------
# Roboflow configuration
# ---------------------------------------------------------

WORKSPACE_NAME = "sarawans-workspace"

# IMPORTANT:
# Copy the exact Workflow ID from:
# Roboflow Workflow → Deploy → generated Python code
WORKFLOW_ID = "custom-workflow"


def get_api_key() -> str:
    """Read the Roboflow API key from Streamlit Secrets."""

    try:
        return st.secrets["ROBOFLOW_API_KEY"]
    except KeyError:
        st.error(
            "The Roboflow API key was not found.\n\n"
            "Add this entry in Streamlit Cloud Secrets:\n\n"
            '`ROBOFLOW_API_KEY = "YOUR_API_KEY"`'
        )
        st.stop()


@st.cache_resource
def create_roboflow_client(
    api_key: str,
) -> InferenceHTTPClient:
    """Create and cache the Roboflow API client."""

    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )


# ---------------------------------------------------------
# Workflow-output parsing
# ---------------------------------------------------------

def is_detection(item: Any) -> bool:
    """Check whether a dictionary resembles a detection."""

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


def recursively_find_detections(
    data: Any,
) -> list[dict]:
    """
    Recursively search nested Workflow output for detections.

    This is used as a fallback because Workflow results can have
    different nesting depending on the configured output block.
    """

    detections: list[dict] = []

    if isinstance(data, list):
        for item in data:
            if is_detection(item):
                detections.append(item)
            else:
                detections.extend(
                    recursively_find_detections(item)
                )

    elif isinstance(data, dict):
        if is_detection(data):
            detections.append(data)
        else:
            for value in data.values():
                detections.extend(
                    recursively_find_detections(value)
                )

    return detections


def extract_workflow_predictions(
    workflow_result: Any,
) -> list[dict]:
    """
    Extract detections from the Workflow output named 'model_output'.

    Expected possibilities include:

    [
        {
            "model_output": {
                "predictions": [...]
            }
        }
    ]

    or:

    [
        {
            "model_output": [...]
        }
    ]
    """

    if not workflow_result:
        return []

    # run_workflow often returns one result item per image.
    if isinstance(workflow_result, list):
        if len(workflow_result) == 0:
            return []

        result_item = workflow_result[0]
    else:
        result_item = workflow_result

    if not isinstance(result_item, dict):
        return recursively_find_detections(workflow_result)

    model_output = result_item.get("model_output")

    # Case 1:
    # {"model_output": {"predictions": [...]}}
    if isinstance(model_output, dict):
        predictions = model_output.get(
            "predictions",
            model_output.get(
                "detections",
                model_output.get("results", []),
            ),
        )

        if isinstance(predictions, list):
            extracted = [
                item
                for item in predictions
                if isinstance(item, dict)
            ]

            if extracted:
                return extracted

    # Case 2:
    # {"model_output": [...]}
    elif isinstance(model_output, list):
        extracted = [
            item
            for item in model_output
            if isinstance(item, dict)
        ]

        if extracted:
            return extracted

    # Fallback: search the complete output recursively.
    return recursively_find_detections(workflow_result)


# ---------------------------------------------------------
# Drawing function (PIL-only, no cv2 / no system libGL needed)
# ---------------------------------------------------------

def get_label_font():
    """Load a reasonably legible font, falling back to PIL's default."""

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[dict],
) -> Image.Image:
    """Draw bounding boxes and labels on an RGB PIL image."""

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_label_font()

    image_width, image_height = annotated.size

    box_color = (255, 0, 0)
    text_color = (255, 255, 255)
    box_thickness = 3

    for detection in detections:
        x = float(detection["x"])
        y = float(detection["y"])
        width = float(detection["width"])
        height = float(detection["height"])
        confidence = float(detection["confidence"])

        class_name = detection.get(
            "class",
            detection.get(
                "class_name",
                "wall-crack",
            ),
        )

        # Roboflow detection coordinates are centre-based.
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        # Keep coordinates inside the image.
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=box_color,
            width=box_thickness,
        )

        label = f"{class_name} {confidence:.2f}"

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding = 5
        label_top = max(0, y1 - text_height - 2 * padding)
        label_bottom = label_top + text_height + 2 * padding
        label_right = min(
            x1 + text_width + 2 * padding,
            image_width - 1,
        )

        draw.rectangle(
            [(x1, label_top), (label_right, label_bottom)],
            fill=box_color,
        )

        draw.text(
            (x1 + padding, label_top + padding - text_bbox[1]),
            label,
            fill=text_color,
            font=font,
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
    step=0.01,
    help=(
        "This filters predictions after they are returned "
        "by the Workflow. It does not change the confidence "
        "setting inside Roboflow."
    ),
)

show_raw_output = st.sidebar.checkbox(
    "Show raw Workflow output",
    value=True,
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
    original_image = Image.open(uploaded_file)

    # Respect EXIF orientation, then convert to RGB.
    original_image = ImageOps.exif_transpose(
        original_image
    ).convert("RGB")

except Exception as error:
    st.error(
        "The uploaded file could not be opened as an image."
    )
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Encode image in memory and run Workflow
# ---------------------------------------------------------
#
# NOTE: We base64-encode the image ourselves and pass the
# encoded string directly to run_workflow(). Passing a local
# file path can cause the SDK to serialize the image as a raw
# NumPy payload under the hood, which Roboflow's hosted
# serverless Workflow endpoint rejects with a 400 error
# ("NumPy image type is not supported in this configuration
# of inference"). Base64 avoids that entirely and also means
# we don't need any temp files on disk.

try:
    buffered = BytesIO()
    original_image.save(
        buffered,
        format="JPEG",
        quality=100,
        subsampling=0,
    )
    image_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")

    with st.spinner(
        "Analysing the image with Roboflow..."
    ):
        workflow_result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={
                "image": image_b64,
            },
            # Disable cache while debugging a new Workflow.
            use_cache=False,
        )

except Exception as error:
    st.error("Roboflow Workflow inference failed.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Extract and filter detections
# ---------------------------------------------------------

all_detections = extract_workflow_predictions(
    workflow_result
)

detections = [
    detection
    for detection in all_detections
    if float(
        detection.get("confidence", 0.0)
    ) >= minimum_confidence
]


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------

st.caption(
    f"Workflow returned {len(all_detections)} raw "
    f"detection(s); {len(detections)} remain after "
    f"the displayed threshold."
)


# ---------------------------------------------------------
# Draw result
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
        "displayed confidence threshold."
    )


# ---------------------------------------------------------
# Raw Workflow result
# ---------------------------------------------------------

if show_raw_output:
    with st.expander(
        "Raw Roboflow Workflow output",
        expanded=True,
    ):
        st.json(workflow_result)
