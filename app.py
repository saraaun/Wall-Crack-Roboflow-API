import streamlit as st
from roboflow import Roboflow
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Wall Crack Detection",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Wall Crack Detection")

api_key = st.secrets["ROBOFLOW_API_KEY"]

rf = Roboflow(api_key=api_key)

project = rf.workspace().project(
    "wall-crack-detection-demo"
)

model = project.version(3).model

# Upload image
uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded is not None:

    image = Image.open(uploaded)

    st.image(image)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp:

        image.save(temp.name)

        prediction = model.predict(
            temp.name,
            confidence=50,
            overlap=50
        ).json()

    st.json(prediction)

# Draw the boxes
import cv2
import numpy as np

image_np = np.array(image)

for pred in prediction["predictions"]:

    x = pred["x"]
    y = pred["y"]

    w = pred["width"]
    h = pred["height"]

    x1 = int(x-w/2)
    y1 = int(y-h/2)
    x2 = int(x+w/2)
    y2 = int(y+h/2)

    cv2.rectangle(
        image_np,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )

    cv2.putText(
        image_np,
        f"{pred['confidence']:.2f}",
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

st.image(image_np)