# 🧱 Wall Crack Detection using Roboflow Hosted API and Streamlit

A lightweight web application for detecting wall cracks using a **YOLOv11 object detection model** trained with **Roboflow** and deployed through the **Roboflow Hosted Inference API**.

Unlike the local deployment version that loads `best.pt`, this application sends uploaded images to the Roboflow cloud for inference and displays the detection results in a user-friendly Streamlit interface.

---

## 🌐 Live Demo

> **Coming Soon**

---

## Features

- 🧱 Detect wall cracks from uploaded images
- ☁️ Cloud inference using Roboflow Hosted API
- 🎯 Adjustable confidence threshold
- 📊 Detection summary with confidence scores
- 🖼️ Side-by-side original and annotated images
- ⚡ Lightweight deployment (no PyTorch model required)
- 🚀 Ready for deployment on Streamlit Community Cloud

---

## Model Information

**Task:** Object Detection

**Framework:** YOLOv11 Nano

**Dataset:** Wall Crack Detection

**Classes**

| ID | Class |
|----|-------|
| 0 | wall-crack |

---

## Roboflow Dataset Preprocessing

The hosted model uses the exact preprocessing configured in Roboflow.

- ✅ Auto-Orient
- ✅ Resize (Stretch to 640 × 640)
- ✅ Contrast Stretching

Since inference is performed directly on Roboflow's servers, the prediction results closely match those shown in the Roboflow web interface.

---

## Project Structure

```
Wall-Crack-Roboflow-API/
│
├── app.py
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── secrets.toml
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your_username>/<repository_name>.git
```

Move into the project

```bash
cd <repository_name>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Roboflow API Key

Create the following file

```
.streamlit/secrets.toml
```

Add your Roboflow API key

```toml
ROBOFLOW_API_KEY="YOUR_API_KEY"
```

> **Never commit your API key to GitHub.**

When deploying to Streamlit Community Cloud, add the same key under:

**App Settings → Secrets**

---

## Run Locally

```bash
streamlit run app.py
```

---

## Deployment

This project is designed for deployment on **Streamlit Community Cloud**.

Deployment steps:

1. Push the repository to GitHub.
2. Log in to Streamlit Community Cloud.
3. Create a new application.
4. Select this repository.
5. Set `app.py` as the entry point.
6. Add your Roboflow API key under **Secrets**.
7. Deploy.

---

## Example Workflow

```
Upload Image
        │
        ▼
Streamlit Web App
        │
        ▼
Roboflow Hosted API
        │
        ▼
YOLOv11 Model
        │
        ▼
Prediction JSON
        │
        ▼
Display Detection Result
```

---

## Technologies Used

- Python
- Streamlit
- Roboflow Python SDK
- OpenCV
- Pillow
- NumPy
- Pandas

---

## Why Use the Hosted API?

Compared with deploying a local YOLO model (`best.pt`), the Hosted API provides several advantages:

- No model file required
- Smaller GitHub repository
- Faster deployment
- Lower memory usage
- Predictions match Roboflow's web interface
- No need to install PyTorch or Ultralytics

This approach is ideal for rapid prototyping, demonstrations, and educational purposes.

---

## Future Improvements

- Batch image inference
- Webcam support
- Video crack detection
- Detection history
- Download annotated images
- Crack severity estimation
- Crack length measurement
- Crack area estimation
- Mobile-friendly interface

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This project was built using:

- Roboflow
- Streamlit
- OpenCV
- NumPy
- Pillow

Special thanks to the Roboflow team for providing an excellent platform for dataset management, model training, and hosted inference.