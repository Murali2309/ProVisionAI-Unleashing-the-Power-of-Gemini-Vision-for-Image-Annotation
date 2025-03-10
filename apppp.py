import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
from PIL import Image
import io

# Replace with your Google Cloud API Key
API_KEY = "AIzaSyAohoQ459tBjxmEBWXyPw0cvDtbG_nylNE"  # Replace with your actual API key

# Streamlit app
st.set_page_config(page_title="ProVisionAI - Image Annotation", layout="wide")

# Custom CSS for a modern look
st.markdown(
    """
    <style>
    .stApp { background-color: #f0f2f6; }
    .stButton>button, .stFileUploader>div>div>div>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stMarkdown h1, .stMarkdown h2 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.title("✨ ProVisionAI: Unleashing the Power of Gemini Vision")
st.write("Upload images and get *AI-powered annotations* with rich insights using Google's Gemini Vision!")

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def get_annotations(encoded_image):
    """Sends the image to Google Vision API and returns annotations"""
    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
    payload = {
        "requests": [
            {
                "image": {"content": encoded_image},
                "features": [
                    {"type": "LABEL_DETECTION", "maxResults": 10},
                    {"type": "TEXT_DETECTION", "maxResults": 10},
                    {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                ],
            }
        ]
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def draw_bounding_boxes(image, objects):
    """Draws bounding boxes on detected objects"""
    img = np.array(image.convert("RGB"))
    for obj in objects:
        vertices = obj["boundingPoly"]["normalizedVertices"]
        x1, y1 = int(vertices[0]['x'] * img.shape[1]), int(vertices[0]['y'] * img.shape[0])
        x2, y2 = int(vertices[2]['x'] * img.shape[1]), int(vertices[2]['y'] * img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, obj["name"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

if uploaded_files:
    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # Show uploaded image
        st.subheader("📷 Uploaded Image")
        st.image(image, caption="Your Image", use_column_width=True)

        # API Call with Spinner
        with st.spinner("Processing image... Please wait ⏳"):
            annotations = get_annotations(encoded_image)

        # Error Handling
        if "error" in annotations:
            st.error(f"⚠ API Error: {annotations['error']}")
            continue

        # Display Results in Tabs
        tab1, tab2, tab3 = st.tabs(["🏷 Labels", "📝 Text", "📦 Objects"])

        # Labels (Objects/Concepts)
        with tab1:
            st.subheader("🏷 Detected Labels")
            if "responses" in annotations and "labelAnnotations" in annotations["responses"][0]:
                for label in annotations["responses"][0]["labelAnnotations"]:
                    st.write(f"- *{label['description']}* (Confidence: {label['score']:.2f})")
            else:
                st.write("No labels detected.")

        # Text Detection (OCR)
        with tab2:
            st.subheader("📝 Extracted Text")
            if "responses" in annotations and "textAnnotations" in annotations["responses"][0]:
                first_text = annotations["responses"][0]["textAnnotations"][0]["description"]
                st.markdown(f"*Extracted Text:* {first_text}")
            else:
                st.write("No text detected.")

        # Object Detection (Bounding Boxes)
        with tab3:
            st.subheader("📦 Detected Objects")
            if "responses" in annotations and "localizedObjectAnnotations" in annotations["responses"][0]:
                objects = annotations["responses"][0]["localizedObjectAnnotations"]
                for obj in objects:
                    st.write(f"- *{obj['name']}* (Confidence: {obj['score']:.2f})")
                
                # Draw Bounding Boxes
                img_with_boxes = draw_bounding_boxes(image, objects)
                st.image(img_with_boxes, caption="Detected Objects", use_column_width=True)
            else:
                st.write("No objects detected.")

        # Download JSON Button
        st.download_button("📥 Download JSON", json.dumps(annotations, indent=4), f"annotations_{uploaded_file.name}.json", "application/json")

# Add a footer
st.markdown(
    """
    ---
    *ProVisionAI* is powered by *Google Gemini Vision*. Built with ❤ for hackathons!
    """
)
