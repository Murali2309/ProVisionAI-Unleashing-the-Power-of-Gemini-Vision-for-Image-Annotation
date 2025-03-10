import streamlit as st
import requests
import base64
import cv2
import numpy as np
from PIL import Image
import io

# Replace with your Google Cloud API Key
API_KEY = "YOUR_GOOGLE_CLOUD_API_KEY"

# Custom CSS for a modern look
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    .stMarkdown h2 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.title("✨ ProVisionAI: Unleashing the Power of Gemini Vision")
st.write(
    "Upload an image and get **AI-powered annotations** with rich insights using Google's Gemini Vision!"
)

# Function to annotate an image with a bounding box and text
def annotate_image(image, box, text=None, font_scale=1, color=(0, 255, 0), thickness=2):
    """
    Annotates an image with a bounding box and optional text.
    
    Args:
        image: PIL Image object.
        box: Tuple of (x, y, width, height) for the bounding box.
        text: Optional text to display near the bounding box.
        font_scale: Font scale for the text.
        color: Color of the bounding box and text (BGR format).
        thickness: Thickness of the bounding box and text.
    
    Returns:
        Annotated PIL Image object.
    """
    # Convert PIL image to OpenCV format
    image_cv = np.array(image.convert("RGB"))
    
    # Unpack the bounding box coordinates and dimensions
    x, y, w, h = box

    # Draw a rectangle (bounding box) on the image
    cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, thickness)

    # If annotation text is provided, add it near the bounding box
    if text:
        # Position the text slightly above the top-left corner, adjusting if near the top edge
        text_y = y - 10 if y - 10 > 10 else y + 20
        cv2.putText(image_cv, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return Image.fromarray(image_cv)

# Function to get annotations from Google Vision API
def get_annotations(encoded_image):
    """
    Sends the image to Google Vision API and returns annotations.
    
    Args:
        encoded_image: Base64-encoded image content.
    
    Returns:
        JSON response from the Vision API or an error message.
    """
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

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("📷 Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Image", use_column_width=True)

    # Convert the image to base64
    image_bytes = uploaded_file.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # API Call with Spinner
    with st.spinner("Processing image... Please wait ⏳"):
        annotations = get_annotations(encoded_image)

    # Error Handling
    if "error" in annotations:
        st.error(f"⚠ API Error: {annotations['error']}")
    else:
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

                # Draw bounding boxes on the image
                for obj in objects:
                    vertices = obj["boundingPoly"]["normalizedVertices"]
                    x1 = int(vertices[0]['x'] * image.width)
                    y1 = int(vertices[0]['y'] * image.height)
                    x2 = int(vertices[2]['x'] * image.width)
                    y2 = int(vertices[2]['y'] * image.height)
                    image = annotate_image(image, (x1, y1, x2 - x1, y2 - y1), text=obj["name"])

                # Display the annotated image
                st.image(image, caption="Detected Objects", use_column_width=True)
            else:
                st.write("No objects detected.")

# Add a footer
st.markdown(
    """
    ---
    **ProVisionAI** is powered by **Google Gemini Vision**. Built with ❤️ for hackathons!
    """
)