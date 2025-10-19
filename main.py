import streamlit as st
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image
import zipfile


def detect_chromosomes(img_array):
    # Default parameters
    thresh_value = 0  # Use adaptive thresholding
    min_area = 100  # Minimum area for contour filtering
    padding = 5  # Padding around chromosomes

    # Convert to grayscale if not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding for better results on varying lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Alternatively, use fixed threshold if selected (not used here with thresh_value=0)
    if thresh_value > 0:
        _, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to close gaps and remove noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chromosomes = []
    bboxes = []
    for i, contour in enumerate(contours):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by area
        area = cv2.contourArea(contour)
        if area > min_area:
            # Extract the chromosome with padding
            pad_y_start = max(0, y - padding)
            pad_y_end = y + h + padding
            pad_x_start = max(0, x - padding)
            pad_x_end = x + w + padding
            chromosome = gray[pad_y_start:pad_y_end, pad_x_start:pad_x_end]
            chromosomes.append((f'chromosome_{i}.png', chromosome))
            bboxes.append((x - padding, y - padding, w + 2 * padding, h + 2 * padding))  # Adjusted for padding

    return chromosomes, bboxes, len(chromosomes)


# Streamlit app
st.title("Advanced Chromosome Detection System")

uploaded_file = st.file_uploader("Upload an image of chromosomes", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    img_pil = Image.open(uploaded_file)
    img_array = np.array(img_pil)

    st.image(img_array, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Chromosomes"):
        try:
            chromosomes, bboxes, count = detect_chromosomes(img_array)

            if chromosomes:
                st.success(f"Detected {count} potential chromosomes.")

                # Display original image with bounding boxes
                img_with_boxes = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR) if len(
                    img_array.shape) == 2 else img_array.copy()
                for bbox in bboxes:
                    x, y, w, h = bbox
                    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                st.image(img_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)

                # Display individual chromosomes
                for name, chrom in chromosomes:
                    chrom_pil = Image.fromarray(chrom)
                    st.image(chrom_pil, caption=name, width=200)

                    # Individual download
                    buf = BytesIO()
                    chrom_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label=f"Download {name}",
                        data=byte_im,
                        file_name=name,
                        mime="image/png"
                    )

                # Download all as ZIP
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zip_file:
                    for name, chrom in chromosomes:
                        chrom_pil = Image.fromarray(chrom)
                        img_buf = BytesIO()
                        chrom_pil.save(img_buf, format="PNG")
                        zip_file.writestr(name, img_buf.getvalue())
                zip_buf.seek(0)
                st.download_button(
                    label="Download All as ZIP",
                    data=zip_buf,
                    file_name="chromosomes.zip",
                    mime="application/zip"
                )
            else:
                st.warning("No chromosomes detected. Try a different image or adjust preprocessing manually if needed.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image to start.")