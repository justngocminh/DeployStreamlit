import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Function to perform OCR and annotate the image
def perform_ocr(image):
    # Convert the uploaded image to RGB (Pillow)
    image_pil = Image.open(image).convert('RGB')
    
    # Convert Pillow image to NumPy array (RGB format)
    img_rgb = np.array(image_pil)
    
    # Convert RGB image to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Perform OCR using EasyOCR
    result = reader.readtext(img_bgr)

    # Create an ImageDraw object to annotate the image
    draw = ImageDraw.Draw(image_pil)

    # Load font
    font_size = 20  # You can change the font size if needed
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Draw bounding boxes and text on the image
    for detection in result:
        bbox = detection[0]
        text = detection[1]
        top_left = (bbox[0][0], bbox[0][1])
        bottom_right = (bbox[2][0], bbox[2][1])

        # Draw bounding box
        draw.rectangle([top_left, bottom_right], outline="green", width=2)
        
        # Draw text
        draw.text((top_left[0], top_left[1] - font_size), text, font=font, fill="green")

    # Convert the annotated Pillow image back to NumPy array (RGB format)
    img_rgb_with_text = np.array(image_pil)
    
    return img_rgb_with_text, result

# Main function
def main():
    # Set title and sidebar
    st.title("Nhận diện văn bản Tiếng Việt")
    st.sidebar.title("Lựa chọn")

    # Upload image
    uploaded_image = st.sidebar.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

    # Perform OCR when button is clicked
    if st.sidebar.button("Thực hiện nhận diện"):
        if uploaded_image is not None:
            # Perform OCR and annotate the image
            result_image, ocr_result = perform_ocr(uploaded_image)
            
            # Display the annotated image
            st.image(result_image, caption="Ảnh kết quả", use_column_width=True)

            # Display the OCR results
            st.markdown("### Nội dung")
            for detection in ocr_result:
                bbox = detection[0]
                text = detection[1]
                confidence = detection[2]
                st.write(f"**Văn bản:** {text}")
                st.write("---")
        else:
            st.sidebar.warning("Hãy tải ảnh lên trước!")

# Run the main function
if __name__ == "__main__":
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en', 'vi'], gpu=False)
    FONT_PATH = 'arial.ttf'
    
    # Run the app
    main()
