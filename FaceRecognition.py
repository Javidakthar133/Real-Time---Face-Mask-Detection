import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from PIL import Image, ExifTags
from tensorflow.keras.models import load_model

#Stresmlit Part

st.set_page_config(page_title="Face Mask Detection", page_icon=r"C:\Users\Javid Akthar\Desktop\DeepLearning_FaceMask\Facemask.png")

# Set background image URL (you can replace this with your image URL)
background_image_url = r"C:\Users\Javid Akthar\Desktop\DeepLearning_FaceMask\face-mask-detection-system.png"  

# Custom CSS to set background image for the entire page
st.markdown(f"""
    <style>
        body {{
            background-image: url("{background_image_url}");
            background-size: cover;  /* Ensures the image covers the entire page */
            background-repeat: no-repeat;  /* Ensures the image does not repeat */
            background-attachment: fixed;  /* Makes the background fixed when scrolling */
            color: white;  /* Text color set to white for contrast against background */
        }}
        .stApp {{
            background-color: transparent;
        }}
        .css-1v0mbp5 {{
            background-color: rgba(255, 255, 255, 0.7);  /* Make the option menu background semi-transparent */
            border-radius: 10px;  /* Optional: round the corners of the menu */
        }}
    </style>
""", unsafe_allow_html=True)

# Title for the app
st.markdown("<h1 style='text-align:center;text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); color:#8bb933; font-size:50px;'>Real Time -Face Mask Detection</h1>", unsafe_allow_html=True)

# Create the option menu

#option menu

selected = option_menu(
    menu_title=None,
    options=["Image","Contact"],
    icons=["image","at"],
    default_index=0,
    orientation="horizontal"
)

if selected =='Image':
    
    # Upload the image

    img_file = st.file_uploader('Upload the Image', type=['png', 'jpg', 'jpeg'])

    if img_file is not None:
        # Open image using PIL
        img = Image.open(img_file)

        # Fix image orientation using EXIF
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(img._getexif().items())

            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError, TypeError):
            pass

        # Show the correctly oriented image
        st.image(img, width=300)

        # Reset file pointer to beginning for OpenCV reading
        img_file.seek(0)

        # Convert image for OpenCV
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)  # Decode as BGR

        # Resize and normalize image for model input
        resized_img = cv2.resize(opencv_image, (128, 128)).astype('float32') / 255.0
        input_data = np.expand_dims(resized_img, axis=0)

        # Load your trained Keras model
        model_1 = load_model(r'C:\Users\Javid Akthar\Desktop\DeepLearning_FaceMask\model.h5')

        # Perform prediction
        prediction = model_1.predict(input_data)

        # Display the result
        if prediction[0][0] > 0.8:
            st.markdown(
                """
                <h2 style="
                    color:#8bb933;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                    font-family: Arial, sans-serif;
                ">
                    Mask Detected
                </h2>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <h2 style="
                    color:#f95835;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                    font-family: Arial, sans-serif;
                ">
                    No Mask Detected
                </h2>
                """, 
                unsafe_allow_html=True
            )
                        

    

if selected == 'Contact':
    # Data
    name = "Javid Akthar VST"
    mail = "javedakthar133@gmail.com"
    description = "An Aspiring DATA-SCIENTIST..!"
    social_media = {
        "GITHUB": "https://github.com/Javidakthar133",
        "LINKEDIN": "www.linkedin.com/in/javid-akthar-928295276"
    }

    # Using the second column to display title, description, and email
    st.markdown("<h1 style='color:#8bb933; font-size:50px;text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);'>Real Time - Face Mask Detection Using Deep Learning</h1>", unsafe_allow_html=True)
    st.write("""
        In recent trend in world wide Lockdowns due to COVID19 outbreak, as Face Mask is became mandatory for everyone while roaming outside, approach of Deep Learning for Detecting Faces With and Without mask were a good trendy practice. Here I have created a model that detects face mask trained on 7553 images with 3 color channels (RGB).
        On Custom CNN architecture Model training accuracy reached 94% and Validation accuracy 96%.
    """, unsafe_allow_html=True)
    
    # Divider line
    st.write("---")
        
    st.markdown(f"<h3 style='color:#8bb933;text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);'>Mail: <a href='mailto:{mail}' style='color:white;'>{mail}</a></h3>", unsafe_allow_html=True)

    # Space between sections
    st.write("#")

    # Display social media links with green color
    cols = st.columns(len(social_media))
    for index, (platform, link) in enumerate(social_media.items()):
        with cols[index]:
            st.markdown(f'<a href="{link}" target="_blank" style="color:white;text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5); font-size: 20px;">{platform}</a>', unsafe_allow_html=True)