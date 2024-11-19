import streamlit as st
import os
from PIL import Image

st.set_page_config(layout="wide")
st.title('Sign Language Detection')

# Tabs for functionality
tab1, tab2 = st.tabs(["Detection", "Live Detection"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
      
        file_path = f"prediction/uploads/{uploaded_file.name}"
        os.makedirs("prediction/uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)

        with col1:
            st.header("Uploaded Image")
            st.image(file_path, width=500) 

        with col2:
            st.header("Detection Results")
            with st.spinner("Processing the image..."):
                detected_dir = "prediction"
                os.makedirs(detected_dir, exist_ok=True)  

                detection_command = (
                    f"cd ./yolov5 && "
                    f"python detect.py "
                    f"--weights ./runs/train/yolov5s_results/weights/best.pt "
                    f"--img 416 --conf 0.4 "
                    f"--source ../{file_path} "
                    f"--project ../{detected_dir} --name results --exist-ok"
                )
                
                result = os.system(detection_command)
                detected_image_path = f"{detected_dir}/results/{uploaded_file.name}"
                
                if result == 0 and os.path.exists(detected_image_path):
                    detected_image = Image.open(detected_image_path)
                    st.image(detected_image, width=500)  
                else:
                    st.error("Detection failed. Please check your YOLO setup and try again.")

with tab2:
    st.write("Coming Soon")
