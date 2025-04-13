import streamlit as st
import cv2
import numpy as np
from PIL import Image
from face_verification_v2 import FaceVerificationModel
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class FaceVerificationTransformer(VideoTransformerBase):
    def __init__(self, reference_img="alaa.jpg"):
        self.model = FaceVerificationModel()
        self.reference_img = reference_img
        self.last_result = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.verify_image(img)
        return img

    def verify_image(self, img):
        verified, distance, threshold, error = self.model.verify_faces(
            img, 
            self.reference_img,
            enforce_detection=False
        )
        
        self.last_result = {
            'verified': verified,
            'distance': distance,
            'threshold': threshold,
            'error': error
        }

def main():
    st.title("Face Verification App")
    
    mode = st.radio("Select Mode:", ("Live Camera Verification", "Upload Image for Verification"))
    
    result_placeholder = st.empty()
    image_placeholder = st.empty()
    
    if mode == "Live Camera Verification":
        st.write("""
        ## Camera Access Instructions:
        1. Click 'Start' below to enable camera
        2. Grant camera permissions when prompted
        3. Face the camera properly for verification
        """)
        
        def video_frame_callback(frame):
            transformer = FaceVerificationTransformer()
            output_frame = transformer.transform(frame)
            
            if transformer.last_result:
                display_result(transformer.last_result, result_placeholder)
            
            return output_frame
        
        webrtc_ctx = webrtc_streamer(
            key="face-verification",
            video_frame_callback=video_frame_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )
    
    else:  # Upload Image mode
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            image_placeholder.image(image, caption="Uploaded Image", use_column_width=True)
            
            transformer = FaceVerificationTransformer()
            transformer.verify_image(img_bgr)
            
            if transformer.last_result:
                display_result(transformer.last_result, result_placeholder)

def display_result(result, placeholder):
    if result['error']:
        placeholder.error(f"Error: {result['error']}")
    else:
        if result['verified']:
            placeholder.success(
                f"Verified! Distance: {result['distance']:.2f} "
                f"(threshold: {result['threshold']:.2f})"
            )
        else:
            placeholder.warning(
                f"Not verified! Distance: {result['distance']:.2f} "
                f"(threshold: {result['threshold']:.2f})"
            )

if __name__ == "__main__":
    main()
