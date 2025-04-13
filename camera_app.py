import streamlit as st
import cv2
import numpy as np
from face_verification_v2 import FaceVerificationModel
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class FaceVerificationTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = FaceVerificationModel()
        self.reference_img = "alaa.jpg"
        self.last_result = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Perform verification on every frame
        verified, distance, threshold, error = self.model.verify_faces(
            img, 
            self.reference_img,
            enforce_detection=False
        )
        
        # Store the latest result
        self.last_result = {
            'verified': verified,
            'distance': distance,
            'threshold': threshold,
            'error': error
        }
        
        return img

def main():
    st.title("Live Face Verification App")
    
    st.write("""
    ## Camera Access Instructions:
    1. Click 'Start' below to enable camera
    2. Grant camera permissions when prompted
    3. Face the camera properly for continuous verification
    """)
    
    result_placeholder = st.empty()
    
    def video_frame_callback(frame):
        transformer = FaceVerificationTransformer()
        output_frame = transformer.transform(frame)
        
        # Display verification result
        if transformer.last_result:
            result = transformer.last_result
            if result['error']:
                result_placeholder.error(f"Error: {result['error']}")
            else:
                if result['verified']:
                    result_placeholder.success(
                        f"Verified! Distance: {result['distance']:.2f} "
                        f"(threshold: {result['threshold']:.2f})"
                    )
                else:
                    result_placeholder.warning(
                        f"Not verified! Distance: {result['distance']:.2f} "
                        f"(threshold: {result['threshold']:.2f})"
                    )
        
        return output_frame
    
    webrtc_ctx = webrtc_streamer(
        key="face-verification",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

if __name__ == "__main__":
    main()
