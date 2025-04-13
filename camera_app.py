import streamlit as st
import cv2
import os
import tempfile
from face_verification_v2 import FaceVerificationModel

def main():
    st.title("Face Verification Camera App")
    
    # Initialize face verification model
    model = FaceVerificationModel()
    reference_img = "alaa.jpg"
    
    # Get list of available cameras
    camera_indices = []
    for i in range(0, 5):  # Check first 5 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_indices.append(i)
            cap.release()
    
    if not camera_indices:
        st.error("No cameras found!")
        return
    
    selected_camera = st.selectbox("Select Camera", camera_indices)
    verification_placeholder = st.empty()
    frame_placeholder = st.empty()
    
    cap = cv2.VideoCapture(selected_camera)
    stop_button = st.button("Stop")
    
    # Verification frequency control (every 10 frames)
    frame_count = 0
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        # Perform verification every 10 frames
        frame_count += 1
        if frame_count % 10 == 0:
            # Save current frame to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, frame)
            
            # Verify against reference image
            verified, distance, threshold, error = model.verify_faces(
                tmp_path, 
                reference_img,
                enforce_detection=False
            )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Display verification result
            if error:
                verification_placeholder.error(f"Error: {error}")
            else:
                if verified:
                    verification_placeholder.success(f"Verified! Distance: {distance:.2f} (threshold: {threshold:.2f})")
                else:
                    verification_placeholder.warning(f"Not verified! Distance: {distance:.2f} (threshold: {threshold:.2f})")
        
        if stop_button:
            break
    
    cap.release()

if __name__ == "__main__":
    main()
