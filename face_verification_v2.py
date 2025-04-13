from deepface import DeepFace
import numpy as np

class FaceVerificationModel:
    def __init__(self):
        """Initialize DeepFace model"""
        self.model = DeepFace.build_model("Facenet")
        print("Face verification model initialized successfully")
        
    def verify_faces(self, img1_path, img2_path, threshold=0.4, enforce_detection=True):
        """
        Verify if two images show the same person
        Returns: (verified, distance, threshold, error_message)
        """
        try:
            # First verify both images can be processed
            for img_path in [img1_path, img2_path]:
                try:
                    faces = DeepFace.extract_faces(img_path=img_path, 
                                                detector_backend='opencv')
                    if not faces or len(faces) == 0:
                        raise ValueError(f"No faces detected in {img_path}")
                except Exception as e:
                    if enforce_detection:
                        raise ValueError(f"Face processing failed in {img_path}: {str(e)}") from e
            
            # If both images are valid, perform verification
            result = DeepFace.verify(img1_path=img1_path,
                                   img2_path=img2_path,
                                   model_name="Facenet",
                                   distance_metric='cosine',
                                   enforce_detection=enforce_detection)
            return result['verified'], result['distance'], result['threshold'], None
            
        except Exception as e:
            error_msg = f"Verification failed: {str(e)}"
            print(error_msg)
            return False, None, None, error_msg

if __name__ == '__main__':
    model = FaceVerificationModel()
    # Test cases
    #print("Same person test (strict):")
    #result = model.verify_faces("person1.jpg", "person2.jpg", enforce_detection=True)
   # print(f"Result: {result}")
    
    print("\nSame person test (lenient):")
    result = model.verify_faces(r"C:\Projects(Behoos_Ai)\Attendance_system_model_selection_process\dataset\archive2\val\n000040\0002_01.jpg", r"C:\Projects(Behoos_Ai)\Attendance_system_model_selection_process\dataset\archive2\val\n000040\0161_02.jpg", enforce_detection=False)
    print(f"Result: {result}")

