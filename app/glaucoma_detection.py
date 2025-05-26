import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import asyncio

class PupilDetector:
    def __init__(self):
        # Initialize the eye cascade classifier for eye detection
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_pupil(self, eye_frame):
        """
        Detect pupil from an eye frame
        
        Args:
            eye_frame: Grayscale image of an eye
            
        Returns:
            pupil_center: (x, y) coordinates of pupil center
            pupil_radius: Radius of detected pupil
            processed_eye: Processed eye frame with pupil highlighted
        """
        # Make a copy for processing and output
        processed_eye = eye_frame.copy()
        if len(eye_frame.shape) > 2:
            gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_frame
            processed_eye = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2BGR)
        
        # Apply histogram equalization to enhance contrast
        gray_eye = cv2.equalizeHist(gray_eye)
        
        # Apply Gaussian blur to reduce noise
        blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        
        # Apply binary thresholding to identify dark regions (pupil)
        _, thresh = cv2.threshold(blurred_eye, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pupil_center = None
        pupil_radius = 0
        
        if contours:
            # Find the largest contour which is most likely to be the pupil
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only proceed if the contour is large enough
            if area > 100:
               
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                pupil_center = (int(x), int(y))
                pupil_radius = int(radius)
                
                # Draw the pupil circle on the output image
                cv2.circle(processed_eye, pupil_center, pupil_radius, (0, 255, 0), 2)
                cv2.circle(processed_eye, pupil_center, 1, (0, 0, 255), 3)
        
        return pupil_center, pupil_radius, processed_eye
    
    def extract_pupil_features(self, eye_frame):
        """
        Extract relevant features from detected pupil
        
        Args:
            eye_frame: Image frame containing the eye
            
        Returns:
            features: Dict containing pupil features
        """
        pupil_center, pupil_radius, _ = self.detect_pupil(eye_frame)
        
        features = {
            'pupil_detected': pupil_center is not None,
            'pupil_radius': pupil_radius,
            'pupil_center': pupil_center
        }
        
        if pupil_center:
            # Calculate additional features if pupil is detected
            features['pupil_area'] = np.pi * (pupil_radius ** 2)
            
            # Analyze pupil shape (circularity)
            if pupil_radius > 0:
                gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY) if len(eye_frame.shape) > 2 else eye_frame
                _, thresh = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(largest_contour, True)
                    area = cv2.contourArea(largest_contour)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        features['circularity'] = circularity
        
        return features

class RealTimeGlaucomaDetection:
    def __init__(self, model_path='models/best_glaucoma_model.keras', threshold=0.5):
        self.model_path = model_path
        self.pupil_detector = PupilDetector()
        self.model = None
        self.threshold = threshold
        
        # Load the model if it exists
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model not found at {model_path}. Make sure to train the model first.")
        
        self.input_shape = (224, 224, 3)
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for model prediction
        """
        # Resize to model input shape
        resized_frame = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize pixel values
        normalized_frame = resized_frame / 255.0
        
        return normalized_frame
    
    def analyze_eye_frame(self, frame):
        """
        Analyze a single eye frame for pupil and glaucoma detection
        """
        # Detect pupil
        pupil_center, pupil_radius, processed_frame = self.pupil_detector.detect_pupil(frame)
        
        # Extract features
        features = self.pupil_detector.extract_pupil_features(frame)
        
        # Prepare for model prediction
        preprocessed_frame = self.preprocess_frame(frame)
        
        glaucoma_probability = None
        detection_result = "No prediction"
        
        # Make prediction if model is loaded
        if self.model is not None:
            # Convert to proper shape for model input
            input_data = np.expand_dims(preprocessed_frame, axis=0)
            
            # Make prediction
            glaucoma_probability = float(self.model.predict(input_data)[0][0])
            
            # Determine classification based on threshold
            if glaucoma_probability > self.threshold:
                detection_result = "Glaucoma"
                color = (0, 0, 255)  # Red for Glaucoma
            else:
                detection_result = "Healthy"
                color = (0, 255, 0)  # Green for Healthy
            
            # Add result to frame
            result_text = f"{detection_result} ({glaucoma_probability:.2f})"
            cv2.putText(processed_frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return processed_frame, detection_result, glaucoma_probability, features
    
    async def run_webcam_detection_async(self):
        """
        Run real-time glaucoma detection using webcam (async version for FastAPI)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
            
        print("Starting real-time detection. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame from webcam.")
                    break
                
                # Convert to grayscale for eye detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect eyes
                eyes = self.pupil_detector.eye_cascade.detectMultiScale(gray, 1.3, 5)
                
                # If no eyes detected
                if len(eyes) == 0:
                    cv2.putText(frame, "No eyes detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Process each detected eye
                for (ex, ey, ew, eh) in eyes:
                    # Extract eye region
                    eye_frame = frame[ey:ey+eh, ex:ex+ew]
                    
                    # Check if eye frame is valid
                    if eye_frame.size == 0:
                        continue
                    
                    # Analyze eye
                    processed_eye, result, prob, features = self.analyze_eye_frame(eye_frame)
                    
                    # Replace eye region in original frame
                    frame[ey:ey+eh, ex:ex+ew] = processed_eye
                    
                    # Draw rectangle around eye
                    cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                    
                    # Display pupil info if detected
                    if features['pupil_detected']:
                        pupil_info = f"Pupil R: {features['pupil_radius']}"
                        cv2.putText(frame, pupil_info, (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Display the resulting frame
                cv2.imshow('Glaucoma Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Add small delay to prevent blocking the event loop
                await asyncio.sleep(0.01)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_webcam_detection(self):
        """
        Run real-time glaucoma detection using webcam (synchronous version for backward compatibility)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
            
        print("Starting real-time detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            
            # Convert to grayscale for eye detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = self.pupil_detector.eye_cascade.detectMultiScale(gray, 1.3, 5)
            
            # If no eyes detected
            if len(eyes) == 0:
                cv2.putText(frame, "No eyes detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Process each detected eye
            for (ex, ey, ew, eh) in eyes:
                # Extract eye region
                eye_frame = frame[ey:ey+eh, ex:ex+ew]
                
                # Check if eye frame is valid
                if eye_frame.size == 0:
                    continue
                
                # Analyze eye
                processed_eye, result, prob, features = self.analyze_eye_frame(eye_frame)
                
                # Replace eye region in original frame
                frame[ey:ey+eh, ex:ex+ew] = processed_eye
                
                # Draw rectangle around eye
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                
                # Display pupil info if detected
                if features['pupil_detected']:
                    pupil_info = f"Pupil R: {features['pupil_radius']}"
                    cv2.putText(frame, pupil_info, (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display the resulting frame
            cv2.imshow('Glaucoma Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run real-time detection
    print("Starting real-time glaucoma detection...")
    detector = RealTimeGlaucomaDetection()
    detector.run_webcam_detection()