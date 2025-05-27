from fastapi import APIRouter, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import dlib
import numpy as np
from joblib import load
import base64
import asyncio
import time
import random
import json

exercise_router = APIRouter()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")  # templates folder inside /app/
MODEL_DIR = os.path.join(BASE_DIR, "utils")

# Load models
model = load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
predictor = dlib.shape_predictor(os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat"))
detector = dlib.get_frontal_face_detector()

@exercise_router.get("/eye-test")
def get_index():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@exercise_router.get("/face-test")
def get_face_test():
    return FileResponse(os.path.join(TEMPLATES_DIR, "face_test.html"))

@exercise_router.get("/basic-face-test")
def get_basic_face_test():
    return FileResponse(os.path.join(TEMPLATES_DIR, "basic_face_test.html"))

@exercise_router.get("/exercise")
def get_exercise():
    return FileResponse(os.path.join(TEMPLATES_DIR, "exercise.html"))

@exercise_router.get("/exercise_full")
def get_exercise_full():
    return FileResponse(os.path.join(TEMPLATES_DIR, "exercise_full.html"))

# # Load models once
model_path = "random_forest_model.pkl"
predictor_path = "shape_predictor_68_face_landmarks.dat"
model = load(model_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

label_map = {
    0: "TopLeft", 1: "TopCenter", 2: "TopRight",
    3: "MiddleLeft", 4: "MiddleRight",
    5: "BottomLeft", 6: "BottomCenter", 7: "BottomRight"
}

exercise_definitions = {
    "circular_motion": {
        "name": "Circular Motion",
        "description": "Follow the target in a circular pattern to strengthen eye muscles",
        "duration": 30,  # seconds
        "target_positions": [
            "TopCenter", "TopRight", "MiddleRight", "BottomRight", 
            "BottomCenter", "BottomLeft", "MiddleLeft", "TopLeft"
        ],
        "position_duration": 3,  # seconds per position
        "repetitions": 1,
        "instructions": "Follow the moving dot with your eyes without moving your head."
    },
    "focus_shifting": {
        "name": "Focus Shifting",
        "description": "Shift focus between near and far points to improve eye flexibility",
        "duration": 30,
        "target_positions": ["MiddleLeft", "MiddleRight"],
        "position_duration": 3,
        "repetitions": 5,
        "instructions": "Focus on the left point, then quickly shift to the right point. Repeat."
    },
    "diagonal_movement": {
        "name": "Diagonal Movement",
        "description": "Move eyes diagonally to exercise all eye muscles",
        "duration": 30,
        "target_positions": ["TopLeft", "BottomRight", "TopRight", "BottomLeft"],
        "position_duration": 3,
        "repetitions": 3,
        "instructions": "Move your eyes diagonally from top-left to bottom-right, then top-right to bottom-left."
    },
    "figure_eight": {
        "name": "Figure Eight",
        "description": "Trace a figure eight pattern to improve eye coordination",
        "duration": 30,
        "target_positions": [
            "MiddleLeft", "TopLeft", "TopCenter", "TopRight", "MiddleRight", 
            "BottomRight", "BottomCenter", "BottomLeft"
        ],
        "position_duration": 2,
        "repetitions": 2,
        "instructions": "Follow the target as it moves in a figure-eight pattern."
    },
    "Up_and_Down_Eye_Movement": {
        "name": "Up and Down Eye Movement",
        "description": "Move eyes Up and Down to exercise all eye muscles",
        "duration": 30,
        "target_positions": [
            "TopCenter", "BottomCenter"
        ],
        "position_duration": 2,
        "repetitions": 2,
        "instructions": "Follow the target as it moves in a up and down pattern."
    }
}

@exercise_router.websocket("/ws/exercise")
async def eye_exercise_ws(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    
    try:
        # Wait for exercise selection from client
        exercise_request = await websocket.receive_json()
        exercise_id = exercise_request.get("exercise_id")
        
        if exercise_id not in exercise_definitions:
            await websocket.send_json({
                "status": "error",
                "message": "Invalid exercise ID"
            })
            return
        
        # Get the selected exercise
        exercise = exercise_definitions[exercise_id]
        
        # Send exercise details to client
        await websocket.send_json({
            "status": "starting",
            "exercise": exercise
        })
        
        # Give user time to prepare
        await asyncio.sleep(3)
        
        # Initialize exercise variables
        start_time = time.time()
        current_position_index = 0
        current_rep = 1
        position_start_time = start_time
        correct_position_time = 0
        exercise_score = 0
        max_score = exercise["duration"]  # Perfect score would be maintaining correct gaze for entire duration
        
        while time.time() - start_time < exercise["duration"]:
            # Calculate which position should be active now
            elapsed_exercise_time = time.time() - start_time
            elapsed_position_time = time.time() - position_start_time
            
            if elapsed_position_time >= exercise["position_duration"]:
                # Move to next position
                current_position_index = (current_position_index + 1) % len(exercise["target_positions"])
                if current_position_index == 0:
                    current_rep += 1
                position_start_time = time.time()
            
            # Current target position
            target_position = exercise["target_positions"][current_position_index]
            
            # Process camera frame
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            # Default values if no face/eyes detected
            gaze_label = "No face detected"
            gaze_match = False
            
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                left_eye_pts = np.array(left_eye_pts)
                x, y, w, h = cv2.boundingRect(left_eye_pts)
                left_eye = gray[y:y+h, x:x+w]
                
                if left_eye.size > 0:
                    eye_resized = cv2.resize(left_eye, (100, 100))
                    eye_flat = eye_resized.flatten()
                    prediction = model.predict([eye_flat])[0]
                    gaze_label = label_map.get(prediction, "other_direction")
                    
                    # Check if user is looking at the target
                    gaze_match = (gaze_label == target_position)
                    
                    if gaze_match:
                        correct_position_time += 0.03  # Approximate frame time
                        exercise_score += 0.03  # Add to score when looking at correct position
                    
                    # Draw feedback on frame
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    cv2.putText(frame, f"Looking: {gaze_label}", (face.left(), face.top()-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Target: {target_position}", (face.left(), face.top()-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                (0, 255, 0) if gaze_match else (0, 0, 255), 2)
            
            # Add exercise info to frame
            cv2.putText(frame, f"Exercise: {exercise['name']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Rep {current_rep}/{exercise['repetitions']}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add timer to frame
            remaining_time = exercise["duration"] - elapsed_exercise_time
            cv2.putText(frame, f"Time: {remaining_time:.1f}s", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw target indicator in corner of frame
            indicator_size = 100
            indicator = np.zeros((indicator_size, indicator_size, 3), dtype=np.uint8)
            
            # Draw the grid
            cell_size = indicator_size // 3
            for i in range(1, 3):
                cv2.line(indicator, (0, i*cell_size), (indicator_size, i*cell_size), (100, 100, 100), 1)
                cv2.line(indicator, (i*cell_size, 0), (i*cell_size, indicator_size), (100, 100, 100), 1)
            
            # Highlight the target position
            position_map = {
                "TopLeft": (0, 0),
                "TopCenter": (1, 0),
                "TopRight": (2, 0),
                "MiddleLeft": (0, 1),
                "Center": (1, 1),
                "MiddleRight": (2, 1),
                "BottomLeft": (0, 2),
                "BottomCenter": (1, 2),
                "BottomRight": (2, 2)
            }
            
            if target_position in position_map:
                col, row = position_map[target_position]
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                cv2.rectangle(indicator, (x1, y1), (x2, y2), (0, 255, 0), -1)
                
                # Mark user's gaze position
                if gaze_label in position_map:
                    gaze_col, gaze_row = position_map[gaze_label]
                    gx1 = gaze_col * cell_size + cell_size//3
                    gy1 = gaze_row * cell_size + cell_size//3
                    gx2 = gx1 + cell_size//3
                    gy2 = gy1 + cell_size//3
                    cv2.circle(indicator, (gx1 + (gx2-gx1)//2, gy1 + (gy2-gy1)//2), 
                               cell_size//4, (0, 0, 255), -1)
            
            # Add indicator to the frame
            h, w = frame.shape[:2]
            padding = 20
            frame[padding:padding+indicator_size, w-indicator_size-padding:w-padding] = indicator
            
            # Convert to base64 and send
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            progress_percentage = (elapsed_exercise_time / exercise["duration"]) * 100
            accuracy_percentage = (correct_position_time / elapsed_exercise_time) * 100 if elapsed_exercise_time > 0 else 0
            
            # Send frame and progress data
            await websocket.send_json({
                "status": "exercising",
                "frame": frame_base64,
                "current_position": target_position,
                "gaze_position": gaze_label,
                "is_matching": gaze_match,
                "progress": progress_percentage,
                "accuracy": accuracy_percentage,
                "time_remaining": remaining_time,
                "current_rep": current_rep,
                "total_reps": exercise["repetitions"]
            })
            
            await asyncio.sleep(0.03)  # ~30 FPS
        
        # Calculate final score
        final_score = int((exercise_score / max_score) * 100)
        
        # Send completion message
        await websocket.send_json({
            "status": "completed",
            "exercise_id": exercise_id,
            "score": final_score,
            "accuracy": (correct_position_time / exercise["duration"]) * 100,
            "message": f"Exercise completed with score: {final_score}/100"
        })
            
    except Exception as e:
        print(f"WebSocket closed or error: {e}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        cap.release()


@exercise_router.websocket("/ws/eye")
async def eye_tracking_ws(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            gaze_label = "No face detected"

            for face in faces:
                landmarks = predictor(gray, face)
                left_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                left_eye_pts = np.array(left_eye_pts)
                x, y, w, h = cv2.boundingRect(left_eye_pts)
                left_eye = gray[y:y+h, x:x+w]

                if left_eye.size > 0:
                    eye_resized = cv2.resize(left_eye, (100, 100))
                    eye_flat = eye_resized.flatten()
                    prediction = model.predict([eye_flat])[0]
                    gaze_label = label_map.get(prediction, "other_direction")

                    # Draw rectangle and label on frame
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    cv2.putText(frame, gaze_label, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Optional: send frame too
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_json({
                "label": gaze_label,
                "frame": frame_base64
            })

            await asyncio.sleep(0.03)  # ~30 FPS
    except Exception as e:
        print("WebSocket closed or error:", e)
    finally:
        cap.release()

# Optional: Add a basic face detection test endpoint
@exercise_router.websocket("/ws/face")
async def face_detection_ws(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Draw rectangles around detected faces
            for face in faces:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, f"Face", (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Convert to base64 and send
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_json({
                "faces_detected": len(faces),
                "frame": frame_base64
            })

            await asyncio.sleep(0.03)  # ~30 FPS
    except Exception as e:
        print("WebSocket closed or error:", e)
    finally:
        cap.release()

