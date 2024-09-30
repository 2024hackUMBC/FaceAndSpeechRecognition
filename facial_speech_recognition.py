import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import threading
import time
import os
from mongodb_database import collection 

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

recognizer = sr.Recognizer()

recognized_text = None 
last_recognized_text = None 
audio_files = {} 

distance_sum = 0
distance_count = 0

last_submission_time = time.time()

def listen_for_speech():
    global recognized_text
    while True:
        try:
            with sr.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic, timeout=0, phrase_time_limit=7)
                text = recognizer.recognize_google(audio)
                recognized_text = text.lower()
                print(f"Recognized: {recognized_text}")

                save_audio(audio, recognized_text)

        except sr.UnknownValueError:
            continue
        except sr.RequestError:
            print("API unavailable")
        except Exception as e:
            print(f"Error: {str(e)}")

def save_audio(audio, text):
    """Saves the audio to a file corresponding to the recognized text."""
    if text not in audio_files:
        audio_filename = f"{text.replace(' ', '_')}.wav"
        audio_files[text] = audio_filename
    else:
        audio_filename = audio_files[text]

    with open(audio_filename, "wb") as f:
        f.write(audio.get_raw_data())
    print(f"Audio saved to {audio_filename}")

speech_thread = threading.Thread(target=listen_for_speech, daemon=True)
speech_thread.start()

cap = cv2.VideoCapture(0)

reference_real_distance = 10
reference_pixel_distance = 100 
scaling_factor = reference_real_distance / reference_pixel_distance

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.resize(image, (640, 480))

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_tip = (
                    int(face_landmarks.landmark[1].x * image.shape[1]),
                    int(face_landmarks.landmark[1].y * image.shape[0])
                )
                upper_lip = (
                    int(face_landmarks.landmark[13].x * image.shape[1]),
                    int(face_landmarks.landmark[13].y * image.shape[0])
                )
                lower_lip = (
                    int(face_landmarks.landmark[14].x * image.shape[1]),
                    int(face_landmarks.landmark[14].y * image.shape[0])
                )
                left_eye = (
                    int(face_landmarks.landmark[33].x * image.shape[1]),
                    int(face_landmarks.landmark[33].y * image.shape[0])
                )
                right_eye = (
                    int(face_landmarks.landmark[263].x * image.shape[1]),
                    int(face_landmarks.landmark[263].y * image.shape[0])
                )

                lip_center = ((upper_lip[0] + lower_lip[0]) // 2, (upper_lip[1] + lower_lip[1]) // 2)

                nose_to_lip_distance = calculate_distance(nose_tip, lip_center)
                left_eye_to_lip_distance = calculate_distance(left_eye, lip_center)
                right_eye_to_lip_distance = calculate_distance(right_eye, lip_center)

                normalized_nose_to_lip_distance = nose_to_lip_distance * scaling_factor
                normalized_left_eye_to_lip_distance = left_eye_to_lip_distance * scaling_factor
                normalized_right_eye_to_lip_distance = right_eye_to_lip_distance * scaling_factor

                camera_distance_pixels = 1 / (face_landmarks.landmark[1].z + 1e-6)
                camera_distance_cm = camera_distance_pixels * scaling_factor

                if -2.5 <= camera_distance_cm <= -1.9:
                    border_color = (0, 255, 0)
                else:
                    border_color = (0, 0, 255)

                for face_landmark in face_landmarks.landmark:
                    x = int(face_landmark.x * image.shape[1])
                    y = int(face_landmark.y * image.shape[0])
                    cv2.circle(rgb_image, (x, y), 1, border_color, -1)

                if -2.5 <= camera_distance_cm <= -1.9:
                    avg_distance = (normalized_nose_to_lip_distance + normalized_left_eye_to_lip_distance + normalized_right_eye_to_lip_distance) / 3
                    
                    distance_sum += avg_distance
                    distance_count += 1

                    current_time = time.time()
                    if recognized_text and current_time - last_submission_time >= 5:
                        overall_avg_distance = distance_sum / distance_count

                        if recognized_text != last_recognized_text:
                            closest_doc = None
                            smallest_margin = float('inf')

                            for doc in collection.find():
                                distance_difference = abs(doc["average_distance"] - avg_distance)

                                if distance_difference < smallest_margin and distance_difference <= 0.15:
                                    smallest_margin = distance_difference
                                    closest_doc = doc

                            if closest_doc:
                                # Update the matched document with the new average distance
                                collection.update_one(
                                    {"_id": closest_doc["_id"]},
                                    {"$set": {
                                        "average_distance": avg_distance
                                    }}
                                )
                                print(f"Updated Document ID: {closest_doc['_id']} with new average distance: {avg_distance}")
                            else:
                                # Insert a new document if no match found
                                document = {
                                    "recognized_text": recognized_text,
                                    "average_distance": avg_distance
                                }
                                collection.insert_one(document)
                                print(f"Inserted New Document: {document}")

                            last_recognized_text = recognized_text
                            
                            last_submission_time = current_time
                            distance_sum = 0
                            distance_count = 0
                            recognized_text = None

                cv2.putText(rgb_image, f'Nose-Lip Distance: {normalized_nose_to_lip_distance:.2f} cm', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(rgb_image, f'Left Eye-Lip Distance: {normalized_left_eye_to_lip_distance:.2f} cm', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(rgb_image, f'Right Eye-Lip Distance: {normalized_right_eye_to_lip_distance:.2f} cm', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(rgb_image, f'Distance from Camera: {camera_distance_cm:.2f} cm', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Live Video Feed', bgr_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()