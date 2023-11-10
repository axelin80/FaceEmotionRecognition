import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from datetime import datetime

# Load the cascade file for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load a pre trained model from DeepFace
emotion_model = DeepFace.build_model("Emotion")

# Define the emotion
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Open the default camera for video capture
path = ".\Emotions.mp4"
cap = cv2.VideoCapture(path)

# Adjust frame width and height for downsampling
frame_width = 640  # Adjust to your desired width
frame_height = 480  # Adjust to your desired height

cap.set(3, frame_width)
cap.set(4, frame_height)

# Define the arrays to storage the timestamps and emotion detected
emotion_detected = []
timetamps_detected = []


while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    for(x, y, w, h) in faces:
        img1 = frame[y:y + h, x:x + w]
        result = DeepFace.analyze(img1, actions=["emotion"], enforce_detection=False)
        #print(emotion_labels)
        emotion_label = result[0]['dominant_emotion']
        timetamps_detected.append(datetime.now())
        emotion_detected.append(emotion_label)
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the emotion label above the face
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face and Emotion Recognition', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if ret:
    #     #Detecting faces in the video
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     # Detect faces in the grayscale frame
    #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #
    #     # Draw rectangles and green lines around the faces
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         img1 = frame[x:x + w, y:y + h]
    #
    #         if img1.size > 0:
    #             result = DeepFace.analyze(img1, actions=['emotion'], enforce_detection=False)  # actions=['emotion'],
    #             cv2.putText(frame, result[0]['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
    #                         (0, 255, 0), 2)
    #             timetamps_detected.append(datetime.now())
    #             emotion_detected.append(result[0]['dominant_emotion'])
    #         else:
    #             print("No face detected")
    #
    #         roi_gray = gray[y:y + h, x:x + w]
    #         roi_color = frame[y:y + h, x:x + w]
    #
    #     cv2.imshow("Faces", frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    # else:
    #     break

cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(10, 6))
plt.plot(timetamps_detected, emotion_detected, marker='o')
plt.xlabel('Time')
plt.ylabel('Emotion detected')
plt.title('Emotion detection over time')
plt.xticks(rotation=45)
plt.yticks(emotion_labels)
plt.grid(True)
plt.tight_layout()
plt.show()