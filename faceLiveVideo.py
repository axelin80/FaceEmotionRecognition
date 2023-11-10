import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from datetime import datetime


# Load the cascade file for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Open the default camera for video capture
cap = cv2.VideoCapture(0)

# Emotion order to be displayed in the y-axis
emotion_order = ['fear', 'angry', 'disgust', 'sad', 'neutral', 'happy', 'surprise']

# Define the arrays to storage the timestamps and emotion detected
emotion_detected = []
timetamps_detected = []


while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles and green lines around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        img1 = frame[x:x + w, y:y + h]

        if img1.size > 0:
            result = DeepFace.analyze(img1, actions=['emotion'], enforce_detection=False) #actions=['emotion'],
            #print(str(result))
            cv2.putText(frame, result[0]['dominant_emotion'], (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            timetamps_detected.append(datetime.now())
            emotion_detected.append(result[0]['dominant_emotion'])
        else:
            print("No face detected")

        #cv2.line(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.line(frame, (x, y + h), (x + w, y), (0, 255, 0), 2)
        #roi_gray = gray[y:y + h, x:x + w]
        #roi_color = frame[y:y + h, x:x + w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in eyes:
        #    cv2.circle(roi_color, (ex + int(ew / 2), ey + int(eh / 2)), int(ew / 2), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Faces", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# Creating emotion detection graph
plt.figure(figsize=(10, 6))
plt.plot(timetamps_detected, emotion_detected, marker='o')
plt.xlabel('Time')
plt.ylabel('Emotion detected')
plt.title('Emotion detection over time')
plt.xticks(rotation=45)
plt.yticks(emotion_order)
plt.grid(True)
plt.tight_layout()
plt.show()
