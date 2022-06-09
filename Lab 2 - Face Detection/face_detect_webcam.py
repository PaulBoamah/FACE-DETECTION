import cv2
from cv2 import imwrite
import imutils
from datetime import datetime

#Set casc Path as file located in script directory
cascPath = 'haarcascade_frontalface_default.xml'

#Create haar cascade using the classifier function
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the Video
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize the Frame to improve speed
    frame = imutils.resize(frame, width=450)

    # Convert to Gray-Scale for easy reading 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(25, 25)
    )

    faces_detected = []
    # Draw a rectangle around the faces detected
    for (x, y, w, h) in faces:
        now = datetime.now()
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #use now.strftime to retrieve current time and store in cur_time variable
        cur_time = now.strftime("%H%M%S")

        
        #After detecting face and drawing rectangle capture frame and store image in script directory
        if cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2).all != 0:
            cv2.imwrite(cur_time + ".png", frame)
        
    
    # Display the resulting Frame
    cv2.imshow('Video', frame)

    #write block to quit program by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
