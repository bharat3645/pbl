import cv2

# initialize the video capture object
cap = cv2.VideoCapture(0)

# define the object detection classifier
object_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # read the frame from the video capture object
    ret, frame = cap.read()

    # convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect objects in the grayscale frame
    objects = object_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # draw bounding boxes around detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display the resulting frame
    cv2.imshow('Object Detection', frame)

    # exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()