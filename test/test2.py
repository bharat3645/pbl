import cv2

# Load the YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Define the list of classes
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture object
    ret, frame = cap.read()

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Set the input for the YOLOv3 model
    net.setInput(blob)

    # Get the output from the YOLOv3 model
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize the list of detected objects
    objects = []

    # Loop over the detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                objects.append([classes[class_id], confidence, x, y, w, h])

    # Draw bounding boxes around detected objects
    for obj in objects:
        class_id, confidence, x, y, w, h = obj
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_id}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()