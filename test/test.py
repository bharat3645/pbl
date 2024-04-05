import cv2
import pytesseract
import numpy as np
import time
cam = cv2.VideoCapture(0)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
while True:
    _,frm = cam.read()
    cv2.imshow('frame',frm)
    time.sleep(1)
    print(pytesseract.image_to_string(frm))

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cam.release() 
cv2.destroyAllWindows()