import cv2 as cv
import numpy as np

cap = cv.VideoCapture("scene.mp4")  #Change file name here to test different clips ;)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

#Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

#Convert 
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255  

while cap.isOpened():
    print("Reading a new frame...")  #debugging message
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * (180 / np.pi) / 2  
    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    flow_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow("Dense Optical Flow", flow_rgb)

    prev_gray = gray.copy()

    if cv.waitKey(10) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()