import cv2 as cv
import numpy as np

# 1. Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=175, qualityLevel=0.01, minDistance=50, blockSize=15)

# 2. Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(25, 25), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))

#Read the video file
cap = cv.VideoCapture("clip1.mp4")     #change clip number here to test different clips ;)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

#Read the first frame
ans, first_frame = cap.read()
if not ans:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

if prev is None:
    print("Error: No good features found!")
    cap.release()
    exit()
else:
    print(f"Initial keypoints detected: {len(prev)}")

prev = np.float32(prev) 
mask = np.zeros_like(first_frame)

while cap.isOpened():
    print("Reading a new frame...")  #debugging message
    boolean, frame = cap.read()
    if not boolean or frame is None:
        print("No more frames to read or error in reading.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # computing optical flow
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

    if next is None or status is None:
        print("Optical flow computation failed.")
        break

    good_old = prev[status == 1]
    good_new = next[status == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
        frame = cv.circle(frame, (int(a), int(b)), 3, (0, 0, 255), 0)

    output = cv.add(frame, mask)
    cv.imshow("Sparse Optical Flow", output)

    prev_gray = gray.copy()
    prev = good_new.reshape(-1, 1, 2)  #updating points for next iteration

    if cv.waitKey(10) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
