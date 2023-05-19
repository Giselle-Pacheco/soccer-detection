# import numpy as np
import cv2
import argparse
from multiprocessing import Pool, cpu_count
import time
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture('/home/giselle/Documents/UDEM/Computer-vision/Codigos/tercer-parcial/2023_05_05_14_59_37-ball-detection.mp4')


# Global variables
drawing = False
points = []

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global drawing, points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Load the video

# Create a window and set the mouse callback function
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)

# Iterate through each frame of the video
while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    # Draw the selected points
    for point in points:
        cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Convert points to a NumPy array
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw the contour if at least 3 points are selected
    if len(pts) >= 3:
        cv2.drawContours(frame, [pts], -1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video.release()
cv2.destroyAllWindows()

