import cv2
import numpy as np

video = cv2.VideoCapture('/home/giselle/Documents/UDEM/Computer-vision/Codigos/tercer-parcial/2023_05_05_14_59_37-ball-detection.mp4')

# Global variables
points = []
mask = None

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global points, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # Create a mask based on the contour
        if len(points) >= 25:
            mask = np.zeros(frame.shape[:2], np.uint8)
            pts = np.array(points, np.int32)
            cv2.fillPoly(mask, [pts], 255)

# Create a window and set the mouse callback function
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)

# Iterate through each frame of the video
while video.isOpened():
    ret, frame = video.read()
    cv2.waitKey(20)

    if not ret:
        break

    # Draw the selected points
    for point in points:
        cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Create a copy of the frame
    frame_copy = frame.copy()

    # Apply the mask to the frame
    if mask is not None:
        frame_copy = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

    # Display the resulting frame
    cv2.imshow('Video', frame_copy)

    # Save the selected region to a separate image when "S" key is pressed
    if mask is not None and cv2.waitKey(1) & 0xFF == ord('s'):
        selected_region = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imwrite('selected_region.png', selected_region)
        print("Selected region saved as 'selected_region.png'")

        # Reset the mask and points
        mask = None
        points = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video.release()
cv2.destroyAllWindows()
