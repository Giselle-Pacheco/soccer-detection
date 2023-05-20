
import cv2
import numpy as np
import datetime

# Global variables
clicked_points = []
crop_mode = False

def get_circle(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
    

def crop_image(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(points)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    return cropped_image

# Read video file
video_path = "/home/enchi/Vídeos/2023_05_05_14_59_37-ball-detection.mp4"  # Replace with your video file path"
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Create a window and bind the mouse callback function
window_name = "Video"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, get_circle)

# Read and display frames from the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Cannot read the video file")
        break

    if crop_mode and len(clicked_points) >= 3:
        cropped_frame = crop_image(frame, clicked_points)
        cv2.imshow(window_name, cropped_frame)
    else:
        # Draw clicked points
        for point in clicked_points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

        # Connect the points with lines
        if len(clicked_points) >= 2:
            for i in range(len(clicked_points) - 1):
                cv2.line(frame, clicked_points[i], clicked_points[i + 1], (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow(window_name, frame)

    # Keyboard event handling
    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):
        crop_mode = not crop_mode
        if crop_mode and len(clicked_points) < 3:
            print("Please select at least 3 points to define the area")
            crop_mode = False

    if key == ord('s'):
        if crop_mode and len(clicked_points) >= 3:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = "/home/enchi/Imágenes/cropped_image{}.jpg".format(timestamp)
            cv2.imwrite(save_path, cropped_frame)
            print("Cropped image saved as:", save_path)
        else:
            print("Cannot save cropped image. Please enter crop mode and select at least 3 points.")

       

    # Exit if 'q' is pressed
    if key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
