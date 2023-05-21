
import cv2
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

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
    neighborhood_size = 5
    dilated_mask = cv2.dilate(mask, np.ones((neighborhood_size, neighborhood_size), np.uint8))
    neighborhood = cv2.absdiff(dilated_mask, mask)

    # Generate the histogram of the neighborhood
    hist_channel_0 = cv2.calcHist([image], [0], neighborhood, [256], [0, 256])
    hist_channel_1 = cv2.calcHist([image], [1],neighborhood, [256], [0, 256])
    hist_channel_2 = cv2.calcHist([image], [2],neighborhood, [256], [0, 256])

    # Calculate the cumulative distribution function (CDF) for each channel
    cdf_channel_0 = np.cumsum(hist_channel_0) / np.sum(hist_channel_0)
    cdf_channel_1 = np.cumsum(hist_channel_1) / np.sum(hist_channel_1)
    cdf_channel_2 = np.cumsum(hist_channel_2) / np.sum(hist_channel_2)

    # Normalize the CDF values
    cdf_channel_0 /= cdf_channel_0[-1]
    cdf_channel_1 /= cdf_channel_1[-1]
    cdf_channel_2 /= cdf_channel_2[-1]

    # Generate random numbers for each pixel in the mask
    random_numbers = np.random.rand(*mask.shape[:2])
    # Create a new image with the same shape as the original image, initialized with zeros
    new_image = np.zeros_like(image)

    # Fill the new image with possible values based on the neighborhood distribution
    for i in range(dilated_mask.shape[0]):
        for j in range(dilated_mask.shape[1]):
            if dilated_mask[i, j] == 255:
                # Get the intensity values from the random number
                intensity_0 = np.argmax(cdf_channel_0 >= random_numbers[i, j])
                intensity_1 = np.argmax(cdf_channel_1 >= random_numbers[i, j])
                intensity_2 = np.argmax(cdf_channel_2 >= random_numbers[i, j])

                # Assign the intensity values to the corresponding pixel in the new image
                new_image[i, j, 0] = intensity_0
                new_image[i, j, 1] = intensity_1
                new_image[i, j, 2] = intensity_2

    #Combine the new image with the original image to replace the object region
    
    result = cv2.bitwise_and(image,image,mask= cv2.bitwise_not(dilated_mask))+new_image
    result=cv2.medianBlur(result,3)
    cv2.imshow('result',result)
    return mask,hist_channel_0,hist_channel_1,hist_channel_2,new_image

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
        mask,hist0,hist1,hist2,new_image=crop_image(frame, clicked_points)
        cv2.imshow('new_image',new_image)
        cv2.imshow('mask',mask)
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
            save_directory = os.path.abspath(os.path.join(os.getcwd(), "..", "img"))  # Replace "desired_directory" with your desired directory name
            os.makedirs(save_directory, exist_ok=True)
            save_path = os.path.join(save_directory, "new_image_{}.jpg".format(timestamp))
            cv2.imwrite(save_path,new_image)
            print("New image saved as:", save_path)
        else:
            print("Cannot save New image. Please enter crop mode and select at least 3 points.")
   

    # Exit if 'q' is pressed
    if key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


# Plot the histogram
plt.figure(figsize=(8, 6))
plt.plot(hist0, color='b', label='Blue')
plt.plot(hist1, color='g', label='Green')
plt.plot(hist2, color='r', label='Red')
plt.title("Histograms of the Neighborhood (RGB)")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
