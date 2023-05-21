
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
    mask = np.zeros_like(image)
    points = np.array(points)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    neighborhood_size = 5
    dilated_mask = cv2.dilate(mask, np.ones((neighborhood_size, neighborhood_size), np.uint8))
    neighborhood = cv2.absdiff(dilated_mask, mask)
    neighborhood_pixels = np.where(neighborhood > 0)
    neighborhood_region = image[min(neighborhood_pixels[0]):max(neighborhood_pixels[0]) + 1,
                            min(neighborhood_pixels[1]):max(neighborhood_pixels[1]) + 1]
    
    # Generate the histogram of the neighborhood
    hist_channel_0 = cv2.calcHist([neighborhood_region], [0], None, [256], [0, 256])
    hist_channel_1 = cv2.calcHist([neighborhood_region], [1], None, [256], [0, 256])
    hist_channel_2 = cv2.calcHist([neighborhood_region], [2], None, [256], [0, 256])

    # Calculate the cumulative distribution function (CDF) for each channel
    cdf_b = hist_channel_0.cumsum() / hist_channel_0.sum()
    cdf_g = hist_channel_1.cumsum() / hist_channel_1.sum()
    cdf_r = hist_channel_2.cumsum() / hist_channel_2.sum()

    # Generate random values for each channel
    rand_vals_b = np.random.rand(mask.shape[0], mask.shape[1])
    rand_vals_g = np.random.rand(mask.shape[0], mask.shape[1])
    rand_vals_r = np.random.rand(mask.shape[0], mask.shape[1])

    # Map the random values to pixel values using the CDFs
    pixels_b = np.interp(rand_vals_b, cdf_b, np.arange(256))
    pixels_g = np.interp(rand_vals_g, cdf_g, np.arange(256))
    pixels_r = np.interp(rand_vals_r, cdf_r, np.arange(256))


    # Create the filled mask by replacing the masked pixels with the mapped values
    filled_mask = mask.copy()
    # Create an RGB array of the mapped pixel values
    rgb_pixels = np.stack((pixels_b.astype(np.uint8), pixels_g.astype(np.uint8), pixels_r.astype(np.uint8)), axis=-1)

    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    return cropped_image,mask,hist_channel_0,hist_channel_1,hist_channel_2

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
        cropped_frame,mask,hist0,hist1,hist2= crop_image(frame, clicked_points)
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
            save_directory = os.path.abspath(os.path.join(os.getcwd(), "..", "img"))  # Replace "desired_directory" with your desired directory name
            os.makedirs(save_directory, exist_ok=True)
            save_path = os.path.join(save_directory, "mask_{}.jpg".format(timestamp))
            cv2.imwrite(save_path,mask)
            print("Cropped image saved as:", save_path)
        else:
            print("Cannot save cropped image. Please enter crop mode and select at least 3 points.")
   

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
