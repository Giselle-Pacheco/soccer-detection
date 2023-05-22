# ----------------------------------------------------------------
# 
# copy line to run:
#    'python main.py --video_file  ../../2023_05_05_14_59_37-ball-detection.mp4'
# ----------------------------------------------------------------



import numpy as np
import cv2
import argparse
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt
import os

start_time = time.time()

#Get the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Folder containing the images (one folder behind)
mask_folder = os.path.join(script_directory, '..', 'mask')

# Get a list of all the image files in the folder
mask_list = os.listdir(mask_folder)

# Get the image path
mask_path = os.path.join(mask_folder, mask_list[0])
# Read the image
field_mask = cv2.imread(mask_path,0)

# Folder containing the images (one folder behind)
patch_folder=os.path.join(script_directory, '..', 'img')

# Get a list of all the image files in the folder
patch_list = os.listdir(patch_folder)

# Create an empty list to store the images
images = []
masks=[]

# Iterate over each file
for file_name in patch_list:
    # Construct the full file path
    file_path = os.path.join(patch_folder, file_name)

    # Read the image using cv2.imread
    image = cv2.imread(file_path)

    # Check if the image was successfully read
    if image is not None:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask=cv2.bitwise_not(mask)

        # Append the image to the list
        images.append(image)
        masks.append(mask)
    
    else:
        print(f'Failed to read image: {file_path}')



h_ch1_accumulated = np.zeros((256, 1), dtype=np.float32)
h_ch2_accumulated = np.zeros((256, 1), dtype=np.float32)
h_ch3_accumulated = np.zeros((256, 1), dtype=np.float32)

#Save the drawn rectangles
rectangles=[]

#Command to maintain the drawn rectangles while you keep drawing rectangles
drawing=False

#Initialise array of coordinates of drawn rectangles
top_left_pt=None
bottom_right_pt=None   

def plot_histogram(frame_used,Xinit,Yinit,Xfin,Yfin):

    #Changing the region selected to HSV
    roi=frame_used[Yinit:Yfin,Xinit:Xfin]

    hist_channel_1=cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_channel_2 = cv2.calcHist([roi], [1], None, [256], [0, 256])
    hist_channel_3 = cv2.calcHist([roi], [2], None, [256], [0, 256])

    return hist_channel_1,hist_channel_2,hist_channel_3

def get_rectangle(event,x,y,flags,params):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt
    
    # Check if the left mouse button was pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
    
        
    # Check if the mouse is being moved while the left button is pressed
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        top_left_pt = (min(x_init, x), min(y_init, y))
        bottom_right_pt = (max(x_init, x), max(y_init, y))
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, color=(0, 255, 0), thickness=1)
        cv2.imshow('Video sequence', frame)
        
    # Check if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangles.append((x_init, y_init, x, y)) 
        
def apply_gaussian_filter(frame,kernel,sigma):
    # Apply a Gaussian filter with a kernel size of 5x5 and sigma value of 1
    filtered_frame = cv2.GaussianBlur(frame, (kernel, kernel), sigma)
    return filtered_frame

def apply_median_filter(frame,kernel):
    filtered_frame=cv2.medianBlur(frame,kernel)
    return filtered_frame


parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
args = parser.parse_args()

cv2.namedWindow('Video sequence',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video sequence', get_rectangle)

#multiprocess
num_processes = cpu_count()
pool = Pool(num_processes)

cap=cv2.VideoCapture(args.video_file)

while(cap.isOpened()):

    #Got the current frame and pass on to 'frame'
    ret,frame=cap.read()

    #if the current frame cannot be capture, ret=0
    if not ret:
        print("frame missed!")
        break
   
    field=cv2.bitwise_or(frame,frame,mask=field_mask)
    field_luv=cv2.cvtColor(field,cv2.COLOR_BGR2LUV)

    for index,image in enumerate(images):  
        field=cv2.bitwise_and(field,field,mask=masks[index])+image

    field=cv2.medianBlur(field,3)
    
    field=cv2.GaussianBlur(field,(5,5),3)

    field_hsv=cv2.cvtColor(field,cv2.COLOR_BGR2HSV)

    green_segmentation=cv2.inRange(field_hsv,(34,29,110),(64,110,221))

    result=cv2.erode(green_segmentation,np.ones((5,5)),iterations=2)

    contours,hierarchy  = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # #array to storage the coordinates
    detected_objects = []

    #Defined area for objects
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<240:
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))

    for x, y, w, h in detected_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

   
    #creating rectangles by coordinates.
    for rect in rectangles:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 255, 0), thickness=1)

        #Create 3 histograms for each R,G,B space color in the region selected        
        h_1,h_2,h_3=plot_histogram(field_hsv,rect[0], rect[1], rect[2], rect[3])

        #The intensity values of R,G,B accumulated in histograms 
        h_ch1_accumulated=h_ch1_accumulated+h_1
        h_ch2_accumulated=h_ch2_accumulated+h_2
        h_ch3_accumulated=h_ch3_accumulated+h_3


    # Visualise the input video
    cv2.imshow('Video sequence',frame)
    cv2.imshow('field',field)
    cv2.imshow('result',result)
    #cv2.imshow('LUB_FRAME',LUV_FRAME)
    #cv2.imshow('hsv_FRAME',HSV_FRAME)


    # The program finishes if the key 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        print("Programm finished, mate!")
        break

# Destroy all visualisation windows
cv2.destroyAllWindows()

# Destroy 'VideoCapture' object
cap.release()
plt.figure(num=1)
plt.plot(h_ch1_accumulated,color='red')
plt.plot(h_ch2_accumulated,color='green')
plt.plot(h_ch3_accumulated,color='blue') 
plt.xlim([0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend(['H','S','V'])
plt.show()

end_time = time.time()

total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')

    


