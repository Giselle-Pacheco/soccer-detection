# ----------------------------------------------------------------
# 
# copy line to run:
#    'python main.py --video_file  ../../2023_05_05_14_59_37-ball-detection.mp4'
#    'python main.py --video_file  ../../2023_05_05_15_02_22-players-and-ball-detection.mp4'

# ----------------------------------------------------------------


import numpy as np
import cv2
import argparse
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt
import os

start_time = time.time()

sign_changes = 0



#Get the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Folder containing the images (one folder behind)
mask_folder = os.path.join(script_directory, '..', 'mask2')

# Get a list of all the image files in the folder
mask_list = os.listdir(mask_folder)

# Get the image path
mask_path = os.path.join(mask_folder, mask_list[0])
# Read the image
field_mask = cv2.imread(mask_path,0)

# Folder containing the images (one folder behind)
patch_folder=os.path.join(script_directory, '..', 'img2')

# Get a list of all the image files in the folder
patch_list = os.listdir(patch_folder)

#Create a list
product_list=list()
coor_list=list()

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

# Global variables
points = []
mask = None       

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global points, mask

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
        points.append((x, y))

        # Create a mask based on the contour
        if len(points) >= 25:
            mask = np.zeros(frame.shape[:2], np.uint8)
            pts = np.array(points, np.int32)
            cv2.fillPoly(mask, [pts], 255)
        drawing = True
        x_init, y_init = x, y
    
        
    # Check if the mouse is being moved while the left button is pressed
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        top_left_pt = (min(x_init, x), min(y_init, y))
        bottom_right_pt = (max(x_init, x), max(y_init, y))
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, color=(0, 255, 0), thickness=1)
        
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


# Iterate through each frame of the video
parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
args = parser.parse_args()

cv2.namedWindow('Video sequence',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video sequence', get_rectangle)

#multiprocess
num_processes = cpu_count()
pool = Pool(num_processes)

cap=cv2.VideoCapture(args.video_file)

def get_real_coordinates(x_obtained,y_obtained):
                
        cx=cap.get(3)/2
        cy=cap.get(4)/2
        f=7500
        z=50

        #Function that calculated the real coordinates
        coor_x,coor_y=x_obtained,y_obtained

        u=coor_x-cx
        v=cy-coor_y

        x_global=float((u/f)*z)
        y_global=float((-v/f)*z)  
        z_global=50

        print(x_global,y_global,x_obtained,y_obtained)

        coor_list.append((x_global,y_global))



def get_cross_product(list_of_object_coordinates):
    list_used=list_of_object_coordinates
    for i in range(1, len(list_used)):
        # Compute the cross product of the two consecutive elements
        cross_product = np.cross(list_used[i - 1], list_used[i])
        if cross_product != 0:
            product_list.append(cross_product)

    # Convert the array elements to floats and store them in a separate list
    cross_product_values = [float(item) for item in product_list]

    # Check for sign changes in the cross product values
    sign_changes = 0

    for i in range(1, len(cross_product_values)):
        if (cross_product_values[i - 1] >= 0 and cross_product_values[i] < 0) or (cross_product_values[i - 1] <= 0 and cross_product_values[i] > 0):
            # Sign change detected
            sign_changes += 1
              
    return sign_changes,cross_product_values


while(cap.isOpened()):

    #Got the current frame and pass on to 'frame'
    ret,frame=cap.read()

    #if the current frame cannot be capture, ret=0
    if not ret:
        print("frame missed!")
        break

    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

    field=cv2.bitwise_and(frame,frame,mask=field_mask)

    field_lab=cv2.cvtColor(field,cv2.COLOR_BGR2LAB)
    field_hls=cv2.cvtColor(field,cv2.COLOR_BGR2HLS)

    for index,image in enumerate(images):
        field=cv2.bitwise_and(field,field,mask=masks[index])

    field=cv2.medianBlur(field,3)
    
    field=cv2.GaussianBlur(field,(5,5),2)

    field_hsv=cv2.cvtColor(field,cv2.COLOR_BGR2HSV)
    
    green_segmentation=cv2.inRange(field_hsv,(34,26,110),(64,110,223))
    white_lines=cv2.inRange(field_hsv,(40,14,160),(66,53,250))
   
    green_segmentation=cv2.bitwise_or(white_lines,green_segmentation)

    kernel = np.ones((5,5))
    field = cv2.erode(green_segmentation,kernel,iterations=3)

    # Aplicar la dilatación
    field = cv2.morphologyEx(field,cv2.MORPH_CLOSE,kernel)

    contours,hierarchy  = cv2.findContours(field, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #array to storage the coordinates
    detected_objects_right = []
    detected_objects_left=[]

    #Defined area for objects
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<1700 and area>500:
            x, y, w, h = cv2.boundingRect(contour)
            if x>608:
                detected_objects_right.append((x, y, w, h))
            if x<608:
                detected_objects_left.append((x, y, w, h))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if area<500 and area>300:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Draw the circle on the output image
            cv2.circle(frame, center, radius, (0, 0, 255), 1)

    players_left=len(detected_objects_left)
    players_right=len(detected_objects_right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Derecha: {players_right}"
    text1=f"Izquierda: {players_left}"
    position = (640, 50)  # (x, y) coordinates of the starting point
    font_scale = 1
    color = (0, 0,255)  # BGR color format (green in this case)
    thickness = 1
    line_type = cv2.LINE_AA  # Anti-aliased line type
    cv2.putText(frame,text,position,font,font_scale,color,thickness,line_type)
    position=(50,50)
    cv2.putText(frame,text1,position,font,font_scale,color,thickness,line_type)

            #get_real_coordinates(x,y)

   
    #creating rectangles by coordinates.
    for rect in rectangles:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 255, 0), thickness=1)

        #Create 3 histograms for each R,G,B space color in the region selected        
        h_1,h_2,h_3=plot_histogram(field_hsv,rect[0], rect[1], rect[2], rect[3])

        #The intensity values of R,G,B accumulated in histograms 
        h_ch1_accumulated=h_ch1_accumulated+h_1
        h_ch2_accumulated=h_ch2_accumulated+h_2
        h_ch3_accumulated=h_ch3_accumulated+h_3

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # number = 10
    # text = f"Number: {number}"
    # position = (50, 50)  # (x, y) coordinates of the starting point
    # font_scale = 1.5
    # color = (0, 255, 0)  # BGR color format (green in this case)
    # thickness = 2
    # line_type = cv2.LINE_AA  # Anti-aliased line type
    # text=f"Number: {sign_changes}"
    # cv2.putText(frame,text,position,font,font_scale,color,thickness,line_type)


    
    # Visualise the input video
    cv2.imshow('Video sequence',frame)
    cv2.imshow('green_segmentation',green_segmentation)
    cv2.imshow('field',field)
    cv2.imshow('hsv',field_hsv)


    # The program finishes if the key 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        print("Programm finished, mate!")
        break

    # Draw the selected points
    for point in points:
        cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Create a copy of the frame
    frame_copy = frame.copy()

    # Apply the mask to the frame
    if mask is not None:
        frame_copy = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)


    # Save the selected region to a separate image when "S" key is pressed
    if mask is not None and cv2.waitKey(1) & 0xFF == ord('s'):
        selected_region = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imwrite('selected_region.png', selected_region)
        print("Selected region saved as 'selected_region.png'")

        # Reset the mask and points
        mask = None
        points = []

    # Iterate over the array elements
    sign_changes=0
    for i in range(1, len(coor_list)):
        if (coor_list[i-1][0] >= 0 and coor_list[i][0] < 0) or (coor_list[i-1][0] <= 0 and coor_list[i][0] > 0):
            # Sign change detected
            sign_changes += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all visualisation windows
cv2.destroyAllWindows()

# object_crossing, cross_product_values=get_cross_product(coor_list)

# Print the list  of ball position coordenates


#print(coor_list)

# Print the number of sign changes
print('Ball crossed:', sign_changes, ' times')


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

    


