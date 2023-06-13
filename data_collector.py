import cv2
import numpy as np
import time
import os
import itertools
from screeninfo import get_monitors
import mtcnn
import random

# Define input and output directories.
project_dir = './'
images_dir = project_dir + 'images/'
# Define both what someone is looking at (either full circle or crosshair) andthe circle color (either random or a static red)
look_at = "crosshair" #Use "circle" or "crosshair"
color = "static" #Use "static" or "random"
# Define resolution of webcam.
webcam_width = 1280
webcam_height = 720
# Set the crop size of the eye image, note that this is dependand on your webcam resolution!
eye_box_shape = 50

# Create instance of MTCNN face detector.
face_detector = mtcnn.MTCNN()

# Set background as a grayscale.
background_value = np.random.randint(1, 256)
bg_color = (background_value, background_value, background_value) #BGR

if (color == "static"):
    circle_color = (0, 0, 255) #BGR
if (color == "random"):
    circle_color = (np.random.randint(1, 256), np.random.randint(1, 256), np.random.randint(1, 256)) #BGR
if (look_at == "crosshair"):
    circle_radius = 20
    circle_thickness = 2
if (look_at == "circle"):
    circle_radius = 20
    circle_thickness = -1

# Add failsafe that one monitor is currently retrieved.
assert (len(get_monitors()) == 1)

# Retrieve width and height of the monitor.
width = get_monitors()[0].width
height = get_monitors()[0].height

# Define a full-sized screen for the test setup.
cv2.namedWindow("Screen", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Define a background for the full resolution with the set background color.
bg = np.ones((height, width, 3), dtype=np.uint8)
bg[:, :, 0] = bg[:, :, 0]*bg_color[0]
bg[:, :, 1] = bg[:, :, 1]*bg_color[1]
bg[:, :, 2] = bg[:, :, 2]*bg_color[2]

# Based on the resolution and the circle radius, compute all positions the circle can be shown at.
all_coordinates = list(itertools.product(np.arange(circle_radius, width - circle_radius, dtype=int), np.arange(circle_radius, height - circle_radius, dtype=int)))

# Iterate over all the current folders in the output directory.
current_ids = [name for name in os.listdir(images_dir) if os.path.isdir(images_dir + name)]
current_coors = []

# From the folder names, retrieve the x and y position coordinates.
for entree in current_ids:
    y_data = tuple(map(int, (entree.split('_')[0:2])))
    current_coors.append(y_data)

# Make the valid_coordinates the currently unsampled screen coordinates.
valid_coordinates = [coor for coor in all_coordinates if coor not in current_coors]

# If every point has been sampled once, use all points as valid points.
if(len(valid_coordinates) == 0):
    valid_coordinates = all_coordinates.copy()

# Define a initial circle coordinate from the valid_coordinates.
circle_coor = random.choice(valid_coordinates)
circle_coor_x = circle_coor[0]
circle_coor_y = circle_coor[1]

# Define initial screen with sampled circle position on background.
img = cv2.circle(bg.copy(), (circle_coor_x, circle_coor_y), circle_radius, circle_color, circle_thickness)
if (look_at == "crosshair"):
    img = cv2.line(img, (circle_coor_x, circle_coor_y - circle_radius - 5), (circle_coor_x, circle_coor_y + circle_radius + 5), circle_color, circle_thickness) 
    img = cv2.line(img, (circle_coor_x - circle_radius - 5, circle_coor_y), (circle_coor_x  + circle_radius + 5, circle_coor_y), circle_color, circle_thickness) 

# Establish connection with the webcam.
cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[
    cv2.CAP_PROP_FRAME_WIDTH, webcam_width,
    cv2.CAP_PROP_FRAME_HEIGHT, webcam_height])

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam...")

while True:
    # Retrieve frame from webcam and show test environment.
    ret, frame = cap.read()
    cv2.imshow('Screen', img)

    # Wait for user input, either ESC (quit environment) or SPACEBAR (take picture).
    c = cv2.waitKey(1)
    if c == 27: #ESC
        break
    elif c == 32: #SPACEBAR
    
        # Get the current time and date, and detect a face in the current frame.
        now = time.strftime("%Y%m%d-%H%M%S")
        face_roi = face_detector.detect_faces(frame)
        
        # If there is one detected face in the frame:
        if (len(face_roi) == 1):
            
            # If both the left_eye and the right_eye are in the available keypoints:
            if(('left_eye' in list(face_roi[0]['keypoints'])) and ('right_eye' in list(face_roi[0]['keypoints']))):
                
                # Retrieve cropped face from frame.
                face_x1, face_y1, face_width, face_height = face_roi[0]['box']
                face_x2, face_y2 = face_x1 + face_width, face_y1 + face_height
                face = frame[face_y1:face_y2, face_x1:face_x2]
                
                # Compute face grid as a white image with a black image where the face was.
                face_pos = 255*np.ones((frame.shape[0], frame.shape[1] , 1))
                face_pos[face_y1:face_y2, face_x1:face_x2] = 0
                
                # Retrieve left eye from frame.
                l_eye_x1, l_eye_y1 = face_roi[0]['keypoints']['left_eye'][0] - eye_box_shape//2, face_roi[0]['keypoints']['left_eye'][1] - eye_box_shape//2
                l_eye_x2, l_eye_y2 = face_roi[0]['keypoints']['left_eye'][0] + eye_box_shape//2, face_roi[0]['keypoints']['left_eye'][1] + eye_box_shape//2
                l_eye = frame[l_eye_y1:l_eye_y2, l_eye_x1:l_eye_x2]
                
                # Retrieve right eye from frame.
                r_eye_x1, r_eye_y1 = face_roi[0]['keypoints']['right_eye'][0] - eye_box_shape//2, face_roi[0]['keypoints']['right_eye'][1] - eye_box_shape//2
                r_eye_x2, r_eye_y2 = face_roi[0]['keypoints']['right_eye'][0] + eye_box_shape//2, face_roi[0]['keypoints']['right_eye'][1] + eye_box_shape//2
                r_eye = frame[r_eye_y1:r_eye_y2, r_eye_x1:r_eye_x2]
                
                # Create a new foldes to store the images, name the folder based on screen position and time and date of when the image was taken.
                new_folder_name = str(circle_coor_x) + "_" + str(circle_coor_y) + "_" + str(now)
                os.makedirs(('./images/' + new_folder_name), exist_ok=True)
                
                # Write the images to the newly created folder.
                cv2.imwrite('./images/' + new_folder_name + "/face.png", cv2.resize(face, (64, 64)))
                cv2.imwrite('./images/' + new_folder_name + "/grid.png", cv2.resize(face_pos, (25, 25)))
                cv2.imwrite('./images/' + new_folder_name + "/left_eye.png", cv2.resize(l_eye, (64, 64)))
                cv2.imwrite('./images/' + new_folder_name + "/right_eye.png", cv2.resize(r_eye, (64, 64)))
                
                # Remove sampled coordinate from the valid_coordinates.
                valid_coordinates.remove(circle_coor)
        
        # If the circle color is set to random, pick a new color.
        if (color == "random"):
            circle_color = (np.random.randint(100, 256), np.random.randint(100, 256), np.random.randint(100, 256)) #BGR
        
        # If the valid_coordinates list is empty, set it as all the coordinates.
        if (len(valid_coordinates) == 0):
            valid_coordinates = all_coordinates
            
        # Sample a new circle position.
        circle_coor = random.choice(valid_coordinates)
        circle_coor_x = circle_coor[0]
        circle_coor_y = circle_coor[1]
        
        # Define a new greyscale value for the background.
        background_value = np.random.randint(1, 256)
        bg_color = (background_value, background_value, background_value) #BGR
        
        # Define a background for the full resolution with the set background color.
        bg = np.ones((height, width, 3), dtype=np.uint8)
        bg[:, :, 0] = bg[:, :, 0]*bg_color[0]
        bg[:, :, 1] = bg[:, :, 1]*bg_color[1]
        bg[:, :, 2] = bg[:, :, 2]*bg_color[2]
        
        # Set background with circle as the new testing image,
        img = cv2.circle(bg.copy(), (circle_coor_x, circle_coor_y), circle_radius, circle_color, circle_thickness)
        
        # If the circle is a crosshair specifically, also add a horizontal and vertical line.
        if (look_at == "crosshair"):
            img = cv2.line(img, (circle_coor_x, circle_coor_y - circle_radius - 5), (circle_coor_x, circle_coor_y + circle_radius + 5), circle_color, circle_thickness)
            img = cv2.line(img, (circle_coor_x - circle_radius - 5, circle_coor_y), (circle_coor_x  + circle_radius + 5, circle_coor_y), circle_color, circle_thickness) 

# If ESC has been pressed, stop the connection with the webcam and destroy the testing window.
cap.release()
cv2.destroyAllWindows()