import time
import datetime
import numpy as np
import socket
import pickle
import struct

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from libcamera import controls
import cv2
from sense_hat import SenseHat
import json

sense=SenseHat()
sense.clear()

ip_addr = "10.193.52.7" # Ashna's actual IP
port_num = 5001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # UDP
sock.connect((ip_addr, port_num))
conn = sock.makefile('wb')

picam2=Picamera2()  ## Create a camera object



dispW=1280
dispH=720
## Next, we configure the preview window size that determines how big should the image be from the camera, the bigger the image the more the details you capture but the slower it runs
## the smaller the size, the faster it can run and get more frames per second but the resolution will be lower. We keep 
picam2.preview_configuration.main.size= (dispW,dispH)  ## 1280 cols, 720 rows. Can also try smaller size of frame as (640,360) and the largest (1920,1080)
## with size (1280,720) you can get 30 frames per second

## since OpenCV requires RGB configuration we set the same format for picam2. The 888 implies # of bits on Red, Green and Blue
picam2.preview_configuration.main.format= "RGB888"
picam2.preview_configuration.align() ## aligns the size to the closest standard format
picam2.preview_configuration.controls.FrameRate=30 ## set the number of frames per second, this is set as a request, the actual time it takes for processing each frame and rendering a frame can be different

picam2.configure("preview")
## 3 types of configurations are possible: preview is for grabbing frames from picamera and showing them, video is for grabbing frames and recording and images for capturing still images.


picam2.start()

## Number of previous frames to update background model
num_history_frames = 8
## Gets the background subtractor object
back_sub = cv2.createBackgroundSubtractorMOG2(history=num_history_frames, varThreshold=25, detectShadows=False)
time.sleep(0.1)
max_foreground=200 # (0-255)
## Create a kernel for morphological operation to remove noise from binary images.
## You can tweak the dimensions of the kernel ## e.g. instead of 20,20 you can try 30,30.
## This creates a square matrix of 20x20 filled with ones, suitable for closing operations.
## Closing operations smoothen out a binary image by removing small holes/gaps in detected objects
kernel= np.ones((20,20), np.uint8)
img_counter = 0

while True:
    for event in sense.stick.get_events():
        if event.action == 'pressed':
            #tstart=time.time()

            frame = picam2.capture_array() ## frame is a large 2D array of rows and cols and at intersection of each point there is an array of three numbers for RGB i.e. [R,G,B] where RGB value ranges from 0 to 255
            #print(type(frame))
            
            
            fgmask= back_sub.apply(frame) ## obtains the foreground mask
            fgmask=cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            fgmask=cv2.medianBlur(fgmask,5) # Remove salt and pepper noise using the filter
            _,fgmask = cv2.threshold(fgmask, max_foreground, 255, cv2.THRESH_BINARY)
            
            # Find the contours of the object inside the binary image
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            areas=[cv2.contourArea(c) for c in contours]
            #print(f"len areas: {len(areas)}")
            
            # if there are no contours
            if len(areas) <10:
                cv2.imshow('Frame', frame)
                key=cv2.waitKey(1) & 0xFF ## wait for key press for 1 millisecond
                if key == ord("q"): ## stops for 1 ms to check if key Q is pressed
                    break
                # go to the top of the for loop
                continue
            else: # goes with "if len(areas)<1"
                # find the largest moving object in the frame
               max_index= np.argmax(areas)
               
            # Write frame to image file, classify, and update inventory
            cv2.imwrite('tmp_img.jpg', frame)
  
            cv2.imshow("Frame", frame)
                        # Send image frames to server
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, frame2 = cv2.imencode('.png', frame, encode_param)
            data = pickle.dumps(frame2, 0)
            

            if event.direction == "up":
                dir_bit = 1 # insert
            else:
                dir_bit = 0 # remove
            size = len(data) + 1
            sock.sendall(struct.pack(">L", size)  + struct.pack("<B", dir_bit) + data)
            img_counter += 1
            print(f"Sent image {img_counter} with direction {event.direction}")
            
            key=cv2.waitKey(1) & 0xFF

            
            if key ==ord(" "):
                cv2.imwrite("frame-" + str(time.strftime("%H:%M:%S", time.localtime())) + ".jpg", frame)
            if key == ord("q"): ## stops for 1 ms to check if key Q is pressed
                break
            
cv2.destroyAllWindows()
print("total images sent: ", img_counter)


