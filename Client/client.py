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

# Initialize SenseHat for joystick input
sense=SenseHat()
sense.clear()

# IP address and port number for socket communication
ip_addr = "10.193.52.7"
port_num = 5001

# Create socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # UDP
sock.connect((ip_addr, port_num))
conn = sock.makefile('wb')

# Create a camera object
picam2=Picamera2() 

dispW=1280
dispH=720

# Configure the preview window size
picam2.preview_configuration.main.size= (dispW,dispH)  

picam2.preview_configuration.main.format= "RGB888"
picam2.preview_configuration.align() 
picam2.preview_configuration.controls.FrameRate=30 

picam2.configure("preview")
picam2.start()

# Number of previous frames to update background model
num_history_frames = 8

# Background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=num_history_frames, varThreshold=25, detectShadows=False)
time.sleep(0.1)
max_foreground=200 # (0-255)

# Create a kernel for morphological operation 
kernel= np.ones((20,20), np.uint8)
img_counter = 0

while True:
    for event in sense.stick.get_events():
        if event.action == 'pressed':
            frame = picam2.capture_array() # Frame is a large 2D array 
 
            fgmask= back_sub.apply(frame) # Obtains the foreground mask
            fgmask=cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            fgmask=cv2.medianBlur(fgmask,5) # Remove salt and pepper noise using the filter
            _,fgmask = cv2.threshold(fgmask, max_foreground, 255, cv2.THRESH_BINARY)
            
            # Find the contours of the object inside the binary image
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            areas=[cv2.contourArea(c) for c in contours]
            
            # If there are less than 10 contours, display frame and continue
            if len(areas) <10:
                cv2.imshow('Frame', frame)
                key=cv2.waitKey(1) & 0xFF
                if key == ord("q"): 
                    break
                continue
            else: 
                # Find the largest moving object in the frame
               max_index= np.argmax(areas)
               
            # Write frame to image file, 
            cv2.imwrite('tmp_img.jpg', frame)
            cv2.imshow("Frame", frame)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, frame2 = cv2.imencode('.png', frame, encode_param)
            data = pickle.dumps(frame2, 0)

            # Determine direction of motion
            if event.direction == "up":
                dir_bit = 1 # Insert
            else:
                dir_bit = 0 # Remove

            # Send image frame to server
            size = len(data) + 1
            sock.sendall(struct.pack(">L", size)  + struct.pack("<B", dir_bit) + data)
            img_counter += 1
            print(f"Sent image {img_counter} with direction {event.direction}")
            
            key=cv2.waitKey(1) & 0xFF
            
            if key ==ord(" "):
                cv2.imwrite("frame-" + str(time.strftime("%H:%M:%S", time.localtime())) + ".jpg", frame)
            if key == ord("q"):
                break
            
cv2.destroyAllWindows()
print("Total images sent: ", img_counter)


