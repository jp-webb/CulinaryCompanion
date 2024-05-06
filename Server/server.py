import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import foodClassifier
import inventory

# Source code: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
HOST="0.0.0.0"
PORT=5001

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

# Accept incoming connections
conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        # print("Recv: {}".format(len(data)))
        data += conn.recv(4096)
    print("Done Recv: {}".format(len(data)))

    # Extract the size of the incoming message
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))

    # Receive the rest of the message
    while len(data) < msg_size:
        data += conn.recv(4096)

    # Extract direction and frame data
    direction_byte = data[0:1]
    direction = struct.unpack("<B", direction_byte)[0]

    frame_data = data[1:msg_size]

    print("Direction =", (direction))
    data = data[msg_size:]

    # Decode the frame data and display the image
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow('ImageWindow',frame)
    cv2.imwrite('tmp_img.jpg', frame)

    # Classify the image using the food classifier
    label = foodClassifier.classify('tmp_img.jpg')
    print(label)
    if direction:
        inventory.update_inventory(label, True)
    else:
        inventory.update_inventory(label, False)

    # Display the updated inventory
    inventory.get_inventory()
    cv2.waitKey(1)