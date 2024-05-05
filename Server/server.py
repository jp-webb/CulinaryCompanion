import socket

# Binding to 0.0.0.0 allows UDP connections to any address
# that the device is using
# import cv2 
# import numpy as np
# UDP_IP = "0.0.0.0"
# UDP_PORT = 5000

# sock = socket.socket(socket.AF_INET, # Use Internet socket family
#                      socket.SOCK_STREAM) # Use UDP packets
# addr = (UDP_IP, UDP_PORT)
# sock.bind(addr)
# print("Socket is bound to", addr)

# while True:
#     data, addr = sock.recvfrom(1024)
#     print("Received message:", data)

#     nparr = np.frombuffer(data, np.uint8)
#     # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # cv2.imshow("Received Image", image)

#     print("Image received and saved.")

import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import foodClassifier
import inventory

# https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
HOST="0.0.0.0"
PORT=5001

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        # print("Recv: {}".format(len(data)))
        data += conn.recv(4096)
    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)

    direction_byte = data[0:1]
    direction = struct.unpack("<B", direction_byte)[0]

    frame_data = data[1:msg_size]

    # frame_data = frame_data[:-1]
    print("Direction =", (direction))
    data = data[msg_size:]

    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow('ImageWindow',frame)
    cv2.imwrite('tmp_img.jpg', frame)
    label = foodClassifier.classify('tmp_img.jpg')
    print(label)
    if direction:
        inventory.update_inventory(label, True)
    else:
        inventory.update_inventory(label, False)
    inventory.get_inventory()
    cv2.waitKey(1)