import socket
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def receive_all(sock, size):
    buffer = bytearray()
    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))
        if not chunk:
            return None
        buffer.extend(chunk)
    return buffer

server_host = "84.237.21.36"
server_port = 5152
status = None

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((server_host, server_port))
    plt.ion()
    plt.figure()

    while status != b"yep":
        client.send(b"get")
        data = receive_all(client, 40002)

        status = b"nope"

        img = np.frombuffer(data[2:40002], dtype="uint8").reshape(data[0], data[1])

        binarized = img > 0
        labeled_img = label(binarized)
        shapes = regionprops(labeled_img)
        if len(shapes) == 2:
            y1, x1 = shapes[0].centroid
            y2, x2 = shapes[1].centroid
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            client.send(f"{distance:.1f}".encode())
            print(client.recv(10))
            client.send(b"beat")
            status = client.recv(10)
            plt.clf()
            plt.subplot(121)
            plt.imshow(img)
            plt.pause(1)