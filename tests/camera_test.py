import socket
import numpy as np
import cv2 as cv

"""
http://localhost:8001/setparams?width=200&quality=100
http://localhost:8011/setparams?width=200&quality=100
http://localhost:8021/setparams?width=200&quality=100
"""

if __name__ == "__main__":
    UDP_IP = "127.0.0.1"
    UDP_PORT = 8001

    print("UDP target IP: {:s}".format(UDP_IP))
    print("UDP target port: {:d}".format(UDP_PORT))

    bufferSize = 60000
    width = 640
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP socket

    for x in range(91):
        msg = "/setparams?width={:d}&quality={:d}".format(width, x)
        MESSAGE = msg.encode()
        # print("message: %s" % MESSAGE)

        sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

        datagramFromClient, datagramSourceAddress = sock.recvfrom(bufferSize)
        print(len(datagramFromClient))

        image = np.asarray(bytearray(datagramFromClient), dtype=np.uint8)
        image = cv.imdecode(image, cv.IMREAD_COLOR)

        cv.putText(
            img=image,
            text=str(x),
            org=(50, 50),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv.LINE_AA)

        cv.imshow("image", image)
        cv.waitKey(100)
