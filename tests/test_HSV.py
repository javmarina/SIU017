"""
2020 - Francisco J Moreno Rodríguez
"""
import io

import cv2 as cv
import numpy as np
from PIL import Image

from RobotHttpInterface import RobotHttpInterface
from RobotModel import RobotModel

##############
#  TEST HSV  #
##############

max_value = 255
max_value_H = 360 // 2  # 180 degrees
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Simulator capture'
window_detection_name = 'HSV thresholding testing'
low_H_name = 'Low Hue'
low_S_name = 'Low Sat'
low_V_name = 'Low Val'
high_H_name = 'High Hue'
high_S_name = 'High Sat'
high_V_name = 'High Val'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = min(high_H - 1, val)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = max(val, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = min(high_S - 1, val)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = max(val, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = min(high_V - 1, val)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = max(val, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


def test_thresholding_HSV(im):
    # init thresholding interface
    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)
    cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)

    # Modify threshold ranges and show
    im_th = cv.inRange(im, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # im_th = morph_op_for_object(im_th)
    im_rgb = cv.cvtColor(im, cv.COLOR_HSV2RGB)
    """
    contours, hierarchy = cv.findContours(im_th, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    hull = []
    for i in range(len(contours)):
        hull.append(cv.convexHull(contours[i]))
    im2 = np.zeros_like(im_rgb)
    cv.drawContours(im2, hull, -1, (255, 0, 0), -1, 8)
    mask = im2[:, :, 0]
    """

    # cv.imshow('mask', mask)
    # cv.imshow('contours', im2)
    cv.imshow(window_capture_name, im_rgb)
    cv.imshow(window_detection_name, im_th)


def test_HSV():
    """
    Testing function for easily find HSV thresholds to detect different objects,
    like G500_2, BLUEROV and the green pipe.
    """
    robot_model = RobotModel.GIRONA_500_1
    http_interface = RobotHttpInterface(robot_model)
    while True:
        response_content = http_interface.get_image_udp()
        im = np.array(Image.open(io.BytesIO(response_content)))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.GaussianBlur(im, (9, 9), 0)
        im = cv.cvtColor(im, cv.COLOR_RGB2HSV)
        test_thresholding_HSV(im)

        if cv.waitKey(16) & 0xff == ord('q'):
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    test_HSV()
