import io
import json
import os
import time

import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from RobotHttpInterface import RobotHttpInterface
from RobotModel import RobotModel


def get_contour_eccentricity(contour):
    (_, _), (ma, MA), _ = cv.fitEllipse(contour)
    return np.sqrt(1 - (ma / MA) ** 2)


def is_touching_border(bb):
    return bb["x1"] <= 0.01 or bb["y1"] <= 0.01 or bb["x2"] >= 0.99 or bb["y2"] >= 0.99


def bb2Points(bb, target_shape):
    pt1 = (int(bb["x1"]*target_shape[1]), int(bb["y1"]*target_shape[0]))
    pt2 = (int(bb["x2"]*target_shape[1]), int(bb["y2"]*target_shape[0]))
    return pt1, pt2


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def detect_tube(img: np.array):
    # Gaussian filter
    filtered = cv.GaussianBlur(img, (9, 9), 1)

    # Saturation thresholding (with red color elimination)
    hsv = cv.cvtColor(filtered, cv.COLOR_RGB2HSV)
    bw = cv.inRange(hsv, (25, 84, 76), (180, 211, 255))

    # Close
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)

    # Find & filter contours
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    big_contours = filter(lambda cnt: cv.contourArea(cnt) > 100, contours)
    tuples = map(lambda cnt: (cnt, get_contour_eccentricity(cnt)), big_contours)
    higher_095 = list(filter(lambda tupl: tupl[1] >= 0.95, tuples))

    # Select tube, if found
    tube = None
    if len(higher_095) > 0:
        tube = max(higher_095, key=lambda tupl: tupl[1])[0]
        # Return convex hull, which should be more robust and use less memory
        tube = cv.convexHull(tube)

    return tube


def go_to_depth(robot_http_interface: RobotHttpInterface, depth):
    _, _, z = robot_http_interface.get_position()
    while abs(z-depth) > 0.01:
        robot_http_interface.set_velocity(0, 0, 0.25*(depth-z)/abs((depth-z)), 0, 100)
        _, _, z = robot_http_interface.get_position()
    robot_http_interface.stop()


def main():
    robot_model = RobotModel.GIRONA_500_1
    robot_http_interface = RobotHttpInterface(robot_model)

    depth_start = 1
    depth_step = 0.5

    current_depth = depth_start
    go_to_depth(robot_http_interface, current_depth)

    # Move
    """target = [-2.25449627638, 1.83655786514]
    Kp = 0.08
    Ki = 0
    Kd = 0
    pids = [
        PID(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=target[0], output_limits=[-0.5, 0.5]),
        PID(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=target[1], output_limits=[-0.5, 0.5])
    ]

    while True:
        p = robot_http_interface.get_position()
        dist = np.sqrt((p[0]-target[0])**2 + (p[1]-target[1])**2)
        print(dist)
        if dist < 0.05:
            break

        v = [0.0, 0.0, 0.0]
        for i, pid, val in zip([1, 0], pids, p[:2]):
            v[i] = pid(val)
        v[0] *= -1
        v[1] *= -1
        robot_http_interface.set_velocity(x=v[0], y=v[1], z=v[2], az=0, percentage=100)"""

    widths = [320, 480, 640]
    qualities = list(range(91))[::-1]

    real_shapes = []
    for i in range(len(widths)):
        real_shapes.append(None)
    results = []
    times = []
    sizes = []

    count = 0
    depths = []
    finished = False
    while True:
        partial_results = np.zeros((len(widths), len(qualities)))
        partial_times = np.zeros((len(widths), len(qualities)))
        partial_sizes = np.zeros((len(widths), len(qualities)))

        for i, width in enumerate(tqdm(widths)):
            for j, quality in enumerate(qualities):

                dt = 0
                t = time.time()
                content = robot_http_interface.get_image_udp(width, quality)
                im = np.array(Image.open(io.BytesIO(content)))
                dt += time.time() - t
                im_full = np.array(Image.open(io.BytesIO(robot_http_interface.get_image())))

                partial_sizes[i, j] = len(content)

                if real_shapes[i] is None:
                    real_shapes[i] = im.shape

                t = time.time()
                tube = detect_tube(im)
                dt += time.time() - t
                partial_times[i, j] = dt
                if tube is None:
                    partial_results[i, j] = -1
                    continue

                tube_full = detect_tube(im_full)
                if tube_full is None:
                    partial_results[i, j] = -2
                    continue

                x1, y1, w1, h1 = cv.boundingRect(tube)
                bb1 = {
                    "x1": x1/im.shape[1],
                    "y1": y1/im.shape[0],
                    "x2": (x1 + w1)/im.shape[1],
                    "y2": (y1 + h1)/im.shape[0]
                }
                x2, y2, w2, h2 = cv.boundingRect(tube_full)
                bb2 = {
                    "x1": x2/im_full.shape[1],
                    "y1": y2/im_full.shape[0],
                    "x2": (x2 + w2)/im_full.shape[1],
                    "y2": (y2 + h2)/im_full.shape[0]
                }
                partial_results[i, j] = get_iou(bb1, bb2)

                if is_touching_border(bb2):
                    finished = True
                    break

                # https://stackoverflow.com/questions/22314949/compare-two-bounding-boxes-with-each-other-matlab
                # If you have a bounding box detection and a ground truth bounding box, then the area of overlap
                # between them should be greater than or equal to 50%

                cv.putText(
                    img=im,
                    text=str(quality),
                    org=(50, 50),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv.LINE_AA)
                pt1, pt2 = bb2Points(bb1, im.shape)
                cv.rectangle(im, pt1, pt2, (0, 255, 0) if partial_results[i, j] > 0.5 else (0, 0, 255), 2)

                pt1, pt2 = bb2Points(bb2, im.shape)
                cv.rectangle(im, pt1, pt2, (255, 0, 0), 2)

                # cv.imshow("Image", im)
                # cv.waitKey(1)

            if finished:
                break

        if not finished:
            results.append(partial_results.tolist())
            times.append(partial_times.tolist())
            sizes.append(partial_sizes.tolist())
            count += 1

        _, _, z = robot_http_interface.get_position()
        print("Depth {:.2f} done (actual value is {:.2f})".format(current_depth, z))
        depths.append(z)

        if z >= 10.7:
            finished = True

        if finished:
            break
        else:
            current_depth += depth_step
            go_to_depth(robot_http_interface, current_depth)

    dict_results = {
        "results": results,
        "times": times,
        "sizes": sizes,
        "depths": depths,
        "widths": widths,
        "real_shapes": real_shapes,
        "qualities": qualities
    }

    path = "quality_results.txt"
    if os.path.exists(path):
        os.remove(path)
    with open(path, "x") as f:
        json.dump(obj=dict_results, fp=f)

    print("Finished!")


def analyze():
    path = "quality_results.txt"
    with open(path, "r") as f:
        dict_results = json.load(f)

    def print_figure(name, format: str = "svg"):
        path = "quality_test_out"
        os.makedirs(path, exist_ok=True)
        plt.show(block=False)
        plt.savefig(fname=os.path.join(path, name + "." + format), format=format)
        plt.close()

    # TODO
    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # matplotlib.rc('text', usetex=True)

    results = np.array(dict_results["results"])
    times = np.array(dict_results["times"])
    sizes = np.array(dict_results["sizes"])
    depths = dict_results["depths"]
    widths = dict_results["widths"]
    real_shapes = dict_results["real_shapes"]
    qualities = dict_results["qualities"]

    # Axes: depth, width, quality

    filtered_results = results.copy()
    filtered_results[filtered_results < 0] = 0.0

    hits = filtered_results >= 0.5

    # IoU vs. depth (for each width)
    selected_qualities = [90, 75, 50, 25]
    for selected_quality in selected_qualities:
        i = qualities.index(selected_quality)
        partial = 100*filtered_results[:, :, i]  # (depths, widths)
        plt.plot(depths, partial)
        plt.legend(["{:d}x{:d}".format(width, height) for height, width, _ in real_shapes])
        plt.xlabel("Depth (m)")
        plt.ylabel("IoU (%)")
        plt.ylim([0, 100])
        plt.grid()
        plt.title("IoU vs. depth (quality = {:d}%)".format(selected_quality))
        print_figure("iou_vs_depth_q{:d}".format(selected_quality))

    # IoU vs. depth for each selected quality (for each width)
    selected_qualities = [90, 75, 50, 25]
    for i, (width, height, _) in enumerate(real_shapes):
        partial = 100 * filtered_results[:, i, [qualities.index(q) for q in selected_qualities]]
        plt.plot(depths, partial)
        plt.legend(["{:d}%".format(quality) for quality in selected_qualities])
        plt.xlabel("Depth (m)")
        plt.ylabel("IoU (%)")
        plt.ylim([0, 100])
        plt.grid()
        plt.title("{:d}x{:d}".format(width, height))
        print_figure("iou_vs_depth_{:d}x{:d}".format(width, height))

    # Average IoU over all depths, vs. quality (for each width)
    for i, (width, height, _) in enumerate(real_shapes):
        partial = 100*np.mean(filtered_results[:, i, :], 0)
        plt.plot(qualities, partial)
        plt.title("{:d}x{:d}".format(width, height))
        plt.xlabel("Quality (%)")
        plt.ylabel("Average IoU (%)")
        plt.ylim([0, 100])
        plt.grid()
        print_figure("avg_iou_vs_depth_{:d}x{:d}".format(width, height))

    # Average accuracy vs. depth, over qualities (for each width)
    partial = 100*np.mean(hits, 2)
    plt.plot(depths, partial)
    plt.title("Average accuracy")
    plt.legend(["{:d}x{:d}".format(width, height) for height, width, _ in real_shapes])
    plt.xlabel("Depth (m)")
    plt.ylabel("Average accuracy (%)")
    plt.ylim([0, 100])
    plt.grid()
    print_figure("avg_acc_vs_depth")

    # Average processing time vs. quality (for each width)
    partial = 1000*np.mean(times, 0).transpose()
    plt.plot(qualities, partial)
    plt.title("Average processing time")
    plt.legend(["{:d}x{:d}".format(width, height) for height, width, _ in real_shapes])
    plt.xlabel("Quality (%)")
    plt.ylabel("Average processing time (ms)")
    plt.grid()
    print_figure("avg_time_vs_depth")

    # Average size (bytes) vs. resolution and quality
    partial = 1000*np.mean(sizes, 0).transpose()/(1024**2)
    plt.plot(qualities, partial)
    plt.title("Average size (MB)")
    plt.legend(["{:d}x{:d}".format(width, height) for height, width, _ in real_shapes])
    plt.xlabel("Quality (%)")
    plt.ylabel("Average size (MB)")
    plt.grid()
    print_figure("avg_bytes_vs_depth")


if __name__ == "__main__":
    # main()
    analyze()
