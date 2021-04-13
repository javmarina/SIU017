import os
import time

import numpy as np
from PIL import Image
import cv2 as cv

use_ellipse = False
count = 0


def print_image(im: np.array):
    global count
    filename_format = "{:d}_ellipse.png" if use_ellipse else "{:d}.png"
    Image.fromarray(im).convert("RGB").save(os.path.join("inertia_out", filename_format.format(count)))
    count += 1


def get_grip_points(img: np.array, contour, use_ellipse: bool = False):
    os.makedirs("inertia_out", exist_ok=True)

    t1 = time.time()

    contour_mask = np.zeros(img.shape[:2])
    cv.drawContours(contour_mask, [contour], -1, 255, -1)
    print_image(img)
    print_image(contour_mask)

    if use_ellipse:
        ellipse = cv.fitEllipse(contour)
        center, (d1, d2), angle_deg = ellipse
        angle_deg = 90 - angle_deg
        angle_rad = np.deg2rad(angle_deg)
        # Draw ellipse
        cv.ellipse(img, ellipse, color=(0, 255, 255), thickness=2)
        print_image(img)
    else:
        mat = np.argwhere(contour_mask != 0)
        mat[:, [0, 1]] = mat[:, [1, 0]]
        mat = np.array(mat).astype(np.float32)  # have to convert type for PCA

        m, e = cv.PCACompute(mat, mean=np.array([]))
        center = tuple(m[0])  # Note that center is the center of mass, not (max+min)/2 (geometric center)
        eig1 = e[0]
        angle_rad = np.arctan2(-eig1[1], eig1[0])

        cv.drawContours(img, contour, -1, (0, 255, 255), 1)
        print_image(img)

    xc, yc = center

    # draw circle at center
    cv.circle(img, (int(xc), int(yc)), 6, (255, 255, 255), -1)
    print_image(img)

    draw_line(img, -angle_rad, (xc, yc), color=(255, 255, 0), thickness=2)
    print_image(img)
    draw_line(img, -angle_rad - np.pi / 2, (xc, yc), color=(0, 255, 0), thickness=1)
    print_image(img)

    major_axis_bw = np.zeros_like(contour_mask)
    draw_line(major_axis_bw, -angle_rad, (xc, yc), color=(255, 255, 255), thickness=1)
    print_image(major_axis_bw)
    major_axis_bool = major_axis_bw == 255

    contour_outline_mask = np.zeros(img.shape[:2])
    cv.drawContours(contour_outline_mask, [contour], -1, 255, 1)
    print_image(contour_outline_mask)
    contour_mask_bool = contour_outline_mask == 255

    major_axis_intersect = np.logical_and(major_axis_bool, contour_mask_bool)
    print_image(np.logical_or(major_axis_bool, contour_mask_bool))
    print_image(major_axis_intersect)
    intersect_pt_1, intersect_pt_2 = np.fliplr(np.transpose(np.where(major_axis_intersect)))

    cv.line(img, tuple(intersect_pt_1), tuple(intersect_pt_2), color=(255, 255, 0), thickness=2)
    print_image(img)

    center = np.array(center)
    grip1 = (center + intersect_pt_1) // 2
    grip2 = (center + intersect_pt_2) // 2

    cv.circle(img, tuple(grip1.astype(np.int64)), 5, (255, 0, 255), -1)
    print_image(img)
    cv.circle(img, tuple(grip2.astype(np.int64)), 5, (255, 0, 255), -1)
    print_image(img)

    t2 = time.time()
    print("Time elapsed: {:.2f} ms".format((t2 - t1) * 1000))

    return img, grip1, grip2


def draw_line(im, angle_rad, point, *args, **kwargs):
    x, y = point
    m = np.tan(angle_rad)
    x1 = 0
    x2 = im.shape[1]
    y1 = m * (x1 - x) + y
    y2 = m * (x2 - x) + y
    cv.line(im, (x1, int(y1)), (x2, int(y2)), *args, **kwargs)


if __name__ == "__main__":
    bw_path = "im-tests/out/8_convex_hull/test7_normalized.png"
    pil_bw = Image.open(bw_path)
    bw = np.array(pil_bw)

    im_path = "im-tests/test7.jpg"
    pil_im = Image.open(im_path)
    im = np.array(pil_im)

    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    assert len(contours) == 1
    contour = contours[0]

    im, grip1, grip2 = get_grip_points(im, contour, use_ellipse=use_ellipse)

    cv.imshow(__file__, im)
    cv.waitKey()
