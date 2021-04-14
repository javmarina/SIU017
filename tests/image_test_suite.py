import glob
import os
import shutil

import cv2 as cv
import numpy as np
from PIL import Image

import visualPercepUtils as vpu


def get_contour_eccentricity(contour):
    (_, _), (ma, MA), _ = cv.fitEllipse(contour)
    return np.sqrt(1-(ma/MA)**2)


# def filter(orig: np.array, last_modified: np.array) -> np.array:
#     return cv.GaussianBlur(last_modified, (17, 17), 0)


def hue(orig: np.array, last_modified: np.array) -> np.array:
    hsv = cv.cvtColor(orig, cv.COLOR_RGB2HSV)
    return hsv[:, :, 0]


def saturation(orig: np.array, last_modified: np.array) -> np.array:
    hsv = cv.cvtColor(orig, cv.COLOR_RGB2HSV)
    return hsv[:, :, 1]


def value(orig: np.array, last_modified: np.array) -> np.array:
    hsv = cv.cvtColor(orig, cv.COLOR_RGB2HSV)
    return hsv[:, :, 2]


def threshold_sat(orig: np.array, last_modified: np.array) -> np.array:
    filtered = cv.GaussianBlur(orig, (9, 9), 0)
    hsv = cv.cvtColor(filtered, cv.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    max_sat = np.max(sat)
    mean_sat = np.mean(sat)
    th_sat = (max_sat+mean_sat)/2 - 15

    # ret, th = cv.threshold(sat, th_sat, 255, cv.THRESH_BINARY)
    return cv.inRange(hsv, (10, th_sat, 0), (180, 255, 255))


def threshold_sat_adaptive(orig: np.array, last_modified: np.array) -> np.array:
    filtered = cv.GaussianBlur(orig, (17, 17), 0)
    hsv = cv.cvtColor(filtered, cv.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    th = cv.adaptiveThreshold(sat, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, -4)
    return th


def threshold_value(orig: np.array, last_modified: np.array) -> np.array:
    filtered = cv.GaussianBlur(orig, (5, 5), 0)
    hsv = cv.cvtColor(filtered, cv.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    ret, th = cv.threshold(hsv[:, :, 1], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return np.logical_and(value > np.mean(value) + 5, th)


def canny_sat(orig: np.array, last_modified: np.array) -> np.array:
    filtered = cv.GaussianBlur(orig, (9, 9), 0)
    hsv = cv.cvtColor(filtered, cv.COLOR_RGB2HSV)
    edges = cv.Canny(hsv[:, :, 1], 30, 70)
    contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] >= 0:  # has parent, inner (hole) contour of a closed edge (looks good)
            cv.drawContours(orig, contours, i, (255, 255, 255), 5, 8)
    return orig


def opening(orig: np.array, last_modified: np.array) -> np.array:
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return cv.morphologyEx(last_modified, cv.MORPH_OPEN, kernel)


def close(orig: np.array, last_modified: np.array) -> np.array:
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    return cv.morphologyEx(last_modified, cv.MORPH_CLOSE, kernel)


def remove_spurious_contours(orig: np.array, last_modified: np.array) -> np.array:
    contours, _ = cv.findContours(last_modified, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 200]
    mask = np.zeros(last_modified.shape, np.uint8)
    cv.drawContours(mask, contours, -1, 255, -1, 8)
    return mask


def max_ecc(orig: np.array, last_modified: np.array) -> np.array:
    contours, _ = cv.findContours(last_modified, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    tuples = map(lambda cnt: (cnt, get_contour_eccentricity(cnt)), contours)
    higher_095 = list(filter(lambda tupl: tupl[1] >= 0.95, tuples))

    mask = np.zeros(last_modified.shape, np.uint8)
    if len(higher_095) > 0:
        max_ecc = max(higher_095, key=lambda tupl: tupl[1])[0]  # returns a tuple
        cv.drawContours(mask, [max_ecc], -1, 255, -1, 8)
    return mask


def convex_hull(orig: np.array, last_modified: np.array) -> np.array:
    contours, _ = cv.findContours(last_modified, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    tube = cv.convexHull(contours[0])
    mask = np.zeros(last_modified.shape, np.uint8)
    cv.drawContours(mask, [tube], -1, 255, -1, 8)
    return mask


# Utility method
def normalize(im: np.array) -> np.array:
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    im = im.astype(np.float)
    minval = np.min(im)
    maxval = np.max(im)
    if minval != maxval:
        im -= minval
        im *= (255.0/(maxval-minval))
    return im.astype(np.uint8)


if __name__ == "__main__":
    use_critical = False
    show_intermediate_figures = False

    if use_critical:
        input_path = "tests/critical/"
        files = [input_path + "critical" + str(num) + ".jpg"
                 for num in [66, 71, 76, 77, 78, 80, 84, 85, 90, 101]]
    else:
        input_path = "im-tests/"
        files = glob.glob(input_path + "test*.jpg")

    if os.path.exists(input_path + "out/"):
        shutil.rmtree(input_path + "out/")

    transforms = [hue, saturation, threshold_sat, opening, close, remove_spurious_contours, max_ecc, convex_hull]

    imgs_pil = [Image.open(file) for file in files]
    imgs_np = [np.array(img_pil) for img_pil in imgs_pil]
    imgs_transformed = imgs_np.copy()

    vpu.showInGrid(imgs_np, title="Imágenes originales", subtitles=files)

    for index in range(len(transforms)):
        transform = transforms[index]
        imgs_transformed = [transform(imgs_np[i], imgs_transformed[i]) for i in range(len(imgs_np))]

        if show_intermediate_figures:
            vpu.showInGrid(imgs_transformed, title=transform.__name__, subtitles=files)

        for j in range(len(imgs_np)):
            _, filename = os.path.split(files[j])
            folder = input_path + "out/" + str(index+1) + "_" + transform.__name__
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename1 = os.path.splitext(filename)[0] + ".png"
            filename2 = os.path.splitext(filename)[0] + "_normalized.png"
            Image.fromarray(imgs_transformed[j]).save(folder + "/" + filename1)
            Image.fromarray(normalize(imgs_transformed[j])).save(folder + "/" + filename2)

    vpu.showInGrid(imgs_transformed, title="Imágenes procesadas")
