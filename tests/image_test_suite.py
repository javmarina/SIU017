from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import visualPercepUtils as vpu


def get_contour_eccentricity(contour):
    (_, _), (ma, MA), _ = cv.fitEllipse(contour)
    return np.sqrt(1-(ma/MA)**2)


def detection(img: np.array):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    max_saturation = np.amax(saturation)
    mask = saturation > (max_saturation-60)

    bw = mask.astype(np.uint8) * 255
    contours, _ = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    mask2 = np.zeros(hue.shape, np.uint8)
    cv.drawContours(mask2, contours, -1, 255, -1, 8)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)
    dilation = cv.dilate(opening, kernel, iterations=2)
    contours, _ = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    plt.imshow(dilation)
    plt.show()

    contours = [cnt for cnt in contours
                if cnt.shape[0] > 10 and get_contour_eccentricity(cnt) >= 0.98]

    if len(contours) == 1:
        tube = contours[0]
    else:
        mask2 *= 0
        diffs = []
        for cnt in contours:
            cv.drawContours(mask2, [cnt], -1, 255, -1)
            mean = cv.mean(hue, mask=mask2)
            diffs.append(np.abs(mean[0]-84))
            mask2 *= 0
        tube = contours[diffs.index(min(diffs))]
    cv.drawContours(img, [tube], 0, (0, 255, 0), 4)

    # for green_cnt, mean in green_contours:
    #     x, y, w, h = cv.boundingRect(green_cnt)
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show(block=True)
    input()


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
    th_sat = (max_sat+mean_sat)/2 - 10

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
    contours, _ = cv.findContours(last_modified, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cnt.shape[0] > 20]
    mask = np.zeros(last_modified.shape, np.uint8)
    cv.drawContours(mask, contours, -1, 255, -1, 8)
    return mask


def max_ecc(orig: np.array, last_modified: np.array) -> np.array:
    contours, _ = cv.findContours(last_modified, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    tuples = map(lambda cnt: (cnt, get_contour_eccentricity(cnt)), contours)
    higher_095 = list(filter(lambda tupl: tupl[1] >= 0.95, tuples))

    mask = np.zeros(last_modified.shape, np.uint8)
    if len(higher_095) > 0:
        max_ecc = max(higher_095, key=lambda tupl: tupl[1])
        cv.drawContours(mask, [max_ecc[0]], -1, 255, -1, 8)
    return mask


if __name__ == "__main__":
    use_critical = True
    if use_critical:
        files = ["tests/critical/critical" + str(num) + ".jpg"
                 for num in [66, 71, 76, 77, 78, 80, 84, 85, 90, 101]]
    else:
        input_path = "tests/im-tests/"
        files = glob.glob(input_path + "test*.jpg")

    transforms = [hue, threshold_sat, opening, close, remove_spurious_contours, max_ecc]

    imgs_pil = [Image.open(file) for file in files]
    imgs_np = [np.array(img_pil) for img_pil in imgs_pil]
    imgs_transformed = imgs_np.copy()

    vpu.showInGrid(imgs_np, title="Imágenes originales", subtitles=files)

    for transform in transforms:
        imgs_transformed = [transform(imgs_np[i], imgs_transformed[i]) for i in range(len(imgs_np))]
        vpu.showInGrid(imgs_transformed, title=transform.__name__, subtitles=files)
    vpu.showInGrid(imgs_transformed, title="Imágenes procesadas")

    #detection(img)
