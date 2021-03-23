import io
import os
import pickle
import time

import cv2 as cv
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from RobotHttpInterface import RobotHttpInterface
from RobotModel import RobotModel
from pipeline.Pipeline import StraightPipeline
from pipeline.PipelineStage import Producer, PipelineStage, Consumer
from utils.CubicPath import CubicPath


class ImagePipeline(StraightPipeline):
    def __init__(self, address: str, robot_model: RobotModel, adq_rate):
        if address == "localhost":
            # Avoid DNS resolve for localhost
            # TODO: cache DNS resolution for other IPs?
            address = "127.0.0.1"
        http_interface = RobotHttpInterface(robot_model, address)
        super().__init__([
            FiringStage(adq_rate),
            AdqStage(http_interface),
            ImageConversionStage(),
            ObjectDetectionStage(),
            PositionControlStage(http_interface)
        ])

    def get_last_frame(self):
        return self.get_consumer_output()


class FiringStage(Producer):
    def __init__(self, adq_rate):
        super().__init__()
        self._sleep_seconds = 1.0 / adq_rate

    def _produce(self):
        time.sleep(self._sleep_seconds)
        return ()


class AdqStage(PipelineStage):
    def __init__(self, http_interface: RobotHttpInterface):
        super().__init__()
        self._http_interface = http_interface

    def _process(self, _):
        return self._http_interface.get_image_udp(width=640, quality=90)


class ImageConversionStage(PipelineStage):
    def _process(self, in_data):
        datagramFromClient = in_data
        return np.array(Image.open(io.BytesIO(datagramFromClient)))


class ObjectDetectionStage(PipelineStage):
    @staticmethod
    def _get_contour_eccentricity(contour):
        (_, _), (ma, MA), _ = cv.fitEllipse(contour)
        return np.sqrt(1 - (ma / MA) ** 2)

    def _process(self, in_data):
        img = in_data

        # Gaussian filter
        filtered = cv.GaussianBlur(img, (9, 9), 0)

        # Saturation thresholding (with red color elimination)
        hsv = cv.cvtColor(filtered, cv.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        max_sat = np.max(sat)
        mean_sat = np.mean(sat)
        th_sat = (max_sat + mean_sat) / 2 - 15
        bw = cv.inRange(hsv, (10, th_sat, 0), (180, 255, 255))  # Remove H<10 means red color from lasers won't appear

        # Opening
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)

        # Close
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)

        # Find & filter contours
        contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        big_contours = filter(lambda cnt: cv.contourArea(cnt) > 200, contours)
        tuples = map(lambda cnt: (cnt, self._get_contour_eccentricity(cnt)), big_contours)
        higher_095 = list(filter(lambda tupl: tupl[1] >= 0.95, tuples))

        # Select tube, if found
        tube = None
        if len(higher_095) > 0:
            tube = max(higher_095, key=lambda tupl: tupl[1])[0]
            # Return convex hull, which should be more robust and use less memory
            tube = cv.convexHull(tube)

        return img, tube


class PositionControlStage(Consumer):
    Kp_lineal = 0.005
    Kp_angular = 0.08

    compute_area_z_relationship = False

    def __init__(self, http_interface: RobotHttpInterface, z_speed_profile: CubicPath = None):
        super().__init__()
        self._http_interface = http_interface

        if z_speed_profile is None:
            # Initialize with default values
            self._z_velocity_computer = CubicPath(
                x1=2, x2=8,
                y1=0.25, y2=0.1
            )
        else:
            self._z_velocity_computer = z_speed_profile

        self._z_estimator = PositionControlStage.ZEstimator()

        if PositionControlStage.compute_area_z_relationship:
            self._area_z_list = []

        self._stopped = False

    @staticmethod
    def _is_touching_border(bw):
        return bool(np.sum(bw[0, :]) + np.sum(bw[-1, :]) + np.sum(bw[:, 0]) + np.sum(bw[:, -1]))

    def _consume(self, in_data):
        img, contour = in_data
        if contour is not None:
            img = img.copy()
            width = img.shape[1]
            height = img.shape[0]

            mask = np.zeros(img.shape[0:2])
            cv.drawContours(mask, [contour], -1, 255, -1)
            mat = np.argwhere(mask != 0)
            mat[:, [0, 1]] = mat[:, [1, 0]]
            mat = np.array(mat).astype(np.float32)  # have to convert type for PCA

            m, e = cv.PCACompute(mat, mean=np.array([]))
            center = tuple(m[0])  # Note that center is the center of mass, not (max+min)/2 (geometric center)
            eig1 = e[0]
            angle = np.arctan2(-eig1[1], eig1[0])
            area = mat.shape[0]

            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv.putText(
                img=img,
                text="({:.1f}, {:.1f})".format(center[0], center[1]),
                org=(x + w, y + h),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=1,
                lineType=cv.LINE_AA)

            # center[0] selecciona la columna
            # center[1] selecciona la fila

            # Velocidades
            # Positiva en X: hacia arriba de la imagen
            # Negativa en X: hacia abajo de la imagen
            # Positiva en Y: hacia la derecha de la imagen
            # Negativa en Y: hacia la izda
            # Positivo en Z: hacia el fondo
            # Negativo en Z: hacia el cielo
            # AZ positivo: el robot gira con sacacorchos hacia el fondo (turn right)
            # AZ negativo: el robot gira con sacacorchos hacia arriba (turn left)

            if self._is_touching_border(mask):
                self._http_interface.stop()
                self._stopped = True
            elif not self._stopped:
                if PositionControlStage.compute_area_z_relationship:
                    self._area_z_list.append((area, self._http_interface.get_position()[2]))

                z = self._z_estimator(area)

                self._http_interface.set_velocity(
                    x=PositionControlStage.Kp_lineal * (height / 2 - center[1]),
                    y=PositionControlStage.Kp_lineal * (center[0] - width / 2),
                    z=self._z_velocity_computer(z),
                    az=-PositionControlStage.Kp_angular * angle
                )
        return Image.fromarray(img)

    def is_stopped(self):
        return self._stopped

    def _on_stopped(self):
        self._http_interface.stop()
        if PositionControlStage.compute_area_z_relationship:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(current_dir, "tests", "area_z.p")
            if os.path.exists(path):
                os.remove(path)
            with open(path, "xb") as f:
                pickle.dump(obj=self._area_z_list, file=f)

    class ZEstimator:
        def __init__(self):
            current_dir = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(current_dir, "tests", "area_z.p")

            with open(path, "rb") as f:
                list = pickle.load(f)

            self._interpolator = None
            if list is not None:
                areas = []
                zs = []
                for area, z in list:
                    areas.append(area)
                    zs.append(z)

                areas = savgol_filter(areas, window_length=51, polyorder=3)
                self._interpolator = interp1d(areas, zs, kind='cubic')
                self._min_area = min(areas)
                self._max_area = max(areas)
                self._min_z = min(zs)
                self._max_z = max(zs)

        def __call__(self, area):
            if self._interpolator is None:
                return None
            else:
                if area <= self._min_area:
                    return self._min_z
                elif area >= self._max_area:
                    return self._max_z
                else:
                    return self._interpolator(area)
