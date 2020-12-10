import io
import time

import cv2 as cv
import numpy as np
import requests
from PIL import Image

from RobotHttpInterface import RobotHttpInterface
from RobotModel import RobotModel
from pipeline.Pipeline import StraightPipeline
from pipeline.PipelineStage import Producer, PipelineStage, Consumer


class ImagePipeline(StraightPipeline):
    def __init__(self, address, robot_model: RobotModel, adq_rate):
        if address == "localhost":
            # Avoid DNS resolve for localhost
            # TODO: cache DNS resolution for other IPs?
            address = "127.0.0.1"
        super().__init__([
            FiringStage(adq_rate, address, robot_model),
            AdqStage(),
            ImageConversionStage(),
            ObjectDetectionStage(),
            PositionControlStage(address, robot_model)
        ])

    def get_last_frame(self):
        return self.get_consumer_output()


class FiringStage(Producer):
    def __init__(self, adq_rate, address, robot_model: RobotModel):
        super().__init__()
        self._sleep_seconds = 1.0 / adq_rate
        self._url = "http://" + address + ":" + str(robot_model.get_camera_port())

    def _produce(self):
        time.sleep(self._sleep_seconds)
        return self._url


class AdqStage(PipelineStage):

    # Whether to use the low-level urllib library instead of requests when fetching the camera image
    # urllib takes ~58 ms to download the image, while requests spends ~90 ms (~75 ms when verify=False)
    use_urllib = True

    def _process(self, in_data):
        url = in_data
        if AdqStage.use_urllib:
            from urllib import request
            with request.urlopen(url) as f:
                return f.read()
        else:
            response = requests.get(url, verify=False)
            return response.content


class ImageConversionStage(PipelineStage):
    def _process(self, in_data):
        response_content = in_data
        return np.array(Image.open(io.BytesIO(response_content)))


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

    A1 = 100
    v1 = 0.25
    A2 = 5000
    v2 = 0.04

    compute_area_z_relationship = True

    def __init__(self, address, robot_model: RobotModel):
        super().__init__()
        self._http_interface = RobotHttpInterface(robot_model, address)
        self._z_velocity_computer = PositionControlStage.VelocityZComputer(
            A1=PositionControlStage.A1,
            A2=PositionControlStage.A2,
            v1=PositionControlStage.v1,
            v2=PositionControlStage.v2
        )
        if PositionControlStage.compute_area_z_relationship:
            self._area_z_list = []
        self._stopped = False

    @staticmethod
    def _is_touching_border(bw):
        return bool(np.sum(bw[0, :]) + np.sum(bw[-1, :]) + np.sum(bw[:, 0]) + np.sum(bw[:, -1]))

    def _consume(self, in_data):
        img, contour = in_data
        if contour is not None:
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

            if PositionControlStage.compute_area_z_relationship:
                self._area_z_list.append((area, self._http_interface.get_position()[2]))

            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv.putText(
                img=img,
                text="({:.1f}, {:.1f})".format(center[0], center[1]),
                org=(x+w, y+h),
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
                self._http_interface.set_velocity(
                    x=PositionControlStage.Kp_lineal*(height/2-center[1]),
                    y=PositionControlStage.Kp_lineal*(center[0]-width/2),
                    z=self._z_velocity_computer.compute_velocity(area),
                    az=-PositionControlStage.Kp_angular*angle
                )
        return Image.fromarray(img)

    def _on_stopped(self):
        self._http_interface.stop()
        if PositionControlStage.compute_area_z_relationship:
            import pickle
            import os
            current_dir = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(current_dir, "tests", "area_z.p")
            if os.path.exists(path):
                os.remove(path)
            with open(path, "xb") as f:
                pickle.dump(obj=self._area_z_list, file=f)

    class VelocityZComputer:
        def __init__(self, A1, A2, v1, v2):
            delta_A = A2-A1
            delta_v = v2-v1
            self._a = -2*delta_v/(delta_A**3)
            self._b = 3*delta_v/(delta_A**2)
            self._c = 0
            self._d = v1

            self._A1 = A1
            self._A2 = A2
            self._v1 = v1
            self._v2 = v2

        def compute_velocity(self, area):
            if area <= self._A1:
                return self._v1
            elif area >= self._A2:
                return self._v2
            else:
                return self._a*(area-self._A1)**3 + self._b*(area-self._A1)**2\
                       + self._c*(area-self._A1) + self._d
