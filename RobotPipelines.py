import io
import os
import pickle
import time
from itertools import combinations

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


def _is_touching_border(bw):
    return bool(np.sum(bw[0, :]) + np.sum(bw[-1, :]) + np.sum(bw[:, 0]) + np.sum(bw[:, -1]))


class LeaderPipeline(StraightPipeline):
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
            TubeDetectionStage(),
            TubeGripStage(http_interface)
        ])

    def get_last_frame(self):
        return self.get_consumer_output()


class FollowerPipeline(StraightPipeline):
    def __init__(self, address, robot_model: RobotModel, leader: RobotModel, adq_rate):
        if address == "localhost":
            # Avoid DNS resolve for localhost
            address = "127.0.0.1"
        http_interface = RobotHttpInterface(robot_model, address)
        super().__init__([
            FiringStage(adq_rate),
            AdqStage(http_interface),
            ImageConversionStage(),
            RobotDetectionStage(target_robot=leader),
            FollowerStage(http_interface)
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
        return self._http_interface.get_image()


class ImageConversionStage(PipelineStage):
    def _process(self, in_data):
        response_content = in_data
        return np.array(Image.open(io.BytesIO(response_content)))


class TubeDetectionStage(PipelineStage):
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


class RobotDetectionStage(PipelineStage):
    def __init__(self, target_robot: RobotModel):
        super().__init__()
        self._target_robot = target_robot
        self._hsv_range = target_robot.get_hsv_range()

    def _process(self, in_data):
        img = in_data

        # Gaussian filter
        filtered = cv.GaussianBlur(img, (9, 9), 0)

        hsv = cv.cvtColor(filtered, cv.COLOR_RGB2HSV)
        bw = cv.inRange(hsv, self._hsv_range[0], self._hsv_range[1])

        # Opening
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)

        # Close
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)

        # Find & filter contours
        contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours = filter(lambda cnt: cv.contourArea(cnt) > 200, contours)
        contours = [cv.convexHull(cnt) for cnt in contours]

        mask = np.zeros_like(bw)
        cv.drawContours(mask, contours, -1, 255, -1)

        # Compute convex hull of all contours
        for cnt1, cnt2 in combinations(contours, 2):
            M1 = cv.moments(cnt1)
            cX1 = int(M1["m10"] / M1["m00"])
            cY1 = int(M1["m01"] / M1["m00"])

            M2 = cv.moments(cnt2)
            cX2 = int(M2["m10"] / M2["m00"])
            cY2 = int(M2["m01"] / M2["m00"])

            cv.line(mask, (cX1, cY1), (cX2, cY2), color=255, thickness=4)

        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        robot = None
        if len(contours) > 0:
            robot = cv.convexHull(max(contours, key=lambda cnt: cv.contourArea(cnt)))

        return img, robot


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

        self._z_estimator = ZEstimator()

        if PositionControlStage.compute_area_z_relationship:
            self._area_z_list = []

        self._stopped = False

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

            if _is_touching_border(mask):
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


class FollowerStage(Consumer):
    Kp_lineal = 0.0005
    target_size_pixels = 9000

    def __init__(self, http_interface: RobotHttpInterface):
        super().__init__()
        self._http_interface = http_interface

    def _consume(self, in_data):
        img, contour = in_data
        if contour is None:
            self._http_interface.stop()
        else:
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
            # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.drawContours(img, [contour], -1, (0, 255, 0), -1)

            cv.putText(
                img=img,
                text="({:.1f}, {:.1f})".format(center[0], center[1]),
                org=(x + w, y + h),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=1,
                lineType=cv.LINE_AA)

            if _is_touching_border(mask):
                self._http_interface.stop()
            else:
                z_speed = FollowerStage.Kp_lineal * (FollowerStage.target_size_pixels - area)
                self._http_interface.set_velocity(
                    x=FollowerStage.Kp_lineal * (height / 2 - center[1]),
                    y=FollowerStage.Kp_lineal * (center[0] - width / 2),
                    z=max(z_speed, 0),
                    az=0.0
                )
        return Image.fromarray(img)

    def _on_stopped(self):
        self._http_interface.stop()


class TubeGripStage(Consumer):
    Kp_lineal = 0.005
    Kp_angular = 0.08
    use_ellipse = False
    z_threshold = 14.9
    time_close = 7.5

    robot1_ready = False
    robot2_ready = False

    def __init__(self, http_interface: RobotHttpInterface, z_speed_profile: CubicPath = None):
        super().__init__()
        self._http_interface = http_interface
        self._idx = 0 if http_interface.robot_model is RobotModel.GIRONA_500_1 else 2

        self._grip_point = None
        self._step = 0
        self._close_start = None
        self._wait_start = None

        if z_speed_profile is None:
            # Initialize with default values
            self._z_velocity_computer = CubicPath(
                x1=2, x2=12,
                y1=0.25, y2=0.1
            )
        else:
            self._z_velocity_computer = z_speed_profile

        self._z_estimator = ZEstimator()
        self._stopped = False

    def _consume(self, in_data):
        img, contour = in_data

        if self._step == 0:
            if contour is not None:
                img = img.copy()

                img, grip1, grip2, contour_mask, angle, area =\
                    TubeGripStage.get_grip_points(img, contour, use_ellipse=TubeGripStage.use_ellipse)

                if grip1 is not None and grip2 is not None:
                    if grip1[0] < grip2[0]:
                        grip1, grip2 = grip2, grip1
                    self._grip_point = grip1 if self._idx == 0 else grip2

                if _is_touching_border(contour_mask):
                    self._http_interface.stop()
                    self._step = 1
                elif self._grip_point is not None:
                    z = self._z_estimator(area)

                    width = img.shape[1]
                    height = img.shape[0]
                    self._http_interface.set_velocity(
                        x=TubeGripStage.Kp_lineal * (height / 2 - self._grip_point[1]),
                        y=TubeGripStage.Kp_lineal * (self._grip_point[0] - width / 2),
                        z=self._z_velocity_computer(z),
                        az=-TubeGripStage.Kp_angular * angle
                    )
        elif self._step == 1:
            _, _, z = self._http_interface.get_position()
            if z > TubeGripStage.z_threshold:
                self._http_interface.stop()
                self._step = 2
            else:
                self._http_interface.set_velocity(x=0, y=0, z=0.1, az=0, percentage=100)
        elif self._step == 2:
            if self._close_start is None:
                self._close_start = time.time()
                self._http_interface.close_gripper()
            if time.time() - self._close_start > TubeGripStage.time_close:
                self._http_interface.stop_gripper()
                self._step = 3
        elif self._step == 3:
            if self._idx == 0:
                TubeGripStage.robot1_ready = True
            else:
                TubeGripStage.robot2_ready = True

            if TubeGripStage.robot1_ready and TubeGripStage.robot2_ready:
                self._step = 4
        elif self._step == 4:
            self._http_interface.set_velocity(x=0, y=0, z=-0.25, az=0, percentage=100)
        return Image.fromarray(img)

    @staticmethod
    def get_grip_points(img: np.array, contour, use_ellipse: bool = False):
        contour_mask = np.zeros(img.shape[:2])
        cv.drawContours(contour_mask, [contour], -1, 255, -1)

        if use_ellipse:
            ellipse = cv.fitEllipse(contour)
            center, (d1, d2), angle_deg = ellipse
            angle_deg = 90 - angle_deg
            angle_rad = np.deg2rad(angle_deg)
            # Draw ellipse
            cv.ellipse(img, ellipse, color=(0, 255, 255), thickness=1)
            area = None
        else:
            mat = np.argwhere(contour_mask != 0)
            mat[:, [0, 1]] = mat[:, [1, 0]]
            mat = np.array(mat).astype(np.float32)  # have to convert type for PCA

            m, e = cv.PCACompute(mat, mean=np.array([]))
            center = tuple(m[0])  # Note that center is the center of mass, not (max+min)/2 (geometric center)
            eig1 = e[0]
            area = mat.shape[0]
            angle_rad = np.arctan2(-eig1[1], eig1[0])

            cv.drawContours(img, [contour], -1, (0, 255, 255), 2)

        xc, yc = center

        # draw circle at center
        cv.circle(img, (int(xc), int(yc)), 10, (255, 255, 255), -1)

        TubeGripStage.draw_line(img, -angle_rad, (xc, yc), color=(0, 255, 0), thickness=1)
        TubeGripStage.draw_line(img, -angle_rad - np.pi / 2, (xc, yc), color=(0, 255, 0), thickness=1)

        major_axis_bw = np.zeros_like(contour_mask)
        TubeGripStage.draw_line(major_axis_bw, -angle_rad, (xc, yc), color=(255, 255, 255), thickness=1)
        major_axis_bool = major_axis_bw == 255

        contour_outline_mask = np.zeros(img.shape[:2])
        cv.drawContours(contour_outline_mask, [contour], -1, 255, 1)
        contour_mask_bool = contour_outline_mask == 255

        major_axis_intersect = np.logical_and(major_axis_bool, contour_mask_bool)
        intersect_points = np.fliplr(np.transpose(np.where(major_axis_intersect)))
        if len(intersect_points) == 2:
            intersect_pt_1, intersect_pt_2 = intersect_points

            center = np.array(center)
            grip1 = (center + intersect_pt_1) // 2
            grip2 = (center + intersect_pt_2) // 2

            cv.circle(img, tuple(grip1.astype(np.int64)), 5, (255, 0, 255), -1)
            cv.circle(img, tuple(grip2.astype(np.int64)), 5, (255, 0, 255), -1)
        else:
            grip1 = None
            grip2 = None

        return img, grip1, grip2, contour_mask, angle_rad, area

    @staticmethod
    def draw_line(im, angle_rad, point, *args, **kwargs):
        x, y = point
        m = np.tan(angle_rad)
        x1 = 0
        x2 = im.shape[1]
        y1 = m * (x1 - x) + y
        y2 = m * (x2 - x) + y
        cv.line(im, (x1, int(y1)), (x2, int(y2)), *args, **kwargs)

    def is_stopped(self):
        return self._stopped

    def _on_stopped(self):
        self._http_interface.stop()
