import socket
from threading import Lock
from urllib import request

import requests

from RobotModel import RobotModel


class RobotHttpInterface:
    """
    HTTP interface for controlling Girona 500 movement.
    """

    lock = Lock()

    def __init__(self, robot_model: RobotModel, address: str = "127.0.0.1"):
        """
        Create a HTTP robot controller.
        :param robot_model: model of the robot (see RobotModel enum for reference).
        :param address: controller address, which can be an URL or an IP address. Defaults to "127.0.0.1" (localhost).
        """
        self._movement_url = "http://" + address + ":" + str(robot_model.get_movement_controller_port())
        try:
            self._gripper_url = "http://" + address + ":" + str(robot_model.get_gripper_controller_port())
        except ValueError:
            # No gripper available in this robot
            self._gripper_url = None

        self._camera_url = "http://" + address + ":" + str(robot_model.get_camera_port())
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._camera_udp_address = (address, robot_model.get_camera_port())

    def stop(self):
        """
        Stop the robot.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/stop"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def forward(self):
        """
        Move the robot forward.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/forward"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def backward(self):
        """
        Move the robot backwards.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/backward"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def move_left(self):
        """
        Move the robot to the left, without turning around itself.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/left"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def move_right(self):
        """
        Move the robot to the right, without turning around itself.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/right"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def turn_left(self):
        """
        Make the robot rotate to its left, without changing its position.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/turnleft"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def turn_right(self):
        """
        Make the robot rotate to its right, without changing its position.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/turnright"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def move_up(self):
        """
        Move the robot up.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/up"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def move_down(self):
        """
        Move the robot down.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/down"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def set_velocity(self, x, y, z, az, percentage=100):
        """
        Set the robot velocity.
        :param x: velocity in X axis.
        :param y: velocity in Y axis.
        :param z: velocity in Z axis.
        :param az: rotation speed around Z axis (?).
        :param percentage: set the robot velocity to this percentage of the previous values. Defaults to 100,
        which means the arguments are used directly. If 0, it is equivalent to calling stop().
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        if percentage > 100 or percentage < 0:
            raise ValueError(f"Percentage must be between 0 and 100, was {percentage}")

        query_params = {
            "X": x,
            "Y": y,
            "Z": z,
            "AZ": az,
            "PERCENTAGE": percentage
        }
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + "/setVelocity", params=query_params)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def get_position(self):
        """
        Get current robot position.
        :return: a 3-item tuple with the robot coordinates in X, Y and Z.
        """
        command = "/getPosition"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._movement_url + command)
        RobotHttpInterface.lock.release()
        position = r.json()
        return position["x"], position["y"], position["z"]

    def open_gripper(self):
        """
        Open the robot gripper.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/open"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._gripper_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def close_gripper(self):
        """
        Close the robot gripper.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/close"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._gripper_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def stop_gripper(self):
        """
        Stop the robot gripper.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/stop"
        RobotHttpInterface.lock.acquire()
        r = requests.get(self._gripper_url + command)
        RobotHttpInterface.lock.release()
        return r.status_code == 200

    def get_image(self):
        """
        Send petition to download camera image.
        :return: received response content (bytes).
        """
        with request.urlopen(self._camera_url) as f:
            return f.read()

    def get_image_udp(self, width: int = 640, quality: int = 90):
        """
        Send petition to download camera image via UDP.
        :param width: desired width of image in pixels. Height will be computed automatically.
        :param quality: quality percentage, integer from 0 to 100.
        :return: received response content (bytes).
        """
        if quality > 90:
            quality = 90
        if quality < 0:
            quality = 0
        message = "/setparams?width={:d}&quality={:d}".format(width, quality)
        self._udp_socket.sendto(message.encode(), self._camera_udp_address)
        datagramFromClient, _ = self._udp_socket.recvfrom(65000)
        return datagramFromClient

# if __name__ == "__main__":
#     robot_controller = RobotHttpInterface(8000)
#     pass
