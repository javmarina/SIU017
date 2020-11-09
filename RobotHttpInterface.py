import requests


class RobotHttpInterface:
    """
    HTTP interface for controlling Girona 500 movement.
    """

    def __init__(self, port: int, address="127.0.0.1"):
        """
        Create a HTTP robot controller.
        :param port: controller port.
        :param address: controller address, which can be an URL or an IP address. Defaults to "127.0.0.1" (localhost).
        """
        self._url_base = "http://" + address + ":" + str(port)

    def stop(self):
        """
        Stop the robot.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/stop"
        r = requests.get(self._url_base + command)
        return r.status_code == 200

    def forward(self):
        """
        Move the robot forward.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/forward"
        r = requests.get(self._url_base + command)
        return r.status_code == 200

    def backward(self):
        """
        Move the robot backwards.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/backward"
        r = requests.get(self._url_base + command)
        return r.status_code == 200

    def move_left(self):
        """
        Move the robot to the left, without turning around itself.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/left"
        r = requests.get(self._url_base + command)
        return r.status_code == 200

    def move_right(self):
        """
        Move the robot to the right, without turning around itself.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/right"
        r = requests.get(self._url_base + command)
        return r.status_code == 200

    def turn_left(self):
        """
        Make the robot rotate to its left, without changing its position.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/turnleft"
        r = requests.get(self._url_base + command)
        return r.status_code == 200

    def turn_right(self):
        """
        Make the robot rotate to its right, without changing its position.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/turnright"
        r = requests.get(self._url_base + command)
        return r.status_code == 200

    def move_up(self):
        """
        Move the robot up.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/up"
        r = requests.get(self._url_base + command)
        return r.status_code == 200

    def move_down(self):
        """
        Move the robot down.
        :return: True if the HTTP GET request was accepted (200 OK), False otherwise.
        """
        command = "/down"
        r = requests.get(self._url_base + command)
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
        r = requests.get(self._url_base + "/setVelocity", params=query_params)
        return r.status_code == 200

    def get_position(self):
        """
        Get current robot position.
        :return: a 3-item tuple with the robot coordinates in X, Y and Z.
        """
        command = "/getPosition"
        r = requests.get(self._url_base + command)
        position = r.json()
        return position["x"], position["y"], position["z"]

# if __name__ == "__main__":
#     robot_controller = RobotHttpInterface(8000)
#     pass
