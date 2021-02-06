from enum import Enum, unique


@unique
class RobotModel(Enum):
    GIRONA_500_1 = "Girona 500 1"
    GIRONA_500_2 = "Girona 500 2"
    BLUE_ROV = "BlueRov"
    
    def get_movement_controller_port(self) -> int:
        """
        :return: the HTTP movement control interface port for this robot model.
        """
        if self == RobotModel.GIRONA_500_1:
            return 8000
        elif self == RobotModel.GIRONA_500_2:
            return 8010
        elif self == RobotModel.BLUE_ROV:
            return 8020
        else:
            raise ValueError("Invalid robot model: " + str(self))

    def get_gripper_controller_port(self) -> int:
        """
        :return: the HTTP gripper control interface port for this robot model.
        """
        if self == RobotModel.GIRONA_500_1:
            return 8002
        elif self == RobotModel.GIRONA_500_2:
            return 8012
        else:
            raise ValueError("Invalid robot model: " + str(self))

    def get_camera_port(self) -> int:
        """
        :return: the camera port for this robot model.
        """
        if self == RobotModel.GIRONA_500_1:
            return 8001
        elif self == RobotModel.GIRONA_500_2:
            return 8011
        elif self == RobotModel.BLUE_ROV:
            return 8021
        else:
            raise ValueError("Invalid robot model: " + str(self))
