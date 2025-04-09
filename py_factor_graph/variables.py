import attr
from typing import Tuple, Optional, Union
import numpy as np
from py_factor_graph.utils.matrix_utils import (
    get_quat_from_rotation_matrix,
    _check_transformation_matrix,
    get_rotation_matrix_from_transformation_matrix,
    get_translation_from_transformation_matrix,
    get_theta_from_transformation_matrix,
)
from py_factor_graph.utils.attrib_utils import (
    optional_float_validator,
    make_rot_matrix_validator,
    make_variable_name_validator,
)


@attr.s(frozen=False)
class PoseVariable2D:
    """A variable which is a robot pose

    Args:
        name (str): the name of the variable (defines the frame)
        true_position (Tuple[float, float]): the true position of the robot
        true_theta (float): the true orientation of the robot
        timestamp (float): seconds since epoch
    """

    name: str = attr.ib(validator=make_variable_name_validator("pose"))
    true_position: np.ndarray = attr.ib(default=None)
    true_theta: float = attr.ib(default=None)
    estimated_position: np.ndarray = attr.ib(default=None)
    estimated_theta: float = attr.ib(default=None)
    timestamp: Optional[float] = attr.ib(default=None)

    def error(self):
        e_rot = (
            np.linalg.norm(
                self.true_rotation_matrix - self.estimated_rotation_matrix, "fro"
            )
            ** 2
        )
        e_pos = (
            np.linalg.norm(
                np.array(self.true_position) - np.array(self.estimated_position), 2
            )
            ** 2
        )
        return e_rot + e_pos

    # @true_position.validator
    # def _check_true_position(self, attribute, value):
    #     if len(value) != 2:
    #         raise ValueError(f"true_position should be a tuple of length 2")
    #     assert all(isinstance(x, (float, int)) for x in value)

    @property
    def true_rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for the measurement
        """
        return np.array(
            [
                [np.cos(self.true_theta), -np.sin(self.true_theta)],
                [np.sin(self.true_theta), np.cos(self.true_theta)],
            ]
        )

    @property
    def estimated_rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for the measurement
        """
        return np.array(
            [
                [np.cos(self.estimated_theta), -np.sin(self.estimated_theta)],
                [np.sin(self.estimated_theta), np.cos(self.estimated_theta)],
            ]
        )

    @property
    def position_vector(self) -> np.ndarray:
        """
        Get the position vector for the measurement
        """
        return self.true_position

    @property
    def true_x(self) -> float:
        return self.true_position[0]

    @property
    def true_y(self) -> float:
        return self.true_position[1]

    @property
    def true_z(self) -> float:
        return 0

    @property
    def true_quat(self) -> np.ndarray:
        quat = np.array(
            [0.0, 0.0, np.sin(self.true_theta / 2), np.cos(self.true_theta / 2)]
        )
        return quat

    @property
    def true_transformation_matrix(self) -> np.ndarray:
        """Returns the transformation matrix representing the true latent pose
        of this variable

        Returns:
            np.ndarray: the transformation matrix
        """
        T = np.eye(3)
        T[0:2, 0:2] = self.true_rotation_matrix
        T[0, 2] = self.true_position[0]
        T[1, 2] = self.true_position[1]
        return T

    @property
    def estimated_transformation_matrix(self) -> np.ndarray:
        """Returns the transformation matrix representing the true latent pose
        of this variable

        Returns:
            np.ndarray: the transformation matrix
        """
        T = np.eye(3)
        T[0:2, 0:2] = self.estimated_rotation_matrix
        T[0, 2] = self.estimated_position[0]
        T[1, 2] = self.estimated_position[1]
        return T

    def transform(self, T: np.ndarray) -> "PoseVariable2D":
        """Returns the transformation matrix representing the true latent pose
        of this variable

        Returns:
            PoseVariable2D: the transformed pose
        """
        _check_transformation_matrix(T)
        assert T.shape == (3, 3)
        current_transformation = self.transformation_matrix
        new_transformation = current_transformation @ T
        new_position = get_translation_from_transformation_matrix(new_transformation)
        new_theta = get_theta_from_transformation_matrix(new_transformation)
        pos2d = (float(new_position[0]), float(new_position[1]))
        return PoseVariable2D(self.name, pos2d, new_theta, self.timestamp)


@attr.s(frozen=False)
class PoseVariable3D:
    """A variable which is a robot pose

    Args:
        name (str): the name of the variable (defines the frame)
        true_position (Tuple[float, float, float]): the true position of the robot
        true_rotation (np.ndarray): the true orientation of the robot
        timestamp (float): seconds since epoch
    """

    name: str = attr.ib(validator=make_variable_name_validator("pose"))
    true_position: Tuple[float, float, float] = attr.ib()
    true_rotation: np.ndarray = attr.ib(validator=make_rot_matrix_validator(3))
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @true_position.validator
    def _check_true_position(self, attribute, value):
        if len(value) != 3:
            raise ValueError(f"true_position should be a tuple of length 3: {value}")
        assert all(isinstance(x, float) for x in value)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for the measurement
        """
        return self.true_rotation

    @property
    def position_vector(self) -> np.ndarray:
        """
        Get the position vector for the measurement
        """
        return np.array(self.true_position)

    @property
    def true_x(self) -> float:
        return self.true_position[0]

    @property
    def true_y(self) -> float:
        return self.true_position[1]

    @property
    def true_z(self) -> float:
        return self.true_position[2]

    @property
    def true_quat(self) -> np.ndarray:
        rot = self.rotation_matrix
        quat = get_quat_from_rotation_matrix(rot)
        return quat

    @property
    def transformation_matrix(self) -> np.ndarray:
        """Returns the transformation matrix representing the true latent pose
        of this variable

        Returns:
            np.ndarray: the transformation matrix
        """
        T = np.eye(self.dimension + 1)
        T[: self.dimension, : self.dimension] = self.true_rotation
        T[: self.dimension, self.dimension] = self.true_position
        assert T.shape == (4, 4)
        return T

    def transform(self, T: np.ndarray) -> "PoseVariable3D":
        """Returns the transformation matrix representing the true latent pose
        of this variable

        Returns:
            PoseVariable3D: the transformed pose
        """
        _check_transformation_matrix(T)
        assert T.shape == (4, 4)
        current_transformation = self.transformation_matrix
        new_transformation = current_transformation @ T
        new_position = get_translation_from_transformation_matrix(new_transformation)
        new_rotation = get_rotation_matrix_from_transformation_matrix(
            new_transformation
        )
        pos3d = (float(new_position[0]), float(new_position[1]), float(new_position[2]))
        return PoseVariable3D(self.name, pos3d, new_rotation, self.timestamp)


@attr.s(frozen=False)
class LandmarkVariable2D:
    """A variable which is a landmark

    Arguments:
        name (str): the name of the variable
        true_position (Tuple[float, float]): the true position of the landmark
    """

    name: str = attr.ib(validator=make_variable_name_validator("landmark"))
    true_position: Tuple[float, float] = attr.ib()
    estimated_position: Tuple[float, float] = attr.ib(default=None)

    @true_position.validator
    def _check_true_position(self, attribute, value):
        if len(value) != 2:
            raise ValueError(f"true_position should be a tuple of length 2")
        assert all((isinstance(x, float) or isinstance(x, int)) for x in value)

    @property
    def true_x(self):
        return self.true_position[0]

    @property
    def true_y(self):
        return self.true_position[1]


@attr.s(frozen=True)
class LandmarkVariable3D:
    """A variable which is a landmark

    Arguments:
        name (str): the name of the variable
        true_position (Tuple[float, float, float]): the true position of the landmark
    """

    name: str = attr.ib(validator=make_variable_name_validator("landmark"))
    true_position: Tuple[float, float, float] = attr.ib()

    @true_position.validator
    def _check_true_position(self, attribute, value):
        if len(value) != 3:
            raise ValueError(f"true_position should be a tuple of length 3")
        assert all(isinstance(x, float) for x in value)

    @property
    def true_x(self):
        return self.true_position[0]

    @property
    def true_y(self):
        return self.true_position[1]

    @property
    def true_z(self):
        return self.true_position[2]


POSE_VARIABLE_TYPES = Union[PoseVariable2D, PoseVariable3D]
LANDMARK_VARIABLE_TYPES = Union[LandmarkVariable2D, LandmarkVariable3D]
