import attr
from typing import Optional, Tuple, Union
import numpy as np
from py_factor_graph.utils.attrib_utils import (
    positive_float_validator,
    make_variable_name_validator,
    make_rot_matrix_validator,
    optional_float_validator,
)
from py_factor_graph.utils.matrix_utils import (
    get_covariance_matrix_from_measurement_precisions,
    get_quat_from_rotation_matrix,
)


@attr.s(frozen=False)
class PoseMeasurement2D:
    """
    An pose measurement

    Args:
        base_pose (str): the pose which the measurement is in the reference frame of
        to_pose (str): the name of the pose the measurement is to
        x (float): the measured change in x coordinate
        y (float): the measured change in y coordinate
        theta (float): the measured change in theta
        covariance (np.ndarray): a 3x3 covariance matrix from the measurement model
        timestamp (float): seconds since epoch
    """

    base_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    to_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    rotation_matrix: np.ndarray = attr.ib()
    translation_vector: np.ndarray = attr.ib()
    weight_rot: float = attr.ib()
    weight_pos: float = attr.ib()
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        Get the transformation matrix
        """
        return np.block(
            [
                [self.rotation_matrix, self.translation_vector.reshape(-1, 1)],
                [np.array([0, 0, 1]).reshape(1, -1)],
            ]
        )


@attr.s(frozen=False)
class PoseToLandmarkMeasurement2D:
    pose_name: str = attr.ib(validator=make_variable_name_validator("pose"))
    landmark_name: str = attr.ib(validator=make_variable_name_validator("landmark"))
    r_b_lb: np.ndarray = attr.ib()  # relative translation
    # x: float = attr.ib(validator=attr.validators.instance_of((float, int)))
    # y: float = attr.ib(validator=attr.validators.instance_of((float, int)))
    # translation_precision: float = attr.ib(validator=positive_float_validator)
    weight: float = attr.ib()
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @property
    def translation_vector(self) -> np.ndarray:
        """
        Get the translation vector for the measurement
        """
        return self.r_b_lb


@attr.s(frozen=False)
class PoseToKnownLandmarkMeasurement2D:
    pose_name: str = attr.ib(validator=make_variable_name_validator("pose"))
    r_a_la: np.ndarray = attr.ib()  # known landmark position
    r_b_lb: np.ndarray = attr.ib()  # measurement
    weight: float = attr.ib()
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )


@attr.s(frozen=False)
class KnownPoseToLandmarkMeasurement2D:
    C_ab: np.ndarray = attr.ib()  # known robot orientation
    r_a_ba: np.ndarray = attr.ib()  # known landmark orientation
    r_b_lb: np.ndarray = attr.ib()  # measurement
    landmark_name: str = attr.ib()
    weight: float = attr.ib()
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )


@attr.s(frozen=False)
class PoseToLandmarkMeasurement3D:
    pose_name: str = attr.ib(validator=make_variable_name_validator("pose"))
    landmark_name: str = attr.ib(validator=make_variable_name_validator("landmark"))
    x: float = attr.ib(validator=attr.validators.instance_of(float))
    y: float = attr.ib(validator=attr.validators.instance_of(float))
    z: float = attr.ib(validator=attr.validators.instance_of(float))
    translation_precision: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @property
    def translation_vector(self) -> np.ndarray:
        """
        Get the translation vector for the measurement
        """
        return np.array([self.x, self.y, self.z])

    @property
    def covariance(self) -> np.ndarray:
        """
        Get the covariance matrix
        """
        return np.diag([1 / self.translation_precision] * 3)


@attr.s(frozen=False)
class PoseMeasurement3D:
    """
    An pose measurement

    Args:
        base_pose (str): the pose which the measurement is in the reference frame of
        to_pose (str): the name of the pose the measurement is to
        translation (np.ndarray): the measured change in x, y, z coordinates
        rotation (np.ndarray): the measured change in rotation
        translation_precision (float): the weight of the translation measurement
        rotation_precision (float): the weight of the rotation measurement
        timestamp (float): seconds since epoch
    """

    base_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    to_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    translation: np.ndarray = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    rotation: np.ndarray = attr.ib(validator=make_rot_matrix_validator(3))
    translation_precision: float = attr.ib(validator=positive_float_validator)
    rotation_precision: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    def __attrs_post_init__(self):
        if self.base_pose == self.to_pose:
            raise ValueError(
                f"base_pose and to_pose cannot be the same: base: {self.base_pose}, to: {self.to_pose}"
            )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for the measurement

        Returns:
            np.ndarray: the 3x3 rotation matrix
        """
        return self.rotation

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        Get the transformation matrix

        Returns:
            np.ndarray: the 4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    @property
    def translation_vector(self) -> np.ndarray:
        """
        Get the translation vector for the measurement

        Returns:
            np.ndarray: the 3x1 translation vector
        """
        return self.translation

    @property
    def x(self) -> float:
        """
        Get the x translation

        Returns:
            float: the x translation
        """
        return self.translation[0]

    @property
    def y(self) -> float:
        """
        Get the y translation

        Returns:
            float: the y translation
        """
        return self.translation[1]

    @property
    def z(self) -> float:
        """
        Get the z translation

        Returns:
            float: the z translation
        """
        return self.translation[2]

    @property
    def quat(self) -> np.ndarray:
        """
        Get the quaternion in the form [x, y, z, w]

        Returns:
            np.ndarray: the 4x1 quaternion
        """
        return get_quat_from_rotation_matrix(self.rotation)

    @property
    def covariance(self):
        """
        Get the 6x6 covariance matrix. Right now uses isotropic covariance
        for the translation and rotation respectively

        Returns:
            np.ndarray: the 6x6 covariance matrix
        """
        return get_covariance_matrix_from_measurement_precisions(
            self.translation_precision, self.rotation_precision, mat_dim=6
        )


POSE_MEASUREMENT_TYPES = Union[PoseMeasurement2D, PoseMeasurement3D]
POSE_LANDMARK_MEASUREMENT_TYPES = Union[
    PoseToLandmarkMeasurement2D, PoseToLandmarkMeasurement3D
]
