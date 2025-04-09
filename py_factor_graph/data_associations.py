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
from typing import Callable, List
from py_factor_graph.variables import PoseVariable2D
from attrs import define, field
from py_factor_graph.measurements import PoseToLandmarkMeasurement2D
from abc import ABC


@attr.s(frozen=False)
class BooleanVariable:
    name: str = attr.ib()
    true_value: bool = attr.ib(default=False)
    estimated_value: bool = attr.ib(default=False)
    timestamp: Optional[float] = attr.ib(default=None)


@define(frozen=False)
class Prior(ABC):
    name: str = attr.ib()  # name of vector variable


@define(frozen=False)
class ToyExamplePrior(Prior):
    center: np.ndarray = attr.ib()
    offset: float = attr.ib()
    Q: np.ndarray = attr.ib()


@define(frozen=False)
class PosePrior2D(Prior):
    rotation: np.ndarray = attr.ib()
    position: Tuple[float, float] = attr.ib()
    weight_rot: float = attr.ib()
    weight_pos: float = attr.ib()


# Need to be able to split PosePrior
# to rotation, position component.
@define(frozen=False)
class RotationPrior2D(Prior):
    rotation: np.ndarray = attr.ib()
    weight: float = attr.ib()


@define(frozen=False)
class PositionPrior2D(Prior):
    position: Tuple[float, float] = attr.ib()
    weight: float = attr.ib()


@attr.s(frozen=False)
class VectorVariable:
    name: str = attr.ib()
    dims: int = attr.ib()
    true_value: np.ndarray = attr.ib(default=None)
    estimated_value: np.ndarray = attr.ib(default=None)
    timestamp: Optional[float] = attr.ib(default=None)


@attr.s(frozen=False)
class UnknownDataAssociationMeasurement:
    measurement_list: List[Union[ToyExamplePrior, PoseToLandmarkMeasurement2D]] = (
        attr.ib()
    )
    boolean_variables: List[BooleanVariable] = attr.ib()
    timestamp: Optional[float] = attr.ib(default=None)


MEASUREMENTS_WITH_SINGLE_STATE_TYPES = Union[
    ToyExamplePrior, PosePrior2D, RotationPrior2D, PositionPrior2D
]


def get_measurement_state_names(
    meas: Union[MEASUREMENTS_WITH_SINGLE_STATE_TYPES, PoseToLandmarkMeasurement2D]
):
    for candidate in MEASUREMENTS_WITH_SINGLE_STATE_TYPES:
        if isinstance(meas, candidate):
            return meas.name
    if isinstance(meas, PoseToLandmarkMeasurement2D):
        return meas.pose_name, meas.landmark_name


from typing import Union

PRIOR_FACTOR_TYPES = Union[
    ToyExamplePrior,
    PosePrior2D,
    PositionPrior2D,
    RotationPrior2D,
]

PRIOR_FACTOR_TYPES_LIST = [
    ToyExamplePrior,
    PosePrior2D,
    PositionPrior2D,
    RotationPrior2D,
]
