import numpy as np
import os
import math
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_quat,
    get_symmetric_matrix_from_list_column_major,
    get_list_column_major_from_symmetric_matrix,
    get_measurement_precisions_from_covariance_matrix,
)
from py_factor_graph.variables import (
    PoseVariable2D,
    PoseVariable3D,
    LandmarkVariable2D,
    LandmarkVariable3D,
    POSE_VARIABLE_TYPES,
    LANDMARK_VARIABLE_TYPES,
)
from py_factor_graph.priors import (
    PosePrior2D,
    PosePrior3D,
    LandmarkPrior2D,
    LandmarkPrior3D,
    POSE_PRIOR_TYPES,
    LANDMARK_PRIOR_TYPES,
)
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    PoseMeasurement3D,
    PoseToLandmarkMeasurement2D,
    PoseToLandmarkMeasurement3D,
    FGRangeMeasurement,
    POSE_MEASUREMENT_TYPES,
    POSE_LANDMARK_MEASUREMENT_TYPES,
)
from py_factor_graph.utils.logging_utils import logger


POSE_TYPE_2D = "VERTEX_SE2"
POSE_TYPE_3D = "VERTEX_SE3:QUAT"
POSE_PRIOR_2D = "VERTEX_SE2:PRIOR"
POSE_PRIOR_3D = "VERTEX_SE3:QUAT:PRIOR"
LANDMARK_TYPE_2D = "VERTEX_XY"
LANDMARK_TYPE_3D = "VERTEX_XYZ"
LANDMARK_PRIOR_2D = "VERTEX_XY:PRIOR"
LANDMARK_PRIOR_3D = "VERTEX_XYZ:PRIOR"
REL_POSE_POSE_TYPE_2D = "EDGE_SE2"
REL_POSE_POSE_TYPE_3D = "EDGE_SE3:QUAT"
REL_POSE_LANDMARK_TYPE_2D = "EDGE_SE2_XY"
REL_POSE_LANDMARK_TYPE_3D = "EDGE_SE3_XYZ"
RANGE_MEASURE_TYPE = "EDGE_RANGE"


def _num_elems_symmetric_matrix(dim: int) -> int:
    """Get the number of elements in a symmetric matrix of dimension dim.

    Args:
        dim (int): dimension of the matrix.

    Returns:
        int: number of elements in the matrix.
    """
    return int(dim * (dim + 1) / 2)


def _get_measurement_noise_str_from_covariance_matrix(
    covar: np.ndarray, fprec: int
) -> str:
    """Get the measurement noise string from a covariance matrix

    Args:
        covar (np.ndarray): covariance matrix.
        fprec (int): format precision.

    Returns:
        str: string containing PyFG formatted measurement noise
    """

    # make sure that the matrix is full rank even after rounding to fprec
    rounded_covar = np.round(covar, fprec)
    if np.linalg.matrix_rank(rounded_covar) != rounded_covar.shape[0]:
        raise ValueError(
            f"Covariance matrix is not full rank after rounding to {fprec} decimal places. "
            "This will likely cause numerical issues. We suggest increasing the variance."
            f"Matrix before rounding:\n{covar}\nMatrix after rounding:\n{rounded_covar}"
        )

    covar_mat_elems = get_list_column_major_from_symmetric_matrix(covar)
    measurement_noise = " ".join([f"{x:.{fprec}f}" for x in covar_mat_elems])

    # correct formatting of negative zeros
    measurement_noise = measurement_noise.replace("-0. ", "0. ")
    return measurement_noise


def _get_pyfg_types(dim: int) -> tuple:
    """Get PyFG types based on the data dimension (i.e. 2D or 3D)

    Args:
        dim (int): data dimension. Must be 2 or 3.

    Returns:
        pose_var_type (str): PyFG formatted string for pose variables
        landmark_var_type (str): PyFG formatted string for landmark variables
        pose_prior_type (str): PyFG formatted string for pose priors
        landmark_prior_type (str): PyFG formatted string for landmark priors
        rel_pose_pose_measure_type (str): PyFG formatted string for pose to pose measurements
        rel_pose_landmark_measure_type (str): PyFG formatted string for pose to landmark measurements
        range_measure_type (str): PyFG formatted string for range measurements
    """
    assert dim == 2 or dim == 3
    if dim == 2:
        pose_var_type = POSE_TYPE_2D
        landmark_var_type = LANDMARK_TYPE_2D
        pose_prior_type = POSE_PRIOR_2D
        landmark_prior_type = LANDMARK_PRIOR_2D
        rel_pose_pose_measure_type = REL_POSE_POSE_TYPE_2D
        rel_pose_landmark_measure_type = REL_POSE_LANDMARK_TYPE_2D
    else:
        pose_var_type = POSE_TYPE_3D
        landmark_var_type = LANDMARK_TYPE_3D
        pose_prior_type = POSE_PRIOR_3D
        landmark_prior_type = LANDMARK_PRIOR_3D
        rel_pose_pose_measure_type = REL_POSE_POSE_TYPE_3D
        rel_pose_landmark_measure_type = REL_POSE_LANDMARK_TYPE_3D
    range_measure_type = RANGE_MEASURE_TYPE

    return (
        pose_var_type,
        landmark_var_type,
        pose_prior_type,
        landmark_prior_type,
        rel_pose_pose_measure_type,
        rel_pose_landmark_measure_type,
        range_measure_type,
    )


def save_to_pyfg_text(
    fg: FactorGraphData, fpath: str, use_gt_measurements: bool = False
) -> None:
    """Save factor graph to PyFG file format.

    Args:
        fg (FactorGraphData): factor graph data.
        fpath (str): file path.
    """

    (
        pose_var_type,
        landmark_var_type,
        pose_prior_type,
        landmark_prior_type,
        rel_pose_pose_measure_type,
        rel_pose_landmark_measure_type,
        range_measure_type,
    ) = _get_pyfg_types(fg.dimension)

    # enforce formatting precisions
    time_fprec = 9
    translation_fprec = 9
    theta_fprec = 7
    quaternion_fprec = 7
    covariance_fprec = 9

    def _get_pose_var_string(pose_var: POSE_VARIABLE_TYPES):
        if isinstance(pose_var, PoseVariable2D):
            return f"{pose_var_type} {pose_var.timestamp:.{time_fprec}f} {pose_var.name} {pose_var.true_x:.{translation_fprec}f} {pose_var.true_y:.{translation_fprec}f} {pose_var.true_theta:.{theta_fprec}f}"
        elif isinstance(pose_var, PoseVariable3D):
            qx, qy, qz, qw = pose_var.true_quat
            return f"{pose_var_type} {pose_var.timestamp:.{time_fprec}f} {pose_var.name} {pose_var.true_x:.{translation_fprec}f} {pose_var.true_y:.{translation_fprec}f} {pose_var.true_z:.{translation_fprec}f} {qx:.{quaternion_fprec}f} {qy:.{quaternion_fprec}f} {qz:.{quaternion_fprec}f} {qw:.{quaternion_fprec}f}"
        else:
            raise ValueError(f"Unknown pose type {type(pose_var)}")

    def _get_landmark_var_string(landmark_var: LANDMARK_VARIABLE_TYPES):
        if isinstance(landmark_var, LandmarkVariable2D):
            return f"{landmark_var_type} {landmark_var.name} {landmark_var.true_x:.{translation_fprec}f} {landmark_var.true_y:.{translation_fprec}f}"
        elif isinstance(landmark_var, LandmarkVariable3D):
            return f"{landmark_var_type} {landmark_var.name} {landmark_var.true_x:.{translation_fprec}f} {landmark_var.true_y:.{translation_fprec}f} {landmark_var.true_z:.{translation_fprec}f}"
        else:
            raise ValueError(f"Unknown landmark type {type(landmark_var)}")

    def _get_pose_prior_string(pose_prior: POSE_PRIOR_TYPES):
        if isinstance(pose_prior, PosePrior2D):
            measurement_values = f"{pose_prior.x:.{translation_fprec}f} {pose_prior.y:.{translation_fprec}f} {pose_prior.theta:.{theta_fprec}f}"
        elif isinstance(pose_prior, PosePrior3D):
            qx, qy, qz, qw = pose_prior.quat
            measurement_values = f"{pose_prior.x:.{translation_fprec}f} {pose_prior.y:.{translation_fprec}f} {pose_prior.z:.{translation_fprec}f} {qx:.{quaternion_fprec}f} {qy:.{quaternion_fprec}f} {qz:.{quaternion_fprec}f} {qw:.{quaternion_fprec}f}"
        else:
            raise ValueError(f"Unknown pose prior type {type(pose_prior)}")
        measurement_noise = _get_measurement_noise_str_from_covariance_matrix(
            pose_prior.covariance, covariance_fprec
        )
        return f"{pose_prior_type} {pose_prior.timestamp:.{time_fprec}f} {pose_prior.name} {measurement_values} {measurement_noise}"

    def _get_landmark_prior_string(landmark_prior: LANDMARK_PRIOR_TYPES):
        if isinstance(landmark_prior, LandmarkPrior2D):
            measurement_values = f"{landmark_prior.x:.{translation_fprec}f} {landmark_prior.y:.{translation_fprec}f}"
        elif isinstance(landmark_prior, LandmarkPrior3D):
            measurement_values = f"{landmark_prior.x:.{translation_fprec}f} {landmark_prior.y:.{translation_fprec}f} {landmark_prior.z:.{translation_fprec}f}"
        else:
            raise ValueError(f"Unknown landmark prior type {type(landmark_prior)}")
        measurement_noise = _get_measurement_noise_str_from_covariance_matrix(
            landmark_prior.covariance, covariance_fprec
        )
        return f"{landmark_prior_type} {landmark_prior.timestamp:.{time_fprec}f} {landmark_prior.name} {measurement_values} {measurement_noise}"

    def _get_pose_pose_measure_string(rel_pose_pose_measure: POSE_MEASUREMENT_TYPES):
        measurement_connectivity = (
            f"{rel_pose_pose_measure.base_pose} {rel_pose_pose_measure.to_pose}"
        )
        if isinstance(rel_pose_pose_measure, PoseMeasurement2D):
            measurement_values = f"{rel_pose_pose_measure.x:.{translation_fprec}f} {rel_pose_pose_measure.y:.{translation_fprec}f} {rel_pose_pose_measure.theta:.{theta_fprec}f}"
        elif isinstance(rel_pose_pose_measure, PoseMeasurement3D):
            qx, qy, qz, qw = rel_pose_pose_measure.quat
            measurement_values = f"{rel_pose_pose_measure.x:.{translation_fprec}f} {rel_pose_pose_measure.y:.{translation_fprec}f} {rel_pose_pose_measure.z:.{translation_fprec}f} {qx:.{quaternion_fprec}f} {qy:.{quaternion_fprec}f} {qz:.{quaternion_fprec}f} {qw:.{quaternion_fprec}f}"
        else:
            raise ValueError(f"Unknown measurement type {type(rel_pose_pose_measure)}")
        measurement_noise = _get_measurement_noise_str_from_covariance_matrix(
            rel_pose_pose_measure.covariance, covariance_fprec
        )
        if rel_pose_pose_measure.timestamp is None:
            timestamp = 0.0
        else:
            timestamp = rel_pose_pose_measure.timestamp
        return f"{rel_pose_pose_measure_type} {timestamp:.{time_fprec}f} {measurement_connectivity} {measurement_values} {measurement_noise}"

    def _get_pose_landmark_measure_string(
        rel_pose_landmark_measure: POSE_LANDMARK_MEASUREMENT_TYPES,
    ):
        measurement_connectivity = f"{rel_pose_landmark_measure.pose_name} {rel_pose_landmark_measure.landmark_name}"
        if isinstance(rel_pose_landmark_measure, PoseToLandmarkMeasurement2D):
            measurement_values = f"{rel_pose_landmark_measure.x:.{translation_fprec}f} {rel_pose_landmark_measure.y:.{translation_fprec}f}"
        elif isinstance(rel_pose_landmark_measure, PoseToLandmarkMeasurement3D):
            measurement_values = f"{rel_pose_landmark_measure.x:.{translation_fprec}f} {rel_pose_landmark_measure.y:.{translation_fprec}f} {rel_pose_landmark_measure.z:.{translation_fprec}f}"
        else:
            raise ValueError(
                f"Unknown measurement type {type(rel_pose_landmark_measure)}"
            )
        measurement_noise = _get_measurement_noise_str_from_covariance_matrix(
            rel_pose_landmark_measure.covariance, covariance_fprec
        )
        return f"{rel_pose_landmark_measure_type} {rel_pose_landmark_measure.timestamp:.{time_fprec}f} {measurement_connectivity} {measurement_values} {measurement_noise}"

    def _get_range_measure_string(range_measure: FGRangeMeasurement):
        # make sure that the variance is non-zero after rounding to
        # the desired precision
        rounded_variance = round(range_measure.variance, covariance_fprec)
        if rounded_variance == 0.0:
            raise ValueError(
                f"Range measurement variance is zero after rounding to {covariance_fprec} decimal places. "
                "This will likely cause numerical issues. We suggest increasing the variance."
                f"Variance before rounding: {range_measure.variance} Variance after rounding: {rounded_variance}"
            )
        if range_measure.timestamp is None:
            timestamp = 0.0
        else:
            timestamp = range_measure.timestamp
        return f"{range_measure_type} {timestamp:.{time_fprec}f} {range_measure.first_key} {range_measure.second_key} {range_measure.dist:.{translation_fprec}f} {range_measure.variance:.{covariance_fprec}f}"

    # if base dir does not exist, create it
    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))

    with open(fpath, "w") as f:
        for pose_chain in fg.pose_variables:
            for pose in pose_chain:
                f.write(_get_pose_var_string(pose) + "\n")

        for landmark in fg.landmark_variables:
            f.write(_get_landmark_var_string(landmark) + "\n")

        for pose_prior in fg.pose_priors:
            f.write(_get_pose_prior_string(pose_prior) + "\n")

        for landmark_prior in fg.landmark_priors:
            f.write(_get_landmark_prior_string(landmark_prior) + "\n")

        for odom_chain in fg.odom_measurements:
            for odom in odom_chain:
                f.write(_get_pose_pose_measure_string(odom) + "\n")

        for loop_closure in fg.loop_closure_measurements:
            if isinstance(loop_closure, (PoseMeasurement2D, PoseMeasurement3D)):
                f.write(_get_pose_pose_measure_string(loop_closure) + "\n")
            elif isinstance(
                loop_closure, (PoseToLandmarkMeasurement2D, PoseToLandmarkMeasurement3D)
            ):
                f.write(_get_pose_landmark_measure_string(loop_closure) + "\n")
            else:
                raise ValueError(f"Unknown measurement type {type(loop_closure)}")

        for pose_landmark_measure in fg.pose_landmark_measurements:
            f.write(_get_pose_landmark_measure_string(pose_landmark_measure) + "\n")

        for range_measure in fg.range_measurements:
            f.write(_get_range_measure_string(range_measure) + "\n")

    logger.info(f"Saved factor graph in PyFG text format to {fpath}")


def read_from_pyfg_text(fpath: str) -> FactorGraphData:
    # assume that dimension is consistent

    # quickly read the first line to get the dimension
    f_temp = open(fpath, "r")

    def _get_dim_from_first_line(line: str) -> int:
        if line.startswith(POSE_TYPE_2D):
            return 2
        elif line.startswith(POSE_TYPE_3D):
            return 3
        else:
            raise ValueError(f"Unknown pose type {line}")

    dim = _get_dim_from_first_line(f_temp.readline())
    f_temp.close()

    # determine expected line first word based on dim
    (
        pose_var_type,
        landmark_var_type,
        pose_prior_type,
        landmark_prior_type,
        rel_pose_pose_measure_type,
        rel_pose_landmark_measure_type,
        range_measure_type,
    ) = _get_pyfg_types(dim)

    pose_state_dim = 3 if dim == 2 else 7

    def _get_pose_var_from_line(line: str) -> POSE_VARIABLE_TYPES:
        line_parts = line.split(" ")
        pose_var_timestamp = float(line_parts[1])
        pose_var_name = line_parts[2]
        if dim == 2:
            assert line_parts[0] == POSE_TYPE_2D
            assert len(line_parts) == 6
            x, y, theta = [float(x) for x in line_parts[-pose_state_dim:]]
            return PoseVariable2D(
                name=pose_var_name,
                true_position=(x, y),
                true_theta=theta,
                timestamp=pose_var_timestamp,
            )
        elif dim == 3:
            assert line_parts[0] == POSE_TYPE_3D
            assert len(line_parts) == 10
            x, y, z, qx, qy, qz, qw = [float(x) for x in line_parts[-pose_state_dim:]]
            quat = np.array([qx, qy, qz, qw])
            rot_mat = get_rotation_matrix_from_quat(quat)
            return PoseVariable3D(
                name=pose_var_name,
                true_position=(x, y, z),
                true_rotation=rot_mat,
                timestamp=pose_var_timestamp,
            )
        else:
            raise ValueError(f"Unknown dimension {dim}")

    def _get_landmark_var_from_line(line: str) -> LANDMARK_VARIABLE_TYPES:
        line_parts = line.split(" ")
        landmark_var_name = line_parts[1]
        if dim == 2:
            assert line_parts[0] == LANDMARK_TYPE_2D
            assert len(line_parts) == 4
            x, y = [float(x) for x in line_parts[2:]]
            return LandmarkVariable2D(name=landmark_var_name, true_position=(x, y))
        elif dim == 3:
            assert line_parts[0] == LANDMARK_TYPE_3D
            assert len(line_parts) == 5
            x, y, z = [float(x) for x in line_parts[2:]]
            return LandmarkVariable3D(name=landmark_var_name, true_position=(x, y, z))
        else:
            raise ValueError(f"Unknown dimension {dim}")

    prior_metadata_dim = 3
    num_pose_prior_measure_entries = 3 if dim == 2 else 7
    pose_prior_measurement_dim = 3 if dim == 2 else 6
    pose_prior_measure_noise_dim = _num_elems_symmetric_matrix(
        pose_prior_measurement_dim
    )

    def _get_pose_prior_from_line(line: str) -> POSE_PRIOR_TYPES:
        line_parts = line.split(" ")
        prior_timestamp = float(line_parts[1])
        prior_name = line_parts[2]
        measurement = [
            float(x)
            for x in line_parts[
                prior_metadata_dim : prior_metadata_dim + num_pose_prior_measure_entries
            ]
        ]
        covar_elements = [
            float(x)
            for x in line_parts[prior_metadata_dim + num_pose_prior_measure_entries :]
        ]
        assert len(covar_elements) == pose_prior_measure_noise_dim
        covar_mat = get_symmetric_matrix_from_list_column_major(
            covar_elements, pose_prior_measurement_dim
        )
        (
            trans_precision,
            rot_precision,
        ) = get_measurement_precisions_from_covariance_matrix(
            covar_mat, matrix_dim=pose_prior_measurement_dim
        )
        if dim == 2:
            x, y, theta = measurement
            pose_prior_2d = PosePrior2D(
                name=prior_name,
                position=(x, y),
                theta=theta,
                translation_precision=trans_precision,
                rotation_precision=rot_precision,
                timestamp=prior_timestamp,
            )
            return pose_prior_2d
        elif dim == 3:
            x, y, z, qx, qy, qz, qw = measurement
            rotation = get_rotation_matrix_from_quat(np.array([qx, qy, qz, qw]))
            pose_prior_3d = PosePrior3D(
                name=prior_name,
                position=(x, y, z),
                rotation=rotation,
                translation_precision=trans_precision,
                rotation_precision=rot_precision,
                timestamp=prior_timestamp,
            )
            return pose_prior_3d
        else:
            raise ValueError(f"Unknown dimension {dim}")

    landmark_prior_metadata_dim = 3
    num_landmark_prior_measure_entries = dim
    landmark_prior_measurement_dim = dim
    landmark_prior_measure_noise_dim = _num_elems_symmetric_matrix(
        landmark_prior_measurement_dim
    )

    def _get_landmark_prior_from_line(
        line: str,
    ) -> LANDMARK_PRIOR_TYPES:
        line_parts = line.split(" ")
        prior_timestamp = float(line_parts[1])
        prior_name = line_parts[2]
        measurement = [
            float(x)
            for x in line_parts[
                landmark_prior_metadata_dim : landmark_prior_metadata_dim
                + num_landmark_prior_measure_entries
            ]
        ]
        covar_elements = [
            float(x)
            for x in line_parts[
                landmark_prior_metadata_dim + num_landmark_prior_measure_entries :
            ]
        ]
        assert len(covar_elements) == landmark_prior_measure_noise_dim
        covar_mat = get_symmetric_matrix_from_list_column_major(
            covar_elements, landmark_prior_measurement_dim
        )
        trans_precision = dim / (np.trace(covar_mat))
        if dim == 2:
            x, y = measurement
            landmark_prior_2d = LandmarkPrior2D(
                name=prior_name,
                position=(x, y),
                translation_precision=trans_precision,
                timestamp=prior_timestamp,
            )
            return landmark_prior_2d
        elif dim == 3:
            x, y, z = measurement
            landmark_prior_3d = LandmarkPrior3D(
                name=prior_name,
                position=(x, y, z),
                translation_precision=trans_precision,
                timestamp=prior_timestamp,
            )
            return landmark_prior_3d
        else:
            raise ValueError(f"Unknown dimension {dim}")

    measurement_metadata_dim = 4
    num_trans_and_rot_entries = 3 if dim == 2 else 7
    pose_pose_measure_dim = 3 if dim == 2 else 6
    pose_pose_measure_noise_dim = _num_elems_symmetric_matrix(pose_pose_measure_dim)

    def _get_pose_pose_measure_from_line(line: str) -> POSE_MEASUREMENT_TYPES:
        line_parts = line.split(" ")
        measure_timestamp = float(line_parts[1])
        base_pose_name = line_parts[2]
        to_pose_name = line_parts[3]
        measurement = [
            float(x)
            for x in line_parts[
                measurement_metadata_dim : measurement_metadata_dim
                + num_trans_and_rot_entries
            ]
        ]
        covar_elements = [
            float(x)
            for x in line_parts[measurement_metadata_dim + num_trans_and_rot_entries :]
        ]
        assert len(covar_elements) == pose_pose_measure_noise_dim
        covar_mat = get_symmetric_matrix_from_list_column_major(
            covar_elements, pose_pose_measure_dim
        )
        (
            trans_precision,
            rot_precision,
        ) = get_measurement_precisions_from_covariance_matrix(
            covar_mat, matrix_dim=pose_pose_measure_dim
        )
        expected_num_line_parts = (
            measurement_metadata_dim
            + num_trans_and_rot_entries
            + pose_pose_measure_noise_dim
        )
        assert (
            len(line_parts) == expected_num_line_parts
        ), f"Line has {len(line_parts)} components but expected {expected_num_line_parts} for pose-pose measure: {line_parts}"
        if dim == 2:
            assert line_parts[0] == REL_POSE_POSE_TYPE_2D
            x, y, theta = measurement
            return PoseMeasurement2D(
                base_pose=base_pose_name,
                to_pose=to_pose_name,
                x=x,
                y=y,
                theta=theta,
                translation_precision=trans_precision,
                rotation_precision=rot_precision,
                timestamp=measure_timestamp,
            )
        elif dim == 3:
            assert line_parts[0] == REL_POSE_POSE_TYPE_3D
            x, y, z, qx, qy, qz, qw = measurement
            translation = np.array([x, y, z])
            quat = np.array([qx, qy, qz, qw])
            rot_mat = get_rotation_matrix_from_quat(quat)
            return PoseMeasurement3D(
                base_pose=base_pose_name,
                to_pose=to_pose_name,
                translation=translation,
                rotation=rot_mat,
                translation_precision=trans_precision,
                rotation_precision=rot_precision,
                timestamp=measure_timestamp,
            )
        else:
            raise ValueError(f"Unknown dimension {dim}")

    pose_landmark_measure_dim = 2 if dim == 2 else 3
    pose_landmark_measure_noise_dim = _num_elems_symmetric_matrix(
        pose_landmark_measure_dim
    )

    def _get_pose_landmark_measure_from_line(
        line: str,
    ) -> POSE_LANDMARK_MEASUREMENT_TYPES:
        line_parts = line.split(" ")
        assert (
            len(line_parts)
            == 4 + pose_landmark_measure_dim + pose_landmark_measure_noise_dim
        )
        measure_timestamp = float(line_parts[1])
        pose_name = line_parts[2]
        landmark_name = line_parts[3]
        measurement = [
            float(x)
            for x in line_parts[
                measurement_metadata_dim : measurement_metadata_dim
                + pose_landmark_measure_dim
            ]
        ]
        covar_elements = [
            float(x)
            for x in line_parts[measurement_metadata_dim + pose_landmark_measure_dim :]
        ]
        assert len(covar_elements) == pose_landmark_measure_noise_dim
        covar_mat = get_symmetric_matrix_from_list_column_major(
            covar_elements, pose_landmark_measure_dim
        )
        # augment the covariance matrix with identity matrix on block diagonal
        # to be same dim as pose_pose_measure
        augmented_covar_mat = np.eye(pose_pose_measure_dim)
        augmented_covar_mat[
            :pose_landmark_measure_dim, :pose_landmark_measure_dim
        ] = covar_mat
        (
            trans_precision,
            rot_precision,
        ) = get_measurement_precisions_from_covariance_matrix(
            augmented_covar_mat, matrix_dim=pose_pose_measure_dim
        )
        if dim == 2:
            assert line_parts[0] == REL_POSE_LANDMARK_TYPE_2D
            x, y = measurement
            return PoseToLandmarkMeasurement2D(
                pose_name=pose_name,
                landmark_name=landmark_name,
                x=x,
                y=y,
                translation_precision=trans_precision,
                timestamp=measure_timestamp,
            )
        elif dim == 3:
            assert line_parts[0] == REL_POSE_LANDMARK_TYPE_3D
            x, y, z = measurement
            return PoseToLandmarkMeasurement3D(
                pose_name=pose_name,
                landmark_name=landmark_name,
                x=x,
                y=y,
                z=z,
                translation_precision=trans_precision,
                timestamp=measure_timestamp,
            )
        else:
            raise ValueError(f"Unknown dimension {dim}")

    def _get_range_measure_from_line(line: str) -> FGRangeMeasurement:
        line_parts = line.split(" ")
        assert len(line_parts) == 6
        assert line_parts[0] == RANGE_MEASURE_TYPE
        measure_timestamp = float(line_parts[1])
        association = (line_parts[2], line_parts[3])
        dist = float(line_parts[4])
        variance = float(line_parts[5])
        return FGRangeMeasurement(
            association=association,
            dist=dist,
            stddev=math.sqrt(variance),
            timestamp=measure_timestamp,
        )

    def _rel_pose_pose_is_odom(measure: POSE_MEASUREMENT_TYPES) -> bool:
        base_pose_robot_idx = get_robot_idx_from_frame_name(measure.base_pose)
        to_pose_robot_idx = get_robot_idx_from_frame_name(measure.to_pose)
        if not base_pose_robot_idx == to_pose_robot_idx:
            return False

        base_pose_time_idx = get_time_idx_from_frame_name(measure.base_pose)
        to_pose_time_idx = get_time_idx_from_frame_name(measure.to_pose)
        if not base_pose_time_idx + 1 == to_pose_time_idx:
            return False

        return True

    pyfg = None
    f = open(fpath, "r")

    # iterate over lines
    for line in f:
        if pyfg is None:
            dim = _get_dim_from_first_line(line)
            pyfg = FactorGraphData(dimension=dim)

        line = line.strip()
        line_type = line.split(" ")[0]

        if line_type == pose_var_type:
            pose_var = _get_pose_var_from_line(line)
            pyfg.add_pose_variable(pose_var)
        elif line_type == landmark_var_type:
            landmark_var = _get_landmark_var_from_line(line)
            pyfg.add_landmark_variable(landmark_var)
        elif line_type == pose_prior_type:
            pose_prior = _get_pose_prior_from_line(line)
            pyfg.add_pose_prior(pose_prior)
        elif line_type == landmark_prior_type:
            landmark_prior = _get_landmark_prior_from_line(line)
            pyfg.add_landmark_prior(landmark_prior)
        elif line_type == rel_pose_pose_measure_type:
            pose_measure = _get_pose_pose_measure_from_line(line)
            # check if is odom (i.e., the first char is the same and the remainder of the names has a difference of 1)
            if _rel_pose_pose_is_odom(pose_measure):
                robot_idx = get_robot_idx_from_frame_name(pose_measure.base_pose)
                pyfg.add_odom_measurement(robot_idx, pose_measure)
            else:
                pyfg.add_loop_closure(pose_measure)
        elif line_type == rel_pose_landmark_measure_type:
            pose_landmark_measure = _get_pose_landmark_measure_from_line(line)
            pyfg.add_pose_landmark_measurement(pose_landmark_measure)
        elif line_type == range_measure_type:
            range_measure = _get_range_measure_from_line(line)
            pyfg.add_range_measurement(range_measure)
        else:
            raise ValueError(f"Unknown PyFG type {type(line_type)}")

    f.close()

    assert isinstance(pyfg, FactorGraphData)
    logger.info(f"Loaded factor graph in PyFG text format from {fpath}")
    return pyfg
