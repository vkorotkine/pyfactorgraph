import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from typing import Tuple, Union, Optional, List, Sequence
from evo.tools import file_interface, plot as evoplot
from py_factor_graph.utils.solver_utils import (
    SolverResults,
    save_to_tum,
)
from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D
from py_factor_graph.measurements import FGRangeMeasurement
from py_factor_graph.utils.matrix_utils import get_theta_from_rotation_matrix

COLORS = ["blue", "red", "green", "yellow", "black", "cyan", "magenta"]


def get_color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def draw_arrow(
    ax: plt.Axes,
    x: float,
    y: float,
    theta: float,
    quiver_length: float = 1,
    quiver_width: float = 0.1,
    color: str = "black",
) -> mpatches.FancyArrow:
    """Draws an arrow on the given axes

    Args:
        ax (plt.Axes): the axes to draw the arrow on
        x (float): the x position of the arrow
        y (float): the y position of the arrow
        theta (float): the angle of the arrow
        quiver_length (float, optional): the length of the arrow. Defaults to 0.1.
        quiver_width (float, optional): the width of the arrow. Defaults to 0.01.
        color (str, optional): color of the arrow. Defaults to "black".

    Returns:
        mpatches.FancyArrow: the arrow
    """
    dx = quiver_length * math.cos(theta)
    dy = quiver_length * math.sin(theta)
    return ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=quiver_length,
        head_length=quiver_length,
        width=quiver_width,
        color=color,
    )


def draw_line(
    ax: plt.Axes,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    color: str = "black",
) -> mlines.Line2D:
    """Draws a line on the given axes between the two points

    Args:
        ax (plt.Axes): the axes to draw the arrow on
        x_start (float): the x position of the start of the line
        y_start (float): the y position of the start of the line
        x_end (float): the x position of the end of the line
        y_end (float): the y position of the end of the line
        color (str, optional): color of the arrow. Defaults to "black".

    Returns:
        mpatches.FancyArrow: the arrow
    """
    # if color is grey lets make the line dashed and reduce the line width
    if color == "grey":
        line = mlines.Line2D(
            [x_start, x_end],
            [y_start, y_end],
            color=color,
            linestyle="dashed",
            linewidth=0.5,
        )
    else:
        line = mlines.Line2D([x_start, x_end], [y_start, y_end], color=color)

    ax.add_line(line)
    return line


def draw_circle(ax: plt.Axes, circle: np.ndarray, color="red") -> mpatches.Circle:
    assert circle.size == 3
    return ax.add_patch(
        mpatches.Circle(circle[0:2], circle[2], color=color, fill=False)
    )


def _get_pose_xytheta(
    pose: Union[np.ndarray, PoseVariable2D],
) -> Tuple[float, float, float]:
    assert isinstance(pose, np.ndarray) or isinstance(pose, PoseVariable2D)
    if isinstance(pose, PoseVariable2D):
        x = pose.true_x
        y = pose.true_y
        theta = pose.true_theta
    else:
        x = pose[0, 2]
        y = pose[1, 2]
        theta = get_theta_from_rotation_matrix(pose[0:2, 0:2])
    return x, y, theta


def draw_pose(
    ax: plt.Axes,
    pose: Union[np.ndarray, PoseVariable2D],
    color="blue",
    scale: float = 1,
) -> mpatches.FancyArrow:
    true_x, true_y, true_theta = _get_pose_xytheta(pose)
    return draw_arrow(
        ax,
        true_x,
        true_y,
        true_theta,
        color=color,
        quiver_length=scale,
        quiver_width=scale / 10,
    )


def update_pose_arrow(
    arrow: mpatches.FancyArrow,
    pose: Union[np.ndarray, PoseVariable2D],
    scale: float = 1,
):
    x, y, theta = _get_pose_xytheta(pose)
    quiver_length = scale
    dx = quiver_length * math.cos(theta)
    dy = quiver_length * math.sin(theta)
    arrow.set_data(x=x, y=y, dx=dx, dy=dy)


def draw_traj(
    ax: plt.Axes,
    x_traj: Sequence[float],
    y_traj: Sequence[float],
    color: str = "black",
) -> mlines.Line2D:
    assert len(x_traj) == len(y_traj)
    line = mlines.Line2D(x_traj, y_traj, color=color)
    ax.add_line(line)
    return line


def update_traj(
    line: mlines.Line2D,
    x_traj: Sequence[float],
    y_traj: Sequence[float],
):
    assert len(x_traj) == len(y_traj)
    line.set_xdata(x_traj)
    line.set_ydata(y_traj)


def draw_landmark_variable(ax: plt.Axes, landmark: LandmarkVariable2D):
    true_x = landmark.true_x
    true_y = landmark.true_y
    ax.scatter(true_x, true_y, color="green", marker=(5, 2))


def draw_loop_closure_measurement(
    ax: plt.Axes, base_loc: np.ndarray, to_pose: PoseVariable2D
) -> Tuple[mlines.Line2D, mpatches.FancyArrow]:
    assert base_loc.size == 2

    x_start = base_loc[0]
    y_start = base_loc[1]
    x_end = to_pose.true_x
    y_end = to_pose.true_y

    line = draw_line(ax, x_start, y_start, x_end, y_end, color="green")
    arrow = draw_pose(ax, to_pose)

    return line, arrow


def draw_range_measurement(
    ax: plt.Axes,
    range_measure: FGRangeMeasurement,
    from_pose: PoseVariable2D,
    to_landmark: Union[LandmarkVariable2D, PoseVariable2D],
    add_line: bool = True,
    add_circle: bool = True,
) -> Tuple[Optional[mlines.Line2D], Optional[mpatches.Circle]]:
    base_loc = from_pose.true_x, from_pose.true_y
    to_loc = to_landmark.true_x, to_landmark.true_y

    x_start, y_start = base_loc
    landmark_idx = int(to_landmark.name[1:])
    c = get_color(landmark_idx)
    c = "grey"

    if add_line:
        x_end, y_end = to_loc
        line = draw_line(ax, x_start, y_start, x_end, y_end, color=c)
    else:
        line = None
    if add_circle:
        dist = range_measure.dist
        circle = draw_circle(ax, np.array([x_start, y_start, dist]), color=c)
    else:
        circle = None

    return line, circle


def visualize_solution(
    solution: SolverResults,
    gt_files: Optional[List[str]] = None,
    name: str = "estimate",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualizes the solution.

    Args:
        solution (SolverResults): the solution.
        gt_traj (str): the path to the groundtruth trajectory.
    """
    # save the solution to a temporary .tum file
    temp_file = f"/tmp/{name}.tum"
    soln_tum_files = save_to_tum(solution, temp_file)
    assert gt_files is None or len(gt_files) == len(
        soln_tum_files
    ), f"gt_files: {gt_files}, soln_tum_files: {soln_tum_files}"

    def _get_filename_without_extension(filepath: str) -> str:
        return os.path.splitext(os.path.basename(filepath))[0]

    traj_by_label = {}
    for file_idx in range(len(soln_tum_files)):
        file = soln_tum_files[file_idx]
        traj_est = file_interface.read_tum_trajectory_file(file)

        if gt_files is not None:
            gt_traj_path = gt_files[file_idx]
            gt_traj = file_interface.read_tum_trajectory_file(gt_traj_path)

            # get the traj label from the filename without extension
            traj_label = _get_filename_without_extension(gt_traj_path)
            traj_by_label[traj_label] = gt_traj

            # align the estimated trajectory to the groundtruth
            traj_est.align(gt_traj)

        label_letter = _get_filename_without_extension(file)[-1]
        label = f"{name}_{label_letter}"
        traj_by_label[label] = traj_est

    fig = plt.figure()
    plot_mode = evoplot.PlotMode.xy
    # turn off the background grid and legend
    evoplot.trajectories(fig, traj_by_label, plot_mode=plot_mode)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # hide the legend
    plt.gca().get_legend().remove()

    # hide the grid
    plt.grid(False)

    # set the background to white
    plt.gca().set_facecolor("white")

    # set the background to transparent
    background_transparent = True

    if save_path is not None:
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        # save at a higher resolution
        plt.savefig(
            save_path, transparent=background_transparent, bbox_inches="tight", dpi=300
        )
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()

    plt.close(fig)
