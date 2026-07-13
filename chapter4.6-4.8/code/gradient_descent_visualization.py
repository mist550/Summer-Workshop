"""Reusable visualization for optimization in a two-parameter loss space."""

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

LossFunction = Callable[[np.ndarray], float]


def _recorded_indices(length: int, every: int) -> np.ndarray:
    indices = np.arange(0, length, every)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return indices


def _evaluate_surface(
    loss_function: LossFunction,
    parameter_0_values: np.ndarray,
    parameter_1_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameter_0_grid, parameter_1_grid = np.meshgrid(
        parameter_0_values, parameter_1_values
    )
    loss_grid = np.empty_like(parameter_0_grid)

    for row, column in np.ndindex(parameter_0_grid.shape):
        theta = np.array(
            [parameter_0_grid[row, column], parameter_1_grid[row, column]]
        )
        loss_grid[row, column] = loss_function(theta)

    return parameter_0_grid, parameter_1_grid, loss_grid


def plot_two_parameter_optimization(
    history: dict[str, np.ndarray],
    loss_function: LossFunction,
    output_path: str | Path,
    *,
    optimum: np.ndarray | None = None,
    parameter_labels: tuple[str, str] = (r"parameter $\theta_0$", r"parameter $\theta_1$"),
    title: str = "Two-parameter optimization",
    record_every: int = 5,
    surface_resolution: int = 150,
    view_elevation: float = 34,
    view_azimuth: float = 42,
) -> None:
    """Plot a 3D surface, contour path, parameters, and loss history.

    The function only assumes that ``history`` contains ``step``, ``theta`` and
    ``loss`` arrays, and that ``loss_function(theta)`` accepts two parameters.
    """
    steps = np.asarray(history["step"])
    theta_history = np.asarray(history["theta"])
    loss_history = np.asarray(history["loss"])
    if theta_history.ndim != 2 or theta_history.shape[1] != 2:
        raise ValueError("theta history must have shape (number_of_steps, 2)")
    if record_every < 1:
        raise ValueError("record_every must be at least 1")

    recorded = _recorded_indices(len(steps), record_every)
    reference_points = theta_history if optimum is None else np.vstack(
        [theta_history, optimum]
    )
    lower = reference_points.min(axis=0) - np.array([0.8, 0.35])
    upper = reference_points.max(axis=0) + np.array([0.8, 0.35])
    parameter_0_values = np.linspace(lower[0], upper[0], surface_resolution)
    parameter_1_values = np.linspace(lower[1], upper[1], surface_resolution)
    parameter_0_grid, parameter_1_grid, loss_grid = _evaluate_surface(
        loss_function, parameter_0_values, parameter_1_values
    )

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax_surface = fig.add_subplot(grid[0, 0], projection="3d")
    surface = ax_surface.plot_surface(
        parameter_0_grid,
        parameter_1_grid,
        loss_grid,
        cmap="viridis",
        alpha=0.38,
        linewidth=0,
        antialiased=True,
    )
    z_ceiling = max(float(np.percentile(loss_grid, 70)), loss_history[0] * 1.08)
    visible_path_z = loss_history + 0.012 * z_ceiling
    ax_surface.plot(
        theta_history[:, 0],
        theta_history[:, 1],
        visible_path_z,
        color="white",
        linewidth=5,
    )
    ax_surface.plot(
        theta_history[:, 0],
        theta_history[:, 1],
        visible_path_z,
        color="#d62728",
        linewidth=2.6,
        label="gradient descent",
    )
    ax_surface.scatter(
        theta_history[recorded, 0],
        theta_history[recorded, 1],
        visible_path_z[recorded],
        color="#d62728",
        edgecolor="white",
        linewidth=0.7,
        s=28,
    )
    if optimum is not None:
        ax_surface.scatter(
            optimum[0],
            optimum[1],
            loss_function(optimum) + 0.012 * z_ceiling,
            color="black",
            marker="*",
            s=110,
            label="minimum",
        )
    ax_surface.set_xlabel(parameter_labels[0])
    ax_surface.set_ylabel(parameter_labels[1])
    ax_surface.set_zlabel(r"loss $J(\theta)$")
    ax_surface.set_title("3D loss surface and optimization path")
    ax_surface.set_zlim(0, z_ceiling)
    ax_surface.view_init(elev=view_elevation, azim=view_azimuth)
    ax_surface.legend(loc="upper left")
    fig.colorbar(surface, ax=ax_surface, shrink=0.58, pad=0.08, label="loss")

    ax_contour = fig.add_subplot(grid[0, 1])
    positive_losses = loss_grid[loss_grid > 0]
    levels = np.geomspace(positive_losses.min(), positive_losses.max(), 18)
    contours = ax_contour.contour(
        parameter_0_grid,
        parameter_1_grid,
        loss_grid,
        levels=levels,
        cmap="viridis",
        linewidths=1,
    )
    ax_contour.clabel(contours, inline=True, fontsize=7, fmt="%.2g")
    arrow_indices = recorded[recorded < len(steps) - 1]
    updates = theta_history[arrow_indices + 1] - theta_history[arrow_indices]
    ax_contour.quiver(
        theta_history[arrow_indices, 0],
        theta_history[arrow_indices, 1],
        updates[:, 0],
        updates[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#d62728",
        width=0.006,
        label="one-step update",
    )
    ax_contour.plot(
        theta_history[:, 0],
        theta_history[:, 1],
        color="#d62728",
        linewidth=1.5,
    )
    ax_contour.scatter(
        theta_history[recorded, 0],
        theta_history[recorded, 1],
        color="#d62728",
        s=24,
        zorder=3,
    )
    if optimum is not None:
        ax_contour.scatter(
            optimum[0],
            optimum[1],
            color="black",
            marker="*",
            s=110,
            label="minimum",
        )
    label_positions = np.linspace(0, len(recorded) - 1, min(6, len(recorded)), dtype=int)
    for index in recorded[label_positions]:
        ax_contour.annotate(
            f"t={steps[index]}",
            theta_history[index],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax_contour.set_xlabel(parameter_labels[0])
    ax_contour.set_ylabel(parameter_labels[1])
    ax_contour.set_title("2D contour map and optimization path")
    ax_contour.grid(alpha=0.2)
    ax_contour.legend(loc="lower right")

    ax_parameters = fig.add_subplot(grid[1, 0])
    styles = [("o", "#1f77b4"), ("s", "#ff7f0e")]
    for index, (marker, color) in enumerate(styles):
        ax_parameters.plot(
            steps,
            theta_history[:, index],
            marker=marker,
            markevery=recorded.tolist(),
            color=color,
            label=parameter_labels[index],
        )
        if optimum is not None:
            ax_parameters.axhline(
                optimum[index], color=color, linestyle="--", alpha=0.65
            )
    ax_parameters.set_xlabel("iteration t")
    ax_parameters.set_ylabel("parameter value")
    ax_parameters.set_title("Parameter changes over iterations")
    ax_parameters.grid(alpha=0.25)
    ax_parameters.legend()

    ax_loss = fig.add_subplot(grid[1, 1])
    ax_loss.plot(
        steps,
        loss_history,
        color="#2ca02c",
        marker="o",
        markevery=recorded.tolist(),
        linewidth=2,
    )
    ax_loss.fill_between(steps, loss_history, color="#2ca02c", alpha=0.12)
    ax_loss.set_xlabel("iteration t")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss change on a linear y-axis")
    ax_loss.grid(alpha=0.25)
    ax_loss.annotate(
        f"initial: {loss_history[0]:.2f}",
        (steps[0], loss_history[0]),
        xytext=(28, -5),
        textcoords="offset points",
    )
    ax_loss.annotate(
        f"final: {loss_history[-1]:.2e}",
        (steps[-1], loss_history[-1]),
        xytext=(-80, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
    )

    fig.suptitle(title, fontsize=15)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
