"""A minimal two-parameter gradient descent demo."""

from pathlib import Path

import numpy as np

from gradient_descent_visualization import plot_two_parameter_optimization


def mse_loss_and_gradient(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    theta: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute mean squared error and its gradient."""
    residuals = design_matrix @ theta - targets
    loss = float(np.mean(residuals**2))
    gradient = 2.0 / len(targets) * design_matrix.T @ residuals
    return loss, gradient


def gradient_descent(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    theta_initial: np.ndarray,
    learning_rate: float,
    steps: int,
) -> dict[str, np.ndarray]:
    """Run plain batch gradient descent and return its complete history."""
    theta = theta_initial.astype(float).copy()
    history = {"theta": [], "gradient": [], "loss": []}

    for _ in range(steps + 1):
        loss, gradient = mse_loss_and_gradient(design_matrix, targets, theta)
        history["theta"].append(theta.copy())
        history["gradient"].append(gradient.copy())
        history["loss"].append(loss)
        theta = theta - learning_rate * gradient

    return {
        "step": np.arange(steps + 1),
        "theta": np.asarray(history["theta"]),
        "gradient": np.asarray(history["gradient"]),
        "loss": np.asarray(history["loss"]),
    }


def print_history(history: dict[str, np.ndarray], every: int = 5) -> None:
    """Print a sparse table without changing the optimization process."""
    print("step |   theta_0 |   theta_1 |     grad_0 |     grad_1 |         loss")
    print("-" * 78)
    final_step = int(history["step"][-1])

    for step, theta, gradient, loss in zip(
        history["step"],
        history["theta"],
        history["gradient"],
        history["loss"],
    ):
        if step % every == 0 or step == final_step:
            print(
                f"{step:>4d} | {theta[0]:>9.5f} | {theta[1]:>9.5f} | "
                f"{gradient[0]:>10.5f} | {gradient[1]:>10.5f} | {loss:>12.8f}"
            )


def main() -> None:
    """Fit a noisy straight line and visualize the optimization process."""
    x = np.array([0.0, 0.5, 1.1, 1.8, 2.6, 3.4, 4.3, 5.1])
    noise = np.array([0.35, -0.25, 0.50, -0.45, 0.15, -0.35, 0.40, -0.20])
    targets = 1.2 + 1.7 * x + noise
    design_matrix = np.column_stack([np.ones_like(x), x])

    learning_rate = 0.04
    history = gradient_descent(
        design_matrix=design_matrix,
        targets=targets,
        theta_initial=np.array([-1.5, 0.1]),
        learning_rate=learning_rate,
        steps=70,
    )
    print_history(history, every=5)

    optimum = np.linalg.lstsq(design_matrix, targets, rcond=None)[0]
    loss_function = lambda theta: mse_loss_and_gradient(
        design_matrix, targets, theta
    )[0]
    output_path = (
        Path(__file__).resolve().parent
        / "output"
        / "gradient_descent_two_parameter.png"
    )
    plot_two_parameter_optimization(
        history=history,
        loss_function=loss_function,
        output_path=output_path,
        optimum=optimum,
        parameter_labels=(r"intercept $\theta_0$", r"slope $\theta_1$"),
        title=rf"Two-parameter gradient descent ($\eta={learning_rate}$)",
        record_every=5,
    )

    final_theta = history["theta"][-1]
    print(
        f"\nFinal parameters: theta_0={final_theta[0]:.6f}, "
        f"theta_1={final_theta[1]:.6f}"
    )
    print(f"Final loss: {history['loss'][-1]:.10f}")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
