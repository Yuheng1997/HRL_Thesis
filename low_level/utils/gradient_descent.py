import numpy as np
import matplotlib.pyplot as plt

from utils.bspline import BSpline


def plot_bspline(bspline, control_points):
    # plot example
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    x = np.linspace(0, 1, bspline.num_T_pts)
    axes[0].scatter(np.linspace(0, 1, len(control_points)), control_points)
    axes[0].plot(x, np.dot(bspline.N[0], control_points), 'r-', label='trajectory')
    axes[1].plot(x, np.dot(bspline.dN[0], control_points), 'g-', label='velocity')
    axes[2].plot(x, np.dot(bspline.ddN[0], control_points), 'b-', label='acceleration')

    axes[0].set_title('Plot 1')
    axes[1].set_title('Plot 2')
    axes[2].set_title('Plot 3')

    axes[0].set_xlabel('x')
    axes[1].set_xlabel('x')
    axes[2].set_xlabel('x')

    axes[0].set_ylabel('y')
    axes[1].set_ylabel('y')
    axes[2].set_ylabel('y')

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    plt.tight_layout()

    plt.show()


def calculate_boundaries(q_i, qd_i, qdd_i, q_f, qd_f, d_p, c_p, d_r, cp_r):
    """
    Calculates the first three and last two control points to satisfy the start and end configuration.
    :param q_i: initial position
    :param qd_i: initial velocity
    :param qdd_i: initial acceleration
    :param q_f: final position
    :param qd_f: final velocity
    :param d_p: degree of path b-spline p
    :param c_p: number of control points on path b-spline p
    :param d_r: degree of time b-spline r
    :param cp_r: array of control points of time b-spline r
    :return: Two numpy array representing the first three and the last two control points
    """
    # calculate eta_p, beta_p and eta_r
    eta_p = d_p * (c_p - d_p)**2
    beta_p = ((d_p * (d_p - 1)) / 2) * (c_p - d_p)**2
    eta_r = d_r * (len(cp_r) - d_r)**2

    # calculate first three control points
    start, end = np.zeros(3), np.zeros(2)
    start[0] = q_i
    start[1] = start[0] + qd_i / (cp_r[0] * eta_p)
    K = (qdd_i - eta_p * eta_r * (start[1] - start[0]) * (cp_r[1] - cp_r[0]) * cp_r[0]) / (cp_r[0])**2
    start[2] = (K - 2 * beta_p * start[0] + 3 * beta_p * start[1]) / beta_p

    # calculate last two control points
    end[-1] = q_f
    end[-2] = end[-1] - (qd_f / (cp_r[-1] * eta_p))

    return start, end


def generate_bspline(q_i, qd_i, qdd_i, q_f, qd_f, degree, num_cp_pts):
    # generate time b-spline
    d_r = degree
    c_r = num_cp_pts
    cp_r = np.arange(1, c_r+1, 1)

    # generate configuration b-spline
    d_p = degree
    c_p = num_cp_pts
    p = BSpline(c_p, d_p)

    # calculate first and last control points
    start, end = calculate_boundaries(q_i, qd_i, qdd_i, q_f, qd_f, d_p, c_p, d_r, cp_r)

    # fill control points between
    fill = np.linspace(start=q_i, stop=q_f, num=num_cp_pts-5)
    noise = np.random.normal(0, .5, size=num_cp_pts-5)
    cp_p = np.concatenate((start, fill + noise, end))

    return p, cp_p


def generate_example():
    # set initial and final parameters of trajectory
    q_i = 0
    qd_i = 1
    qdd_i = 2
    q_f = 4
    qd_f = 2

    # set parameters for b-splines
    degree = 3
    num_cp_pts = 20

    # generate example
    spl, cp_p = generate_bspline(q_i, qd_i, qdd_i, q_f, qd_f, degree, num_cp_pts)

    plot_bspline(spl, cp_p)


def gradient_descent_on_bspline():
    # Create example
    bspline = BSpline(num_pts=15, degree=3, num_T_pts=100)

    control_points = np.array([1., 2., 3., 5., 9., -1., 5., 1., 5., 7., 1., 5., 12., 1., 2.])
    cp_start, cp_mid, cp_end = np.split(control_points, [3, len(control_points)-2])

    plot_bspline(bspline, control_points)

    # parameters for gd
    learning_rate = 0.01
    num_iterations = 10000
    gradient_clip_threshold = 100

    # gradient descent
    for i in range(num_iterations):
        # Compute the gradient
        gradient = np.dot(bspline.ddN[0].T, np.dot(bspline.ddN[0], control_points))
        if np.linalg.norm(gradient) > gradient_clip_threshold:
            gradient = gradient_clip_threshold * gradient / np.linalg.norm(gradient)

        # Update the control points
        control_points[3:-2] -= learning_rate * gradient[3:-2]

        # Calculate the loss value
        loss = (1 / 2) * np.power(np.dot(bspline.ddN[0], control_points), 2)

        # Print the current loss value
        print("Iteration", i + 1, "- Loss:", np.sum(loss))

    plot_bspline(bspline, control_points)


if __name__ == "__main__":
    gradient_descent_on_bspline()
