import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import BSpline

from utils.bspline import BSpline
from utils.gradient_descent import plot_bspline


def calculate_integral_1d(f_values):
    n = len(f_values)
    integral_values = [0] * n  # Initialize with zeros

    # Use the trapezoidal rule to calculate the integral values
    for i in range(1, n):
        x1, x2 = i - 1, i
        area = (x2 - x1) * (f_values[i - 1] + f_values[i]) / 2
        integral_values[i] = integral_values[i - 1] + area

    return integral_values


def do_impl():
    bspline = BSpline(num_pts=7, degree=3, num_T_pts=100)
    control_points_dt = np.array([1., 2., 3., 2., 1., 2., 1.])
    control_points_t = calculate_integral_1d(control_points_dt)
    control_points_q = np.array([2., 2., 4., 4., 2., 2., 3.])

    plot_bspline(bspline, control_points_dt)
    plot_bspline(bspline, control_points_t)
    plot_bspline(bspline, control_points_t)
    plot_bspline(bspline, control_points_q)

    plt.plot(np.dot(bspline.N[0], control_points_t), np.dot(bspline.N[0], control_points_q))
    plt.show()


def do_impl2():
    control_points_t = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    control_points_q = np.array([2., 2., 4., 4., 2., 2., -3., 1., 1., 1.])
    d = 3
    m = d + len(control_points_q)
    t = np.pad(np.linspace(0., 1., m + 1 - 2 * d), d, 'edge')

    bspline_q = scipy.interpolate.BSpline(t, control_points_q, k=d)
    bspline_t = scipy.interpolate.BSpline(t, control_points_t, k=d)

    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, len(control_points_q))
    plt.plot(x, bspline_q(x))
    plt.scatter(y, control_points_q)
    plt.show()

    plt.plot(x, bspline_q.derivative(1)(x))
    plt.scatter(y, control_points_q)
    plt.show()

    plt.plot(x, bspline_t(x))
    plt.scatter(y, control_points_t)
    plt.show()

    plt.plot(bspline_t(x), bspline_q(x))
    plt.scatter(control_points_t, control_points_q)
    plt.show()


if __name__ == "__main__":
    do_impl2()
