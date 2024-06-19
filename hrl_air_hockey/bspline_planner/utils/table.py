import matplotlib.pyplot as plt
from hrl_air_hockey.bspline_planner.utils.constants import TableConstraint


class Table:
    def __init__(self):
        self.x_min = TableConstraint.XMin
        self.y_min = TableConstraint.YMin
        self.x_max = TableConstraint.XMax
        self.y_max = TableConstraint.YMax
        self.z = TableConstraint.Z

    def plot(self):
        plt.plot([self.x_min, self.x_min, self.x_max, self.x_max, self.x_min],
                 [self.y_min, self.y_max, self.y_max, self.y_min, self.y_min], 'b')

    def inside(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
