from math import pi
import random
import numpy as np
import math


def project_to_screen(points, f=1):
    sz = points[2, :]
    points[0, :] = f * points[0, :] / sz
    points[1, :] = f * points[1, :] / sz
    return points[:2, :]


class SpherePointCloud3D:

    def __init__(self, num_points):
        self.num_points = num_points

    def get_points(self):
        points = np.zeros((3, self.num_points))

        # Z coordinate
        z = points[2] = np.random.uniform(-1, 1, self.num_points)

        # X- and Y coordinate
        theta = np.random.uniform(0, 2 * pi, self.num_points)
        points[0] = np.sqrt(1 - z ** 2) * np.cos(theta)
        points[1] = np.sqrt(1 - z ** 2) * np.sin(theta)

        return points


class RangeDigitizer1D:

    def __init__(self, a, b, num_classes):
        self.start = a
        self.end = b
        self.num_classes = num_classes
        self.bins = np.linspace(a, b, num_classes + 1)

    def classify(self, samples):
        digits = np.digitize(samples, self.bins) - 1
        digits = np.clip(digits, 0, self.num_classes - 1)
        return digits


class Bounce1D:

    def __init__(self, left, right, initial_pos, speed=1.0):
        assert left <= initial_pos <= right
        self.left = left
        self.right = right
        self.speed = speed
        self.pos = initial_pos

    def random_speed(self, min, max):
        self.speed = random.uniform(min, max)

    def random_pos(self):
        self.pos = random.uniform(self.left, self.right)

    def __iter__(self):
        yield self.pos
        while True:
            self.pos += self.speed
            if self.pos <= self.left:
                self.pos = self.left
                self.speed *= -1
            elif self.pos >= self.right:
                self.pos = self.right
                self.speed *= -1

            yield self.pos


class Jump1D:

    def __init__(self, left, right, initial_pos):
        self.left = left
        self.right = right
        self.initial_pos = initial_pos

    def __iter__(self):
        yield self.initial_pos
        while True:
            yield random.uniform(self.left, self.right)


class Transformer:

    def __init__(self, point_cloud):
        # Shape of point cloud: (3, n)
        self.point_cloud = point_cloud
        self.initial = point_cloud.copy()

    @property
    def num_points(self):
        return self.point_cloud.shape[1]

    def transform(self, *args, **kwargs):
        pass


class Translate1D(Transformer):

    def __init__(self, point_cloud):
        super().__init__(point_cloud)

    def transform(self, value, axis=0):
        vector = np.zeros((3, 1))
        vector[axis] = value
        delta = np.repeat(vector, self.num_points, axis=1)
        self.point_cloud = self.initial + delta
        return self.point_cloud


class RotateY(Transformer):

    def __init__(self, point_cloud):
        super().__init__(point_cloud)

    def transform(self, angle):
        sin = math.sin(angle)
        cos = math.cos(angle)
        rot = np.array([
            [cos, 0, sin],
            [0, 1, 0],
            [-sin, 0, cos]
        ])
        self.point_cloud = np.matmul(rot, self.initial)
        return self.point_cloud
