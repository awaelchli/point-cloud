from cloud import *
import matplotlib.pyplot as plt
import colorsys
from math import pi
import numpy as np
import os


def random_colors(n):
    colors = np.zeros((n, 3))
    for i, hue in enumerate(np.arange(0, 1, 1 / n)):
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1.0)
        colors[i, 0] = r
        colors[i, 1] = g
        colors[i, 2] = b

    return colors


def main():
    num_classes = 100
    num_points = 100
    num_clouds = 100
    seq_length = 10
    camera_z = 10
    focal_length = 5
    save = True
    left, right = -pi / 2, pi / 2
    output = 'data/clouds/'

    labels = np.zeros((num_clouds, seq_length), dtype=np.long)
    digitizer = RangeDigitizer1D(left, right, num_classes)

    for c in range(num_clouds):
        pc = SpherePointCloud3D(num_points)
        points = pc.get_points()
        transformer = RotateY(points)

        jump = iter(Jump1D(left, right, 0))
        angles = []
        colors = random_colors(num_points)

        cloud_folder = os.path.join(output, f'{c:04d}')
        if not os.path.exists(cloud_folder):
            os.mkdir(cloud_folder)

        for i in range(seq_length):
            angle = next(jump)
            angles += [angle]
            points = transformer.transform(angle)

            points[2, :] += camera_z
            points2D = project_to_screen(points, focal_length)

            plt.scatter(points2D[0], points2D[1],
                        c=colors,
                        marker='+')

            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis([-1, 1, -1, 1])

            if save:
                filename = os.path.join(cloud_folder, f'{i:04d}.png')
                plt.savefig(filename)
            else:
                plt.pause(0.01)

            plt.clf()

        labels[c] = digitizer.classify(angles)

    # Write labels to file
    with open(os.path.join(output, 'labels.txt'), 'w') as f:
        for c in range(len(labels)):
            strings = [str(x) for x in labels[c]]
            line = ' '.join(strings)
            f.write(line + '\n')


if __name__ == '__main__':
    main()
