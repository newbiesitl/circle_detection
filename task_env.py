import numpy as np
from shapely.geometry import Point
from skimage.draw import circle_perimeter_aa
from global_config import INPUT_SHAPE, RADIUS
# from keras import backend as K
# import tensorflow as tf


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise, return_original=False):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)
    # plt.imshow(img)
    # plt.show()
    # plt.clf();plt.cla()
    # Noise
    noised = img + noise * np.random.rand(*img.shape)
    if return_original:
        return (row, col, rad), (img, noised)
    return (row, col, rad), noised




def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)
    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def get_samples(n=200, norm=False, return_original=True, noise_lvl=2):
    size = INPUT_SHAPE[0]
    for _ in range(n):
        params, img = noisy_circle(size, RADIUS, noise_lvl, return_original=return_original)
        params = list(params)
        if norm:
            params[0] = params[0] / size
            params[1] = params[1] / size
            params[2] = params[2] / RADIUS
        yield img, list(params)
