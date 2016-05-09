from __future__ import print_function, division

import os
import inspect
import pytest
import matplotlib.pyplot as plt

import numpy as np
import numpy.testing as npt
from obspy.geodetics import locations2degrees

from spaceweight import SpherePoint
from spaceweight.sphereweightbase import SphereDistRel

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(
     inspect.currentframe())))


def parse_station_file(filename):
    pts = []
    with open(filename) as fh:
        content = fh.readlines()

    for line in content:
        info = line.split()
        sta = info[0]
        nw = info[1]
        tag = "%s_%s" % (nw, sta)
        lat = float(info[2])
        lon = float(info[3])
        pt = SpherePoint(lat, lon, tag)
        pts.append(pt)

    return pts


@pytest.fixture
def lpoints():
    POINTSFILE = os.path.join(CURRENTDIR, "data", "STATIONS")
    return parse_station_file(POINTSFILE)


@pytest.fixture
def spoints():
    return [SpherePoint(10, 10, "point1"),
            SpherePoint(10, -10, "point2"),
            SpherePoint(-10, -10, "point3"),
            SpherePoint(-10, 10, "point4")]


def test_build_distance_matrix(spoints):
    obj = SphereDistRel(spoints, normalize_mode="average")
    dist_m = obj._build_distance_matrix()

    d1 = 20
    d2 = locations2degrees(10, 10, 10, -10)
    d3 = locations2degrees(10, 10, -10, -10)

    true_m = np.array([[0, d2, d3, d1],
                       [d2, 0, d1, d3],
                       [d3, d1, 0, d2],
                       [d1, d3, d2, 0]])
    npt.assert_allclose(dist_m, true_m)

    obj.calculate_weight(10.0)
    npt.assert_allclose(obj.points_weights, [1.0, 1.0, 1.0, 1.0])


def test_plot_exp_matrix(spoints, tmpdir):
    obj = SphereDistRel(spoints, normalize_mode="average")
    obj.calculate_weight(20.0)

    # Force agg backend.
    plt.switch_backend('agg')

    figname = os.path.join(str(tmpdir), "exp_matrix.png")
    obj.plot_exp_matrix(figname)


def test_transfer_dist_to_weight(spoints):
    obj = SphereDistRel(spoints)
    dist_m = np.array([[0, 2], [2, 0]])
    weight, exp_matrix = obj._transfer_dist_to_weight(dist_m, 2.0)

    v1 = 1.0 / (1.0 / np.e + 1)
    npt.assert_allclose(weight, [v1, v1])

    v1 = 1.0 / np.e
    npt.assert_allclose(exp_matrix, [[1, v1], [v1, 1]])
