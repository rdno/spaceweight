from __future__ import print_function, division

import os
import inspect
import pytest

import numpy as np
import numpy.testing as npt

from spaceweight import SpherePoint
from spaceweight.sphereweightbase import SphereWeightBase
from spaceweight.sphereweightbase import _azimuth, _distance


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
    return [SpherePoint(10, 0, "point1"),
            SpherePoint(0, 10, "point2"),
            SpherePoint(-10, 0, "point3"),
            SpherePoint(0, -10, "point4")]


def test_azimuth():
    npt.assert_allclose(_azimuth(0, 0, 10., 0), 0.0)
    npt.assert_allclose(_azimuth(0, 0, 0, 10.), 90.0)
    npt.assert_allclose(_azimuth(0, 0, -10, 0), 180.0)
    npt.assert_allclose(_azimuth(0, 0, 0, -10), 270.0)


def test_distance():
    npt.assert_allclose(_distance(0, 0, 10., 0), 10.0)


def test_calculate_azimuth_array(spoints):
    center = SpherePoint(0, 0, "center")
    obj = SphereWeightBase(spoints, center=center)
    azis = obj._calculate_azimuth_array()
    trues = np.array([0, 90, 180, 270])
    npt.assert_allclose(azis, trues)


def test_calculate_distance_array(spoints):
    center = SpherePoint(0, 0, "center")
    obj = SphereWeightBase(spoints, center=center)
    dists = obj._calculate_distance_array()
    npt.assert_allclose(dists, [10, 10, 10, 10])
