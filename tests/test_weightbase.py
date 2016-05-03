from __future__ import print_function, division

import os
import inspect
import pytest

import numpy as np
from spaceweight import Point
from spaceweight.weightbase import WeightBase

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(
     inspect.currentframe())))
POINTSFILE = os.path.join(CURRENTDIR, "data", "STATIONS")


@pytest.fixture
def points():
    pts = []
    with open(POINTSFILE) as fh:
        content = fh.readlines()

    for line in content:
        info = line.split()
        sta = info[0]
        nw = info[1]
        tag = "%s_%s" % (nw, sta)
        lat = float(info[2])
        lon = float(info[3])
        coord = np.array([lat, lon])
        pt = Point(coord, tag)
        pts.append(pt)

    return pts


def test_weightbase(points):
    WeightBase(points, sort_by_tag=True, remove_duplicate=True)
