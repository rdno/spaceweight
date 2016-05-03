from __future__ import print_function, division

import os
import inspect
import pytest

from spaceweight import SpherePoint
from spaceweight.sphereweight import SphereWeightBase
from spaceweight import SphereDistRel, SphereAziBin
from spaceweight import SphereAziRel, SphereVoronoi


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
        pt = SpherePoint(lat, lon, tag)
        pts.append(pt)

    return pts


@pytest.fixture
def center():
    return SpherePoint(latitude=0.0, longitude=0.0, tag="center")


def test_sphereweightbase(points, center):
    SphereWeightBase(points, center)


def test_spheredistrel(points, center):
    ref_dist = 1.0
    weight = SphereDistRel(points, ref_distance=ref_dist, center=center)
    weight.calculate_weight()


def test_spherevoronoi(points, center):

    order = 1.0
    weight = SphereVoronoi(points, voronoi_order=order, center=center)
    weight.calculate_weight()


def test_sphereazibin(points, center):

    bin_order = 0.5
    nbins = 12
    weight = SphereAziBin(points, center=center, bin_order=bin_order,
                          nbins=nbins)
    weight.calculate_weight()


def test_sphereazirel(points, center, ref_azi=5.0):

    weight = SphereAziRel(points, center=center, ref_azimuth=ref_azi)
    weight.calculate_weight()
